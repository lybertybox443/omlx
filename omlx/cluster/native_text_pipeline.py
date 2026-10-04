"""Native decoder transport and captures; architecture math stays in its model."""
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.pipeline import PipelineMixin

def _is_sliding(layer):
    return bool(getattr(layer, "is_sliding", False) or
                getattr(layer, "attention_type", None) == "sliding_attention")

class NativeTextPipelineMixin(PipelineMixin):
    def pipeline(self, group, split=None):
        super().pipeline(group, split)
        self.pipeline_group = group
        from omlx.cluster.native_capture_pipeline import configure_capture_stage
        self.pipeline_stage = configure_capture_stage(self, group)

    def boundary_capture_output(self):
        from omlx.cluster.native_capture_pipeline import boundary_capture_output
        return boundary_capture_output(self)

    def forward_pipeline(self, h, cache=None, capture_layer_ids=None,
                         hidden_sink=None, return_raw_hidden=False):
        label = type(self).__name__.removesuffix("Model")
        points = sorted(set(capture_layer_ids or []))
        if any(type(i) is not int or not 0 <= i <= self.num_hidden_layers for i in points):
            raise ValueError(f"{label} capture point exceeds decoder layers")
        captured = {0: h} if 0 in points and self.start_idx == 0 else {}
        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(layers)
        if len(cache) != len(layers):
            raise ValueError(f"{label} stage cache does not match its local layers")
        stage = self.pipeline_stage
        boundary = bool(stage is not None and hidden_sink is not None
                        and getattr(self, "_omlx_boundary_captures", False))
        upstream = []
        if self.pipeline_rank < self.pipeline_size - 1:
            if boundary:
                from omlx.cluster.native_capture_pipeline import capture_wire
                count = sum(point <= self.start_idx for point in points)
                h, _, upstream = capture_wire().receive_boundary_captures(
                    stage, h.shape[0], h.shape[1], count)
            else:
                h = mx.distributed.recv_like(h, self.pipeline_rank + 1, group=self.pipeline_group)
        full = next((c for layer, c in zip(layers, cache)
                     if not _is_sliding(layer)), None)
        sliding = next((c for layer, c in zip(layers, cache)
                        if _is_sliding(layer)), None)
        full_mask = create_attention_mask(h, full)
        sliding_mask = (create_attention_mask(h, sliding, window_size=self.args.sliding_window)
                        if any(_is_sliding(layer) for layer in layers) else None)
        for i, (layer, c) in enumerate(zip(layers, cache), self.start_idx):
            mask = sliding_mask if _is_sliding(layer) else full_mask
            h = layer(h, mask=mask, cache=c)
            if i + 1 in points:
                captured[i + 1] = h
        if boundary:
            from omlx.cluster.native_capture_pipeline import capture_wire
            carried = [*upstream, *(captured[i] for i in sorted(captured))]
            if not stage.is_last:
                placeholder = capture_wire().hand_off(stage, h, None, cache, captures=carried)
                return (placeholder, h) if return_raw_hidden else placeholder
            if len(carried) != len(points):
                raise RuntimeError(f"{label} boundary captures do not cover requested layers")
            shared = dict(zip(points, carried))
            hidden_sink.extend(shared[i] for i in capture_layer_ids or [])
            return (self.norm(h), h) if return_raw_hidden else self.norm(h)
        if self.pipeline_rank != 0:
            h = mx.distributed.send(h, self.pipeline_rank - 1, group=self.pipeline_group)
            if cache and cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)
        if self.pipeline_size > 1:
            h = mx.distributed.all_gather(h, group=self.pipeline_group)[:h.shape[0]]
        raw = h
        out = self.norm(raw)
        if hidden_sink is not None:
            if self.pipeline_size > 1 and points:
                # Drain boundary sends before every rank enters capture collectives.
                mx.eval(out)
                packed = mx.stack([captured.get(i, mx.zeros_like(raw)) for i in points])
                shared = mx.distributed.all_sum(packed, group=self.pipeline_group)
                mx.eval(shared)
                captured = dict(zip(points, shared))
            hidden_sink.extend(captured[i] for i in capture_layer_ids or [])
        return (out, raw) if return_raw_hidden else out
