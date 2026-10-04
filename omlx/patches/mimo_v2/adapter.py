# SPDX-License-Identifier: Apache-2.0
"""Use the maintained MiMo decoder and native MLX PipelineMixin."""
import json
from contextlib import contextmanager
import re
from pathlib import Path
from omlx.cluster.model_adapters import PipelineModelAdapter

_LAYER = re.compile(r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)")


class MiMoAdapter(PipelineModelAdapter):
    model_type = "mimo_v2"
    media = ("text", "audio")
    required_imports = ("mlx_lm",)
    optimizations = ("mtp_enabled",)

    def supplemental_files(self, model_path):
        root = Path(model_path).expanduser().resolve()
        names = set()
        for sub in ("omnimodal", "audio_tokenizer"):
            base = root / sub
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                try:
                    resolved = path.resolve()
                    if path.is_file() and resolved.is_file() and resolved.is_relative_to(root):
                        names.add(path.relative_to(root).as_posix())
                except (OSError, ValueError):
                    continue
        return tuple(sorted(names))

    def supports_pipeline(self, config):
        count = config.get("num_hidden_layers")
        return type(count) is int and count >= 2

    def trunk_layer_index(self, tensor_name):
        match = _LAYER.search(tensor_name)
        return int(match.group(1)) if match else None

    def boundary_bytes_per_token(self, config):
        width = config.get("hidden_size")
        return 4 * width if type(width) is int and width > 0 else None

    def runtime_options(self, config, model_settings):
        from omlx.cluster.native_mtp_options import native_settings
        return native_settings(model_settings)

    def cache_budget(self, model_path, options):
        config = json.loads((Path(model_path) / "config.json").read_text())
        count = config.get("num_hidden_layers")
        pattern = config.get("hybrid_layer_pattern")
        if type(count) is not int or count < 1 or not isinstance(pattern, list) or len(pattern) != count:
            raise ValueError("MiMo cache pattern must match the layer count")
        if any(type(value) is not int or value not in (0, 1) for value in pattern):
            raise ValueError("Invalid MiMo cache layer pattern")
        def dim(key):
            value = config.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError("Invalid MiMo cache dimension: " + key)
            return value
        full = 4 * dim("num_key_value_heads") * (dim("head_dim") + dim("v_head_dim"))
        sliding = 4 * dim("swa_num_key_value_heads") * (dim("swa_head_dim") + dim("swa_v_head_dim"))
        window = dim("sliding_window_size")
        speculative = bool(options.get("mtp_enabled"))
        copies = 2 if speculative else 1
        profile = dict(layer_kv_bytes_per_token=tuple(0 if value else copies * full for value in pattern),
                       layer_kv_fixed_bytes=tuple(copies * sliding * (window + 256) if value else 0 for value in pattern),
                       kv_cache_step=256)
        if speculative:
            heads = config.get("num_nextn_predict_layers", 0)
            if type(heads) is not int or heads < 1:
                raise ValueError("native MiMo MTP requires checkpoint heads")
            # Persistent history, detached draft, retained prefix, old/new
            # rotating allocations and retained trunk rows on every rank.
            profile["replicated_kv_fixed_bytes"] = (
                4 * heads * sliding * (window + 256)
                + 8 * dim("hidden_size") * (heads + 9))
        return profile

    def prepare_worker(self, model_path, options):
        from omlx.cluster.native_mtp_options import validate_depth
        enabled = options.get("mtp_enabled", False)
        if type(enabled) is not bool:
            raise ValueError("mtp_enabled must be boolean")
        expected = {"mtp_enabled", "mtp_depth"} if enabled else set(options).intersection({"mtp_enabled"})
        if enabled:
            depth = validate_depth(options.get("mtp_depth"))
            if "mtp_adaptive" in options:
                if options["mtp_adaptive"] is not True:
                    raise ValueError("mtp_adaptive must be True when present")
                expected.add("mtp_adaptive")
            config = json.loads((Path(model_path) / "config.json").read_text())
            heads = config.get("num_nextn_predict_layers", 0)
            if type(heads) is not int or heads < 1:
                raise ValueError("checkpoint has no native MiMo MTP heads")
        if set(options) - expected:
            raise ValueError("Unsupported MiMo distributed runtime options")
        from omlx.patches.mimo_v2 import apply_mimo_v2_patch, is_applied
        from omlx.patches.mlx_lm_mtp import set_mtp_active, set_mtp_depth
        set_mtp_active(enabled)
        if enabled:
            set_mtp_depth(depth, fixed=not options.get("mtp_adaptive", False))
            from omlx.patches.mlx_lm_mtp import batch_generator, cache_rollback
            if not batch_generator.apply() or not cache_rollback.apply():
                raise RuntimeError("could not install native MiMo MTP loop")
        apply_mimo_v2_patch()
        if not is_applied():
            raise RuntimeError("could not register the maintained MiMo model")
        import mlx_lm.models.mimo_v2 as module
        module.Model._omlx_adapter = self
        return True

    @contextmanager
    def serving(self, model, provider, mlx_server, options):
        from .audio_serving import install_mimo_audio_serving
        from omlx.cluster.mtp_coordination import install_mtp_sampling
        with install_mtp_sampling(model, mlx_server, options):
            with install_mimo_audio_serving(model, provider, mlx_server):
                yield

    def resident_layers(self, model):
        from mlx.utils import tree_flatten
        return {index for name, _ in tree_flatten(model.parameters())
                if (index := self.trunk_layer_index(name)) is not None}

    def verify_contract(self, model, group):
        import mlx.core as mx
        from omlx.cluster.pipeline_compat import planned_layer_range
        expected = planned_layer_range(model.args.num_hidden_layers, group)
        actual = model.model.start_idx, model.model.end_idx
        bad = expected != actual or self.resident_layers(model) != set(range(*actual))
        if int(mx.distributed.all_sum(mx.array(int(bad)), group=group).item()):
            raise ValueError("MiMo pipeline layer ownership differs from the approved plan")
        if getattr(model, "_omlx_mtp_decode_enabled", False):
            if not model.mtp.layers:
                raise RuntimeError("native MiMo MTP enabled without loaded heads")
            from omlx.cluster.mtp_coordination import MTPRankCoordinator
            object.__setattr__(model, "_omlx_mtp_coordinator", MTPRankCoordinator(group))
            model._omlx_mtp_multi_request = True


ADAPTER = MiMoAdapter()
