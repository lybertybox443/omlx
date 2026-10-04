"""Coordinator-safe contract for the maintained Laguna text decoder."""
import re
from omlx.cluster.model_adapters import PipelineModelAdapter

_LAYER = re.compile(r"^(?:language_model\.)?(?:model\.)?layers\.(\d+)(?:\.|$)")

class LagunaAdapter(PipelineModelAdapter):
    model_type = "laguna"
    media = ("text",)
    optimizations = ("dflash_enabled",)

    def supports_pipeline(self, config):
        count = config.get("num_hidden_layers")
        return type(count) is int and count >= 2

    def trunk_layer_index(self, tensor_name):
        match = _LAYER.search(tensor_name)
        return int(match.group(1)) if match else None

    def boundary_bytes_per_token(self, config):
        hidden = config.get("hidden_size")
        return hidden * 2 if type(hidden) is int and hidden > 0 else None

    def cache_budget(self, model_path, options):
        import json
        from pathlib import Path
        config = json.loads((Path(model_path) / "config.json").read_text())
        def dim(key):
            value = config.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError("Invalid Laguna cache dimension: " + key)
            return value
        count = dim("num_hidden_layers")
        types = config.get("layer_types") or ["full_attention"] * count
        if len(types) != count or any(t not in ("full_attention", "sliding_attention") for t in types):
            raise ValueError("Laguna cache pattern must match decoder layers")
        width = 8 * dim("num_key_value_heads") * dim("head_dim")
        window = config.get("sliding_window")
        bounded = type(window) is int and window > 0
        copies = 2 if options.get("dflash_enabled") else 1
        return dict(
            layer_kv_bytes_per_token=tuple(0 if bounded and t == "sliding_attention" else copies * width for t in types),
            layer_kv_fixed_bytes=tuple(copies * width * (window + 256) if bounded and t == "sliding_attention" else 0 for t in types),
            kv_cache_step=256,
        )

    def runtime_options(self, config, model_settings):
        from omlx.cluster.dflash import runtime_settings
        return runtime_settings(model_settings)

    def serving(self, model, provider, mlx_server, options):
        from contextlib import contextmanager
        from omlx.cluster.dflash import install_dflash_serving
        from omlx.cluster.mtp_coordination import install_mtp_sampling
        @contextmanager
        def scope():
            with install_dflash_serving(model, mlx_server, options, provider):
                with install_mtp_sampling(model, mlx_server, options):
                    yield
        return scope()

    def prepare_worker(self, model_path, options):
        from types import SimpleNamespace
        from omlx.cluster.dflash import runtime_settings
        expected = set()
        if options.get("dflash_enabled"):
            normalized = runtime_settings(SimpleNamespace(**options))
            if any(options.get(key) != value for key, value in normalized.items()):
                raise ValueError("Invalid Laguna DFlash runtime options")
            expected.update(normalized)
            for key in ("dflash_max_prompt_tokens", "dflash_reserved_bytes"):
                if type(options.get(key)) is not int or options[key] <= 0:
                    raise ValueError("DFlash requires an approved memory reservation")
                expected.add(key)
        if set(options) - expected:
            raise ValueError("Unsupported Laguna distributed runtime options")
        if options.get("dflash_enabled"):
            from omlx.patches.mlx_lm_mtp import batch_generator, cache_rollback
            from omlx.patches.dflash_laguna import install_dflash_laguna_backend
            install_dflash_laguna_backend()
            if not cache_rollback.apply() or not batch_generator.apply():
                raise RuntimeError("Could not install Laguna DFlash generation")
        from . import apply_laguna_patch
        apply_laguna_patch()
        return True

    def resident_layers(self, model):
        from mlx.utils import tree_flatten
        return {index for name, _ in tree_flatten(model.parameters())
                if (index := self.trunk_layer_index(name)) is not None}

ADAPTER = LagunaAdapter()
