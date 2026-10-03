# SPDX-License-Identifier: Apache-2.0
"""Let mlx-lm load Qwen4-Exp, so a cluster rank can serve it.

Every cluster rank is an ``mlx_lm.server`` and pinned mlx-lm has no
``qwen4_exp`` module: ``_get_classes`` raises *"Model type qwen4_exp not
supported."* before a weight is read. oMLX already serves the architecture
through its vendored mlx-vlm tree on one Mac; this registers that same
implementation as ``mlx_lm.models.qwen4_exp`` (see ``qwen4_exp_model``) rather
than maintaining a second copy of the model. The upstream maintainers' stance
that mlx-lm should ship its own module is unaffected: this is a local bridge
that disappears the day an official one exists.

Selection and the per-architecture answers the cluster needs live in
``adapter.py``; this module only registers the model with mlx-lm.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_QUALNAME = "mlx_lm.models.qwen4_exp"


def _register_module(qualname: str, file_name: str) -> None:
    """Load a local file as if it were ``qualname``. Idempotent."""

    file_path = Path(__file__).parent / file_name
    existing = sys.modules.get(qualname)
    if existing is not None:
        if getattr(existing, "__file__", None) == str(file_path):
            return
        logger.warning(
            "Replacing %s (%s) with the vendored bridge",
            qualname,
            getattr(existing, "__file__", "?"),
        )

    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    # Registered before execution so imports inside resolve, but a failure must
    # not leave a husk without ``Model`` behind: mlx-lm would fail far away
    # with a confusing AttributeError instead of the real cause.
    sys.modules[qualname] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if existing is None:
            sys.modules.pop(qualname, None)
        else:
            sys.modules[qualname] = existing
        raise
    # ``from mlx_lm.models import x`` reads the package attribute first.
    models_pkg = importlib.import_module("mlx_lm.models")
    setattr(models_pkg, qualname.rsplit(".", 1)[1], module)
    logger.info("Registered %s from %s", qualname, file_name)


def apply_qwen4_exp_mlx_lm_patch() -> bool:
    """Make ``qwen4_exp`` loadable by mlx-lm. Safe to call repeatedly."""

    try:
        _register_module(_QUALNAME, "qwen4_exp_model.py")
    except Exception as exc:
        logger.warning("Could not register %s: %s", _QUALNAME, exc)
        return False
    return True
