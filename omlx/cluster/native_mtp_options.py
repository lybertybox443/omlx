"""Pure validation for checkpoint-native MTP adapters."""
def validate_depth(value):
    if type(value) is not int or not 1 <= value <= 8:
        raise ValueError("native MTP depth must be an integer in 1..8")
    return value


def native_settings(settings):
    enabled = getattr(settings, "mtp_enabled", False)
    if type(enabled) is not bool:
        raise ValueError("mtp_enabled must be boolean")
    if not enabled:
        return {}
    if getattr(settings, "dflash_enabled", False):
        raise ValueError("native MTP and DFlash are mutually exclusive")
    fixed = getattr(settings, "mtp_fixed_depth", None)
    adaptive = getattr(settings, "mtp_adaptive_max_depth", None)
    for value in (fixed, adaptive):
        if value is not None:
            validate_depth(value)
    result = dict(mtp_enabled=True, mtp_depth=validate_depth(fixed or adaptive or 1))
    if adaptive and not fixed:
        result["mtp_adaptive"] = True
    return result


