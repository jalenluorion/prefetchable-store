"""The worker's profile arguments for python/serve/dmserve.py.

The defaults themselves (routers and their names rule, Inkling's cross-token names with priority reads, eviction,
the automatic cache size, the prefix cache under <model>/.dmeng/prefix, the saved hot set, drafting off) are resolved
by the engine from the manifest's `profiles` / `defaults` keys with built-in fallbacks (include/dmeng/presets.h,
docs/MANIFEST-PRESETS.md). This file only says which profile and mode to ask for, so the server, dmeng and the C API
cannot disagree."""


def agent_profile(model_dir, drafts=False, reserve_gb=None, profile="auto", mode=None):
    """worker arguments: the profile (auto: the device's), the mode (None: the profile's), drafts, the memory reserve"""
    args = ["--profile", profile]
    if mode:
        args += ["--mode", mode]
    if drafts:
        args += ["--drafts", "1"]
    if reserve_gb is not None:
        args += ["--reserve-gb", str(reserve_gb)]
    return args

