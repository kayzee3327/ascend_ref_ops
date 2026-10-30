"""Four reference KDA operators, from two pinned upstreams, in one process.

    from ref_ops import cann, fla_npu

    cann.chunk_kda(...)        cann.recurrent_kda(...)
    fla_npu.chunk_kda(...)     fla_npu.recurrent_kda(...)

Importing this package is what makes all four reachable at once, and it has to
happen **before the first aclnn call in the process** -- which in practice means
before anything that launches an operator, though not necessarily before
``import torch_npu``.

Why that ordering matters, and why "before the first call" rather than "before
process start": ``ACLNN_CMD`` resolves symbols lazily through ``dlsym``, so
setting the environment from Python is early enough. But
``GetCustOpApiHandlers`` opens every entry on ``ASCEND_CUSTOM_OPP_PATH`` and
``LD_LIBRARY_PATH`` and caches the resulting handler list in a function-local
``static``, built on that first call. Whatever is on the path at that moment is
all the process will ever search -- so both vendor roots are named here, or the
second operator is unreachable for the rest of the run.

The paths are **appended to**, never overwritten. ``fla_npu``'s wheel installs a
``.pth`` that prepends its own vendor at interpreter start, before any user code
runs; preserving that entry is what lets the fla arm and the CANN arm coexist
rather than one process per vendor.

The CANN-side operators are renamed copies (``RefOpsCannChunkKdaFwd``,
``RefOpsCannRecurrentKda``) precisely so that coexistence is structural: CANN's
launcher registry is keyed by the op-type string, and both upstreams otherwise
claim ``ChunkKdaFwd`` and ``RecurrentKda``. See ``pins.py`` for the rename and
``build.py`` for how it is applied.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["cann", "fla_npu", "manifest", "OPP_ROOT", "BUILT"]

_PACKAGE_DIR = Path(__file__).resolve().parent

#: Where ``build.py`` installs the CANN vendor packages. Inside the package, so
#: inside the venv: nothing this project builds is written to the CANN system
#: path, and the toolkit's own ``opp/vendors`` stays empty.
OPP_ROOT = _PACKAGE_DIR / "_opp"

_MANIFEST_PATH = _PACKAGE_DIR / "_manifest.json"


def _append_env_path(name: str, value: Path) -> None:
    """Add ``value`` to the end of a ``PATH``-style variable, once."""
    value_str = str(value)
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value_str in parts:
        return
    os.environ[name] = os.pathsep.join([*parts, value_str])


def _expose_vendors() -> list[Path]:
    """Put every installed CANN vendor root on the aclnn dlsym search paths.

    Returns the vendor roots that were found, which is empty before
    ``python -m ref_ops.build`` has run. That is not an error here: the fla arm
    needs none of this, and ``ref_ops.cann`` reports the absence itself with a
    message naming the build command.
    """
    vendors_dir = OPP_ROOT / "vendors"
    if not vendors_dir.is_dir():
        return []

    found = []
    for vendor in sorted(p for p in vendors_dir.iterdir() if p.is_dir()):
        lib_dir = vendor / "op_api" / "lib"
        if not (lib_dir / "libcust_opapi.so").is_file():
            continue
        # ASCEND_CUSTOM_OPP_PATH wants the vendor root and appends
        # `op_api/lib` itself; LD_LIBRARY_PATH wants the directory as-is.
        _append_env_path("ASCEND_CUSTOM_OPP_PATH", vendor)
        _append_env_path("LD_LIBRARY_PATH", lib_dir)
        found.append(vendor)
    return found


def manifest() -> dict:
    """What ``build.py`` recorded, or ``{}`` if the CANN arm is not built.

    Holds the pin commits, every extracted blob's SHA256, the rename rules, the
    vendor names and the built libraries' SHA256. ``_check_manifest`` verifies
    the libraries still hash to what was recorded, so a stale or half-rebuilt
    package is an error at import rather than a wrong number in a benchmark.
    """
    if not _MANIFEST_PATH.is_file():
        return {}
    return json.loads(_MANIFEST_PATH.read_text())


def _check_manifest(vendors: list[Path]) -> None:
    """Fail loudly if the installed libraries are not the ones built.

    Only checks what the manifest claims. A manifest with no ``vendors`` entry
    (Phase 1, fla arm only) checks nothing and passes.
    """
    import hashlib

    recorded = manifest().get("vendors", {})
    if not recorded:
        return

    by_name = {vendor.name: vendor for vendor in vendors}
    for name, entry in recorded.items():
        vendor = by_name.get(name)
        if vendor is None:
            raise RuntimeError(
                f"ref_ops: the manifest records vendor package {name!r} but it "
                f"is not installed under {OPP_ROOT}. Rerun "
                f"`python -m ref_ops.build`."
            )
        lib = vendor / "op_api" / "lib" / "libcust_opapi.so"
        digest = hashlib.sha256(lib.read_bytes()).hexdigest()
        if digest != entry["libcust_opapi_sha256"]:
            raise RuntimeError(
                f"ref_ops: {lib} is not the library the manifest was written "
                f"for (sha256 {digest[:12]} vs {entry['libcust_opapi_sha256'][:12]}). "
                f"Rerun `python -m ref_ops.build`."
            )


#: The CANN vendor roots this import exposed. Empty until the build has run.
BUILT = _expose_vendors()
_check_manifest(BUILT)

from . import cann, fla_npu  # noqa: E402  must follow the environment setup
