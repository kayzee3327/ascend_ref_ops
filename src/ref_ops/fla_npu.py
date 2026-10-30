"""The two KDA operators from flash-linear-attention-npu.

Nothing is built here. The published wheel already carries everything the
requirements ask for: both operators, their Ascend C vendor package *inside*
``site-packages/fla_npu/opp/vendors/``, and a ``fla_npu_opp_env.pth`` that puts
that OPP on ``ASCEND_CUSTOM_OPP_PATH`` at interpreter start. So this arm is
venv-contained and commit-pinned before ``ref_ops`` touches it -- the version
string embeds the commit (``26.10.0+main.devfa5493f`` <- ``fa5493f``), which
``check_pin`` asserts against ``pins.py``.

Building fla's kernels from source would be strictly worse. There is a
direct-launch example in the upstream tree that bypasses aclnn, but it hardcodes
``tiling.isVarLen = false`` and takes the gate cumsum precomputed, so using it
would mean re-deriving varlen tiling that the wheel's aclnn path already does
correctly: new risk, no gain.

What this module does add is one bug absorption. Six of fla's chunk wrappers
guard on ``if cu_seqlens and not chunk_indices:``, which calls ``bool()`` on a
tensor and raises "Boolean value of Tensor with more than one value is
ambiguous". The kernel is fine with either form -- upstream's own
``npu_chunk_gated_delta_rule_fwd_prepare`` converts tensors to an int list for
exactly this reason, so the chunk wrappers are a missed call site rather than an
interface choice. ``chunk_kda`` converts, so callers pass a device tensor here
like everywhere else. ``recurrent_kda`` is left alone: it reads ``cu_seqlens``
on device (which is what ACLGraph capture needs), so converting it there would
be a regression, not a fix.
"""

from __future__ import annotations

from typing import Any

from .pins import PINS

__all__ = ["chunk_kda", "recurrent_kda", "check_pin", "available"]

_HINT = "install it with `python -m ref_ops.build`"


def check_pin() -> None:
    """Assert the installed wheel is the pinned commit.

    The commit is not recorded separately from the version -- the version *is*
    the record -- so this compares the two strings. A wheel bump that moves the
    commit fails here rather than silently changing what this arm measures.
    """
    from importlib.metadata import PackageNotFoundError, version

    expected = PINS["fla"].wheel_version
    try:
        installed = version("flash-linear-attention-npu")
    except PackageNotFoundError:
        raise RuntimeError(
            f"ref_ops.fla_npu: flash-linear-attention-npu is not installed; "
            f"{_HINT}"
        ) from None
    if installed != expected:
        raise RuntimeError(
            f"ref_ops.fla_npu: flash-linear-attention-npu {installed} is "
            f"installed but pins.py pins {expected}. Either rerun "
            f"`python -m ref_ops.build` or move the pin."
        )


def available() -> tuple[bool, str]:
    """Whether both operators can be called, and why not if they cannot.

    Shaped like ``ref_ops.cann.available`` so a caller probing the four arms
    does not need to know that this one needs no build.
    """
    try:
        check_pin()
        _ops()
    except (RuntimeError, ImportError, OSError) as exc:
        return False, str(exc).splitlines()[0]
    return True, ""


def _ops():
    """The upstream module, imported on first use rather than at module scope.

    Importing ``fla_npu`` loads its embedded OPP and dlopens its
    ``libcust_opapi.so``, which is real work and a real failure surface. Keeping
    it out of ``import ref_ops`` means the CANN arm is usable on a machine where
    this wheel is missing, and vice versa.
    """
    from fla_npu.ops import ascendc

    return ascendc


def _host_cu_seqlens(cu_seqlens):
    """``cu_seqlens`` as a plain int list, which is the form fla's guard takes.

    Accepts the device tensor every other arm here takes, and also a list, so
    this is an added tolerance rather than a different convention.
    """
    if cu_seqlens is None:
        return None
    if hasattr(cu_seqlens, "tolist"):
        return [int(x) for x in cu_seqlens.cpu().tolist()]
    return [int(x) for x in cu_seqlens]


def chunk_kda(*args: Any, cu_seqlens=None, **kwargs: Any):
    """``fla_npu.ops.ascendc.chunk_kda_fwd``, taking a tensor ``cu_seqlens``.

    Every other argument is forwarded untouched, so the upstream signature is
    the documentation for this one:

        chunk_kda(q, k, v, g, beta, scale, chunk_size=64, *, layout="BSND",
                  cu_seqlens=None, safe_gate=False, lower_bound=None,
                  use_gate_in_kernel=False, A_log=None, dt_bias=None, ...)

    Returns upstream's full tuple -- ``(attn_out, final_state, *intermediates)``
    -- rather than trimming it to the first two. The backward exports a caller
    does not request are not computed, but ``gk``/``Aqk``/``Akk`` are mandatory
    and are computed either way; that cost is part of what this kernel does and
    stays in any measurement of it.
    """
    check_pin()
    return _ops().chunk_kda_fwd(
        *args, cu_seqlens=_host_cu_seqlens(cu_seqlens), **kwargs
    )


def recurrent_kda(*args: Any, **kwargs: Any):
    """``fla_npu.ops.ascendc.recurrent_kda``, forwarded as-is.

    No ``cu_seqlens`` conversion: this op's device kernel reads the offsets
    itself, so a tensor is the correct form and a host list would be the bug.

    Two traps in calling it, neither of them this wrapper's to fix, both worth
    knowing before reading a number out of it:

    * ``layout="TND"`` for a packed sequence. ``BSND`` is ``[B,T,H,D]`` and
      reads a squeezed ``[1,T,H,D]``'s ``T`` as the batch.
    * ``ssm_state_indices`` is per *token* in TND, so per-sequence pool indices
      need ``repeat_interleave(lengths)``.

    It mutates ``initial_state`` in place when ``inplace_final_state`` is true,
    which is the default.
    """
    check_pin()
    return _ops().recurrent_kda(*args, **kwargs)
