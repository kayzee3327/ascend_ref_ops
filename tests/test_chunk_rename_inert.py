"""Phase 3: is the renamed CANN op bit-for-bit the original?

The renamed package (`RefOpsCannChunkKdaFwd`) and the original system package
(`ChunkKdaFwd`, vendor `glm53chunk_transformer`) now hold *different* op types
and *different* aclnn symbols, so unlike fla they can coexist -- which is what
makes this comparison possible in one process at all.

fla has to be kept out: it claims `ChunkKdaFwd`, the same name the original
system package claims, and that is the collision this whole exercise is about.
Hence the explicit ASCEND_CUSTOM_OPP_PATH and FLA_NPU_DISABLE_PTH, set before
any import.

Residual risk this exists to catch: a renamed op type changing a
tiling-template selection keyed on the name. Nothing found by reading suggested
that, but reading is not the proof.
"""
import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "4")
# fla's .pth prepends its vendor at interpreter start, before any user code.
os.environ["FLA_NPU_DISABLE_PTH"] = "1"
_SYS = "/usr/local/Ascend/cann-9.0.0/opp/vendors"
_OURS = "/data/wangkaize/ref_ops/src/ref_ops/_opp/vendors"
os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join([
    f"{_OURS}/refopschunk_transformer",
    f"{_OURS}/refopsrecurrent_transformer",
    f"{_SYS}/glm53chunk_transformer",
    f"{_SYS}/glm53recurrent_transformer",
])

import torch
import torch_npu  # noqa: F401

from ref_ops import cann
from ref_ops.cann import _BRIDGE_TEMPLATE, _BUILD_DIR, _source_root

torch.npu.set_device("npu:0")
D = 128
SCALE = D**-0.5


def load_original():
    """A bridge for the *original* aclnnChunkKdaFwd, from the same template.

    Same code path as `ref_ops.cann`, only the symbol differs -- so a
    difference in the result is the kernel's, not the binding's.
    """
    import logging
    import sys

    from torch.utils.cpp_extension import load

    logging.getLogger("cann_ops_transformer.ops").setLevel(logging.ERROR)
    extension_dir = str(_source_root() / "torch_extension")
    if extension_dir not in sys.path:
        sys.path.insert(0, extension_dir)
    from cann_ops_transformer.op_builder import OpBuilder

    class _Bridge(OpBuilder):
        def __init__(self):
            super().__init__("chunk_kda_fwd_original", category=None)

        def sources(self):
            return []

        def schema(self):
            return None

        def register_meta(self):
            return None

    builder = _Bridge()
    builder.ensure_initialized()
    _BUILD_DIR.mkdir(exist_ok=True)
    text = _BRIDGE_TEMPLATE.format(CAMEL="ChunkKdaFwd")
    source = _BUILD_DIR / "chunk_kda_fwd_original_bridge.cpp"
    if not source.is_file() or source.read_text() != text:
        source.write_text(text)
    module = load(
        name="chunk_kda_fwd_original_bridge",
        sources=[str(source)],
        extra_include_paths=builder.include_paths(),
        extra_cflags=builder.cxx_args(),
        extra_ldflags=builder.extra_ldflags(),
        build_directory=str(_BUILD_DIR),
        verbose=False,
    )
    return module.chunk_kda


original = load_original()

CASES = [(h, t, c) for c in (64, 128) for h in (4, 8)
         for t in (1024, 2048, 4096)]

print(f"{'case':<22} {'out max abs':>12} {'state max abs':>14} "
      f"{'out bit-eq':>11} {'state bit-eq':>13}")
print("-" * 76)

all_equal = True
for heads, tokens, chunk in CASES:
    gen = torch.Generator(device="cpu").manual_seed(20261030 + tokens + heads)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=gen) * scale).to(
            torch.bfloat16).npu()

    q, k, v = (rand(tokens, heads, D) for _ in range(3))
    g = (torch.randn(tokens, heads, D, generator=gen) * 0.02 - 0.5).float().npu()
    beta = torch.sigmoid(
        torch.randn(tokens, heads, generator=gen)).to(torch.bfloat16).npu()
    A_log = torch.linspace(-0.5, 0.5, heads, dtype=torch.float32).npu()
    dt_bias = torch.linspace(-0.1, 0.1, heads * D, dtype=torch.float32).npu()
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int64).npu()

    renamed_out, renamed_state = cann.chunk_kda(
        q, k, v, g, beta, scale=SCALE, chunk_size=chunk,
        cu_seqlens=cu_seqlens, A_log=A_log, dt_bias=dt_bias,
    )
    original_out, original_state = original(
        q, k, v, g, beta, A_log, dt_bias, None, cu_seqlens,
        "TND", float(SCALE), int(chunk), True, -5.0, True, False,
    )
    torch.npu.synchronize()

    out_eq = bool(torch.equal(renamed_out, original_out))
    state_eq = bool(torch.equal(renamed_state, original_state))
    all_equal = all_equal and out_eq and state_eq
    out_diff = (renamed_out.float() - original_out.float()).abs().max().item()
    state_diff = (
        renamed_state - original_state).abs().max().item()

    label = f"H{heads:<2d} T{tokens:<5d} C{chunk}"
    print(f"{label:<22} {out_diff:>12.3e} {state_diff:>14.3e} "
          f"{str(out_eq):>11} {str(state_eq):>13}")

print()
print("rename is bit-for-bit inert:" if all_equal
      else "RENAME CHANGED THE RESULT:", all_equal)
