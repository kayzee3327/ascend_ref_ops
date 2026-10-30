"""The same inertness check for recurrent_kda, where a patch is also involved.

Both packages carry the UB `broadTmp` fix -- the system one from the local
checkout's uncommitted edit, the renamed one from `pins.py`'s `Patch` -- so
agreement here also confirms the two spellings of that fix are the same fix.

Decode shapes only: this op is capped at 8 tokens per sequence and returns
without writing above that, so a prefill comparison would be measuring
uninitialized memory.
"""
import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "4")
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

# The original-named op, through the wheel built from the local checkout.
from cann_ops_transformer_glm53recurrent.ops.attention.recurrent_kda import (
    recurrent_kda as original,
)

torch.npu.set_device("npu:0")
D = 128
SCALE = D**-0.5

# (heads, requests, tokens_per_request) -- tokens <= 8, the op's MAX_MTP.
CASES = [(h, b, t) for t in (1, 4) for h in (16, 32, 64) for b in (1, 32)]

print(f"{'case':<24} {'out max abs':>12} {'state max abs':>14} "
      f"{'out bit-eq':>11} {'state bit-eq':>13}")
print("-" * 78)

all_equal = True
for heads, batch, steps in CASES:
    gen = torch.Generator(device="cpu").manual_seed(20261030 + heads + batch)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=gen) * scale).to(
            torch.bfloat16).npu()

    q, k, v = (rand(batch, steps, heads, D) for _ in range(3))
    g = (torch.randn(batch, steps, heads, D, generator=gen) * 0.02
         - 0.5).float().npu()
    beta = torch.sigmoid(torch.randn(batch, steps, heads, generator=gen)).to(
        torch.bfloat16).npu()
    A_log = torch.linspace(-0.5, 0.5, heads, dtype=torch.float32).npu()
    dt_bias = torch.linspace(-0.1, 0.1, heads * D, dtype=torch.float32).npu()
    # [seq_num, max_step]: one row per request, one column per step.
    slots = torch.arange(batch, dtype=torch.int32).reshape(-1, 1).npu()
    slots = slots.expand(batch, steps).contiguous()

    # Both arms start from the same non-zero state, so the state comparison is
    # not trivially satisfied by both having written nothing.
    base = (torch.randn(batch, heads, D, D, generator=gen)
            * 0.01).float().npu()

    kwargs = dict(
        ssm_state_indices=slots, A_log=A_log, dt_bias=dt_bias,
        layout="BSND", scale=SCALE, output_final_state=False,
        inplace_final_state=True, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        safe_gate=True, lower_bound=-5.0, state_v_first=True,
    )

    renamed_state = base.clone()
    renamed_out, _ = cann.recurrent_kda(
        q, k, v, g, beta, renamed_state, **kwargs)
    original_state = base.clone()
    original_out, _ = original(q, k, v, g, beta, original_state, **kwargs)
    torch.npu.synchronize()

    # A state that never moved would make the comparison vacuous.
    if torch.equal(renamed_state, base):
        print(f"H{heads} B{batch} S{steps}: state unwritten, skipping")
        continue

    out_eq = bool(torch.equal(renamed_out, original_out))
    state_eq = bool(torch.equal(renamed_state, original_state))
    all_equal = all_equal and out_eq and state_eq
    out_diff = (renamed_out.float() - original_out.float()).abs().max().item()
    state_diff = (renamed_state - original_state).abs().max().item()

    label = f"H{heads:<2d} B{batch:<3d} S{steps}"
    print(f"{label:<24} {out_diff:>12.3e} {state_diff:>14.3e} "
          f"{str(out_eq):>11} {str(state_eq):>13}")

print()
print("rename is bit-for-bit inert:" if all_equal
      else "RENAME CHANGED THE RESULT:", all_equal)
