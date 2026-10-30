"""All four reference KDA operators, one process. The requirement, tested.

Order matters and is deliberately interleaved: if any arm's registration lost
to another's, the loser segfaults or raises ADD_TO_LAUNCHER_LIST_AICORE. The
second cann call after an fla call is the check that nothing was clobbered.
"""
import os
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "4")

import torch
import torch_npu  # noqa: F401

import ref_ops
from ref_ops import cann, fla_npu

print("vendor roots:", [p.name for p in ref_ops.BUILT])
m = ref_ops.manifest()
print("pins:", {k: v.get("commit", "")[:12] for k, v in m["pins"].items()})
print("patches:", [p["path"] for p in m["patches"]])
print()

torch.npu.set_device("npu:0")
D, H, C = 128, 4, 64
SCALE = D**-0.5
gen = torch.Generator(device="cpu").manual_seed(20261030)


def rand(*shape, scale=0.05):
    return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16).npu()


A_log = torch.linspace(-0.5, 0.5, H, dtype=torch.float32).npu()
dt_bias = torch.linspace(-0.1, 0.1, H * D, dtype=torch.float32).npu()

# --- prefill shapes, for the chunk ops ---
T = 1024
q, k, v = rand(T, H, D), rand(T, H, D), rand(T, H, D)
g = (torch.randn(T, H, D, generator=gen) * 0.02 - 0.5).float().npu()
beta = torch.sigmoid(torch.randn(T, H, generator=gen)).to(torch.bfloat16).npu()
cu_seqlens = torch.tensor([0, T], dtype=torch.int64).npu()

# --- decode shapes, for the recurrent ops. 1 token/seq: the CANN op is
#     capped at 8 and silently no-ops above it, so prefill is out of scope.
B = 8
dq, dk, dv = rand(B, 1, H, D), rand(B, 1, H, D), rand(B, 1, H, D)
dg = (torch.randn(B, 1, H, D, generator=gen) * 0.02 - 0.5).float().npu()
dbeta = torch.sigmoid(torch.randn(B, 1, H, generator=gen)).to(torch.bfloat16).npu()
slots = torch.arange(B, dtype=torch.int32).reshape(-1, 1).npu()


def fresh_state():
    """A V-first contiguous pool, which is what state_v_first=True wants."""
    return torch.zeros(B, H, D, D, dtype=torch.float32).npu()


results = {}

print("1. cann.chunk_kda")
a, s = cann.chunk_kda(q, k, v, g, beta, scale=SCALE, chunk_size=C,
                      cu_seqlens=cu_seqlens, A_log=A_log, dt_bias=dt_bias)
torch.npu.synchronize()
results["cann.chunk_kda"] = (a, s)
print(f"   out {tuple(a.shape)} state {tuple(s.shape)} "
      f"|out| {a.float().abs().mean():.4e}")

print("2. fla_npu.chunk_kda")
out = fla_npu.chunk_kda(q, k, v, g, beta, SCALE, C, layout="TND",
                        cu_seqlens=cu_seqlens, output_final_state=True,
                        safe_gate=True, lower_bound=-5.0,
                        use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias)
torch.npu.synchronize()
results["fla_npu.chunk_kda"] = (out[0], out[1])
print(f"   out {tuple(out[0].shape)} state {tuple(out[1].shape)} "
      f"|out| {out[0].float().abs().mean():.4e}")

print("3. cann.recurrent_kda")
state = fresh_state()
a, _ = cann.recurrent_kda(
    dq, dk, dv, dg, dbeta, state,
    ssm_state_indices=slots, A_log=A_log, dt_bias=dt_bias,
    layout="BSND", scale=SCALE, output_final_state=False,
    inplace_final_state=True, use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
    safe_gate=True, lower_bound=-5.0, state_v_first=True,
)
torch.npu.synchronize()
results["cann.recurrent_kda"] = (a, state)
print(f"   out {tuple(a.shape)} |out| {a.float().abs().mean():.4e} "
      f"state written {bool(state.abs().sum() > 0)}")

print("4. fla_npu.recurrent_kda")
fstate = fresh_state()
fa, _ = fla_npu.recurrent_kda(
    dq, dk, dv, dg, dbeta, fstate,
    ssm_state_indices=slots, A_log=A_log, dt_bias=dt_bias,
    layout="BSND", scale=SCALE, output_final_state=False,
    inplace_final_state=True, use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
    safe_gate=True, lower_bound=-5.0, state_v_first=True,
)
torch.npu.synchronize()
results["fla_npu.recurrent_kda"] = (fa, fstate)
print(f"   out {tuple(fa.shape)} |out| {fa.float().abs().mean():.4e} "
      f"state written {bool(fstate.abs().sum() > 0)}")

print()
print("5. cann.chunk_kda again, after every fla call")
again, _ = cann.chunk_kda(q, k, v, g, beta, scale=SCALE, chunk_size=C,
                          cu_seqlens=cu_seqlens, A_log=A_log, dt_bias=dt_bias)
torch.npu.synchronize()
print("   bit-identical to call 1:",
      bool(torch.equal(again, results["cann.chunk_kda"][0])))

print()
print("--- cann vs fla, same op, same inputs ---")
for label, left, right in (
    ("chunk    out  ", "cann.chunk_kda", "fla_npu.chunk_kda"),
    ("recurrent out ", "cann.recurrent_kda", "fla_npu.recurrent_kda"),
):
    for which, idx in (("out", 0), ("state", 1)):
        a = results[left][idx].float().cpu()
        b = results[right][idx].float().cpu()
        if a.shape != b.shape:
            print(f"{label} {which}: shapes differ {tuple(a.shape)} "
                  f"vs {tuple(b.shape)}")
            continue
        diff = (a - b).abs()
        print(f"{left.split('.')[1]:<13} {which:<6} max abs {diff.max():.3e}  "
              f"mean |fla| {b.abs().mean():.3e}  "
              f"bit-identical {bool(torch.equal(a, b))}")

print()
print("all four operators ran in one process")
