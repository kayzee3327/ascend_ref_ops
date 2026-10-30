# ref_ops

Four reference KDA operators, from two pinned upstreams, callable in one
process:

```python
from ref_ops import cann, fla_npu

cann.chunk_kda(...)        cann.recurrent_kda(...)
fla_npu.chunk_kda(...)     fla_npu.recurrent_kda(...)
```

Nothing is installed into the CANN system path: the vendor packages live inside
`src/ref_ops/_opp/`, so they are inside the venv, and
`/usr/local/Ascend/.../opp/vendors` stays empty.

## Setup

```sh
/data/wangkaize/.vsglang/bin/python -m pip install -e . --no-deps
/data/wangkaize/.vsglang/bin/python -m ref_ops.build
```

Use the venv's interpreter, not a bare `python` -- the vendor wheel installs
into `sys.executable`. The CANN arm takes ~20 minutes (two kernel compiles); the
fla arm compiles nothing. `--check` reports status without building, `--only`
and `--ops` narrow the work.

## Moving a pin

Edit the commit in `pins.py` and rerun `python -m ref_ops.build`. It compares
the pins against the manifest and rebuilds what moved, so no `--force` is
needed; `--check` reports `STALE` rather than `ok` in between. Then rerun the
inertness tests -- after a `cann` bump they are the only thing that can catch a
renamed op type changing kernel behaviour.

**`AGENTS.md` is the one to read first if you are about to do either of these.**
It covers the interpreter trap in full, what a successful install leaves on disk
(and which of it is not scratch), and the four specific ways a pin bump fails --
a patch that stops applying, a missing extraction subtree, a half-applied
rename, and the fla wheel not being on PyPI.

## Why the two arms are not symmetric

Both upstreams define the **same two op types**, `ChunkKdaFwd` and
`RecurrentKda`. CANN's launcher registry is keyed by the op-type string and the
aclnn dlsym handler list is built once, on the first aclnn call, and cached in a
function-local static. Two packages claiming one name means the second loses --
a segfault or `ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaFwd failed`, depending on
resolve order -- and path ordering cannot fix it. One slot, two claimants.

So the CANN-side operators are **renamed**, from sources extracted at the pinned
commit, to `RefOpsCannChunkKdaFwd` and `RefOpsCannRecurrentKda`. That dissolves
the collision structurally instead of selecting one arm per process. Four
identifiers derive from one op type and all four have to move together; `pins.py`
documents which, and why the aclnn symbol is the one that is easy to miss and
still load-bearing.

The fla arm needs none of this. Its wheel already ships both operators, their
vendor OPP *inside* `site-packages/fla_npu/opp/`, and a `.pth` that puts it on
`ASCEND_CUSTOM_OPP_PATH` at interpreter start -- so `ref_ops` appends to those
variables rather than overwriting them, and that is what lets all four coexist.

## Files

| file | what it is |
|---|---|
| `pins.py` | the only file to edit to rebuild: three commits, two rename rules |
| `build.py` | fetch → extract → rename → build → install → verify |
| `__init__.py` | sets `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`, checks the manifest |
| `cann.py` | `chunk_kda`, `recurrent_kda` from the renamed vendor packages |
| `fla_npu.py` | `chunk_kda`, `recurrent_kda` from the wheel |
| `_manifest.json` | written by `build.py`, asserted on import |

`.cache/` holds bare per-remote fetch caches, `_work/` the extracted and renamed
sources, `src/ref_ops/_opp/` the installed vendor packages, `src/ref_ops/_bridge/`
the JIT-compiled aclnn bridge. All four are gitignored and all four are
reproducible from `pins.py`.

`tests/` answers the claims this README makes; see `tests/README.md`. The short
version: all four operators run in one process, and both renamed ops are
bit-for-bit identical to their original-named counterparts across 24 shapes.

## Scope notes on the operators themselves

Two of the four arms are settled questions, recorded before this package
existed, and are here for a symmetric surface rather than because they are
competitive:

- **CANN `RecurrentKda` is decode-only, silently.** `MAX_MTP = 8`; above that
  `ValidateCuSeqlens` fails and `Process` returns *without writing anything*, as
  a successful aclnn call. A prefill-shaped comparison against it measures
  whatever was in the output allocation. It also measured ~2x slower than the
  Triton fallback.
- **fla `recurrent_kda` is 4-9x slower on prefill** -- `blockDim = seq_num * HV`,
  so a packed single sequence gets H cores of 40 -- but at parity on decode.

Benchmarking caveat, unrelated to correctness: card-4 contention inflates one
arm rather than all, so check monotonicity across token counts before believing
any column that compares these four.
