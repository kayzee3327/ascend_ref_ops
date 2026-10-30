# Working on ref_ops

Written for: whoever installs this package or moves a pin next, agent or human.

Read this before changing `pins.py`. Installing is three commands; moving a pin
is also three commands, but the ways it goes wrong are specific and most of them
surface a long way from the cause.

## Install

```sh
cd /data/wangkaize/ref_ops
/data/wangkaize/.vsglang/bin/python -m pip install -e . --no-deps
/data/wangkaize/.vsglang/bin/python -m ref_ops.build
/data/wangkaize/.vsglang/bin/python -m ref_ops.build --check   # expect exit 0
```

**Use `/data/wangkaize/.vsglang/bin/python`, never a bare `python`.** Two
separate reasons, both silent:

- `build.py` installs the `recurrent_kda` vendor wheel into `sys.executable`.
  Run it with the system interpreter (`/usr/local/python3.11.15/bin/python`,
  which is what bare `python` resolves to) and the wheel lands where nothing
  can import it. `--check` then reports the op MISSING with a message about the
  vendor package, which is not where the problem is.
- The system interpreter has torch and the CANN stack but no `triton`,
  `sglang` or `benchmark`, so anything that consumes `ref_ops` dies on an
  unrelated import first.

The CANN arm takes roughly 20 minutes: two kernel compiles. The fla arm
compiles nothing and only asserts a wheel version. `--only=fla` / `--only=cann`
and `--ops=chunk_kda_fwd` narrow the work; `-j` defaults to half the cores.

A CANN environment must be sourced. It already is here, from `.bashrc` — do not
re-source `set_env.sh`.

### What a successful install leaves

| path | what | gitignored |
|---|---|---|
| `.cache/*.git` | bare per-remote fetch caches, ~46M | yes |
| `_work/ops-transformer` | extracted, patched, renamed sources, ~346M | yes |
| `src/ref_ops/_opp/vendors/` | the two vendor packages, ~55M | yes |
| `src/ref_ops/_bridge/` | JIT-compiled aclnn bridge | yes |
| `src/ref_ops/_manifest.json` | what was built, from which commits | yes |

`_work/` is **not** scratch: `ref_ops.cann` imports `OpBuilder` and
`cann_ops_transformer.utils` from `_work/ops-transformer/torch_extension/` at
first call, and the manifest records the path. Deleting it breaks both CANN
operators until the next build. The other four are safe to delete.

## Moving a pin

```sh
# 1. edit the commit in src/ref_ops/pins.py
/data/wangkaize/.vsglang/bin/python -m ref_ops.build        # rebuilds, it notices
cd tests && ASCEND_RT_VISIBLE_DEVICES=4 \
  /data/wangkaize/.vsglang/bin/python test_chunk_rename_inert.py
```

`build` compares `pins.py` against the manifest and rebuilds what moved, so no
`--force` is needed and `--check` reports `STALE` rather than `ok` in between.
That comparison exists because the artifact cannot be asked which commit it came
from: a package built at the old pin is a perfectly valid package, and
`available()` says yes to it.

**Rerun the inertness tests after any `cann` pin bump.** They are the only thing
that can catch a renamed op type changing kernel behaviour, and they compare
against the original-named packages in the toolkit's `opp/vendors` — which is
why those are still installed. See `tests/README.md`.

### The four pin-bump failures, and what each looks like

**1. A patch stops applying.** `pins.py` carries one fix: `recurrent_kda`
under-reserves `broadTmp` and the kernel writes past its UB allocation. A
`Patch` must match its text *exactly once*, so a pin that touches those lines
fails immediately:

```
patch for cann:attention/recurrent_kda/.../recurrent_kda_tiling_processor.h
matches 0 times, expected exactly 1.
```

Re-read the function at the new commit. If upstream fixed it, delete the
`Patch`; if it moved, update the text. Do not loosen the match to make it pass —
a silently-skipped patch ships a kernel that faults the device, and the symptom
is an all-NaN output, not a build error.

**2. The extraction is missing a subtree.** `paths` on the `cann` pin lists 15
prefixes, five of which exist only because the build drags them in regardless of
what the operators include. A new one does not announce itself at configure
time; each of the five surfaced further into the build than the last, the worst
at CPack *after* both kernels had compiled. Read the error for an absolute path
under `_work/`, add its prefix to `paths`, rebuild:

| failure text | add |
|---|---|
| `include could not find requested file: version.cmake` | `version.cmake` |
| `Error copying directory` at ~37% | the named dir, as `gmm/common/*` were |
| `file INSTALL cannot find ...` at CPack | the named dir, as `mc2/common/op_kernel/` was |

**3. The rename misses an identifier.** Four identifiers derive from one op type
and all four must move: the op type, `opFile`/`opInterface` (derived by
camel-to-snake, so the kernel *filename* and entry symbol too), and the aclnn
symbol. Stage 6 nm-checks the renamed aclnn symbol, so a miss is a loud failure
naming what it found instead:

```
ref_ops_cann_chunk_kda_fwd: ... does not export aclnnRefOpsCannChunkKdaFwd.
Kda symbols it does export: ['aclnnChunkKdaFwd']
```

The old name in that list means the rename half-applied. **Do not anchor the
substitution on `\b`** — that was tried and it is why this failure mode is
documented: `aclnnChunkKdaFwd` has no word boundary before the camel and
`aclnn_chunk_kda_fwd.h` none before the snake, so the two aclnn identifiers kept
their original spelling while everything around them was renamed. Plain
substring substitution is correct here.

**4. The fla wheel is not on PyPI.** `build.py` cannot install it; it asserts
the version and tells you where to get it. Moving the `fla` pin means building
that wheel from its own checkout first. The version string embeds the commit
(`26.10.0+main.devfa5493f` ← `fa5493f`), so the two cannot disagree silently.

## Two traps that are not about pins

**Never share a `--vendor_name` between ops.** The installer inside the `.run`
does `rm -rf "$targetdir/vendors/<vendor>/<subdir>"/*` whenever an incoming
filename collides, and every single-op build ships its own `libcust_opapi.so`.
A shared name means the second install silently deletes the first op while
printing SUCCESS. One `Op` per vendor, as `pins.py` has it.

**fla still collides with the *original* CANN packages**, only the renamed ones
coexist with it. If you need the originals in a process — the inertness tests
do — set `ASCEND_CUSTOM_OPP_PATH` explicitly *and* `FLA_NPU_DISABLE_PTH=1`,
because fla's wheel installs a `.pth` that prepends its vendor at interpreter
start, before any user code runs. Appending is not enough; the other package
has to be unreachable.

## Checking your work

```sh
/data/wangkaize/.vsglang/bin/python -m ref_ops.build --check   # exit 0
cd tests && ASCEND_RT_VISIBLE_DEVICES=4 \
  /data/wangkaize/.vsglang/bin/python test_all_four_coexist.py
```

`--check` passing is weak evidence; it only proves the symbols exist. The
coexistence test launching all four operators is the real signal, and a renamed
op that resolves to the wrong kernel fails there rather than in `--check`.

`import ref_ops` also re-hashes each installed `libcust_opapi.so` against the
manifest, so a half-finished or hand-edited build is an error at import instead
of a wrong number in a benchmark.
