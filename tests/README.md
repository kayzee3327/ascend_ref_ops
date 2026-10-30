# Verification

Three scripts, each answering one question the package makes a claim about.
Plain scripts rather than pytest: they need a device, they print tables worth
reading, and a failure is a number to look at rather than an assertion to
count.

```sh
cd tests
ASCEND_RT_VISIBLE_DEVICES=4 /data/wangkaize/.vsglang/bin/python test_all_four_coexist.py
```

| script | question | result on 2026-10-30 |
|---|---|---|
| `test_all_four_coexist.py` | do all four operators run in one process? | yes, no segfault, and a cann call after every fla call is bit-identical to the first |
| `test_chunk_rename_inert.py` | is `RefOpsCannChunkKdaFwd` the same kernel as `ChunkKdaFwd`? | bit-for-bit, 12 shapes |
| `test_recurrent_rename_inert.py` | same, for `RecurrentKda` | bit-for-bit, 12 shapes |

The two inertness scripts are the ones worth rerunning after a pin bump. They
compare against the **original-named** packages in the toolkit's own
`opp/vendors`, so they only work while those are installed; the renamed
packages do not need them, and `test_all_four_coexist.py` was also confirmed to
pass with that directory bind-mounted empty.

Both of them keep fla out deliberately, with an explicit
`ASCEND_CUSTOM_OPP_PATH` and `FLA_NPU_DISABLE_PTH=1`: fla claims `ChunkKdaFwd`,
which is the same name the original system package claims, and that collision is
the thing the rename exists to dissolve. Only the *renamed* packages coexist
with fla.

One expected difference, not a failure: cann and fla on identical inputs land
~1 bf16 ulp apart (9.5e-7 on outputs, `chunk`'s final_state bit-identical).
Different summation order in a recurrence whose output is small.
