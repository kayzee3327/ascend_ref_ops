"""The pinned upstreams, and the rename that lets two of them coexist.

This is the only file to edit to rebuild against different sources: change a
commit here and rerun ``python -m ref_ops.build``.

Three pins, two of which are source and one of which is a build-time third
party:

``fla``      flash-linear-attention-npu. Not built from source at all -- the
             published wheel already carries both operators, their vendor OPP
             *inside* ``site-packages/fla_npu/opp/``, and a ``.pth`` that puts
             it on ``ASCEND_CUSTOM_OPP_PATH`` at interpreter start. The commit
             is pinned by the wheel's own version string, which embeds it.
``cann``     ops-transformer. Built here, because the installed toolkit is CANN
             9.0.0 and neither operator exists in it -- see ``build.py``.
``catlass``  a header-only dependency ops-transformer ``FetchContent``s at
             *configure* time, with the tag hardcoded in
             ``cmake/third_party/catlass.cmake``. Pinned explicitly so the
             build does not reach the network halfway through.

The rename is the load-bearing part. Both upstreams define the same two op
types, ``ChunkKdaFwd`` and ``RecurrentKda``, and CANN's launcher registry is
keyed by the op-type string, so only one package can claim each name per
process. Renaming the CANN-side copies dissolves that structurally instead of
selecting one arm per process.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Patch:
    """One exact-text replacement in one extracted file.

    Exact text rather than a diff, because a diff needs line numbers and those
    are the thing most likely to move under a pin bump. ``build.py`` requires
    ``old`` to occur exactly once and fails if it does not, so a patch that no
    longer applies is an error at build time rather than a silent no-op.

    ``path`` is relative to the extraction root, **before** the rename -- the
    rename runs after patching, so a patch is written against the upstream
    spelling it was read in.
    """

    path: str
    old: str
    new: str
    why: str


@dataclass(frozen=True)
class Pin:
    """One upstream, at one commit.

    ``paths`` is the subtree prefixes to extract; empty means the whole tree.
    ``patches`` are textual fixups applied to the extraction, before the
    rename.
    """

    url: str
    commit: str
    paths: tuple[str, ...] = ()
    patches: tuple[Patch, ...] = ()
    wheel_version: str | None = None


# `recurrent_kda`'s host tiling under-reserves UB, and the kernel then writes
# past its allocation: upstream's own pytest fails with "VEC instruction error:
# the ub address out of bounds". `broadTmpInUb` is used *unconditionally* --
# `Compute()` calls `Broadcast` and two `MatVecMul` through it on every path --
# but it is only budgeted when K != 128, and the op only accepts K == 128. So
# `ubRestBytes` is always short by `4 * aDk * vStep`, about 30 KB of 192 KB.
#
# This is upstream's bug at the pinned commit, not a cross-version artifact of
# building 9.2.0 sources on the 9.0.0 toolkit. Dropping the conditional is the
# only fix that does not touch the kernel; it costs a tile of V parallelism
# (vStep 64 -> 48).
_RECURRENT_KDA_UB_PATCH = Patch(
    path="attention/recurrent_kda/op_host/recurrent_kda_tiling_processor.h",
    old="""        int64_t coeff = 4 * aDk + 8; // state and row outputs.
        if (aDk != RKDA_FUSED_REDUCE_K) {
            coeff += 4 * aDk; // broadTmp for the generic matvec path.
        }
        return coeff;""",
    new="""        int64_t coeff = 4 * aDk + 8; // state and row outputs.
        // broadTmp, always: `Compute()` reaches `broadTmpInUb` on every path,
        // including the K == 128 fused-reduce one this conditional excluded,
        // and K == 128 is the only K the operator accepts. Budgeting it only
        // for the generic path leaves `ubRestBytes` short by `4 * aDk * vStep`
        // and the kernel writes past its UB allocation.
        coeff += 4 * aDk;
        return coeff;""",
    why="upstream under-reserves broadTmp; the kernel writes past its UB "
        "allocation and its own pytest fails with 'ub address out of bounds'",
)


PINS = {
    "fla": Pin(
        url="https://github.com/flashserve/flash-linear-attention-npu.git",
        commit="fa5493f10e0bebe7a96ebcc81a4acf81d79f1e12",
        # The wheel encodes its own commit: `devfa5493f` is this pin's prefix.
        # `build.py` asserts the two agree, so a wheel bump that moves the
        # commit is an error here rather than a silent change of what `fla`
        # measures.
        wheel_version="26.10.0+main.devfa5493f",
    ),
    "cann": Pin(
        url="https://gitcode.com/cann/ops-transformer.git",
        commit="eb9e170147c5c4f255592eb55a51aab824641f12",
        # The two operators plus the build system they need: 227 files of
        # 13561, which keeps `_work/` reviewable -- and that matters, because
        # the rename is applied to it.
        #
        # The non-obvious entries are the ones a first attempt omitted and the
        # configure step then failed on. `version.cmake` is `include`d
        # unconditionally from the top-level `CMakeLists.txt`, `common/` is
        # `add_subdirectory`d unconditionally, and `attention/CMakeLists.txt`
        # is what walks into the two op directories at all. The remaining
        # module directories (`moe/`, `mc2/`, ...) are *not* here, which is
        # why the build passes `--module=attention`: `should_add_module`
        # defaults to true for every module when `ASCEND_MODULE_NAME` is
        # unset, and would then descend into directories this extraction does
        # not contain.
        paths=(
            "attention/CMakeLists.txt",
            "attention/chunk_kda_fwd/",
            "attention/recurrent_kda/",
            "cmake/",
            "common/",
            # Three directories that every operator's build drags in whether
            # or not it includes them, found one failure at a time rather than
            # by reading the build:
            #
            # `gmm/common/*` are `copy_directory`d into the shared
            # kernel-include tree by `add_ops_src_copy` (`cmake/func.cmake`),
            # which fails at ~37% with "Error copying directory".
            # `mc2/common/op_kernel/` is a `file(INSTALL ...)` in the generated
            # `cmake_install.cmake`, so it fails later still -- at CPack, after
            # the kernel has already compiled.
            #
            # The pattern: a missing subtree does not announce itself at
            # configure time. Each one surfaced further into the build than the
            # last.
            "gmm/common/cgmct/",
            "gmm/common/op_kernel/",
            "mc2/common/op_kernel/",
            "scripts/",
            "torch_extension/",
            "build.sh",
            "CMakeLists.txt",
            "requirements.txt",
            "version.cmake",
            "version.info",
        ),
        patches=(_RECURRENT_KDA_UB_PATCH,),
    ),
    "catlass": Pin(
        url="https://gitcode.com/cann/catlass.git",
        commit="769cd40a8716b28650b6bebb08db4834eea4462f",
        # Header-only as far as this build is concerned.
        paths=("include/", "CMakeLists.txt"),
    ),
}


@dataclass(frozen=True)
class Op:
    """One CANN operator, under its original and renamed identities.

    Four identifiers derive from one op type and all four move together:

    1. the **op type**, which ``OP_ADD(opType)`` registers as the stringified
       class name (``register/op_def_registry.h``) -- a textual identifier, not
       a build-system constant;
    2. ``opFile`` / ``opInterface``, which are *derived* from the op type by
       camel-to-snake in the toolkit's ``opdesc_parser.py`` rather than declared
       anywhere, so the kernel source filename and the kernel entry function
       name must match the renamed type;
    3. the **kernel entry symbol**, ``extern "C" __global__ __aicore__ void
       <snake>(...)``;
    4. the **aclnn symbol**, ``aclnn`` + camel. Easy to miss and still
       load-bearing: the dlsym search walks every vendor's ``libcust_opapi.so``
       and takes the first match, so two vendors exporting ``aclnnChunkKdaFwd``
       means one silently wins *even with distinct op types*.

    Substituting the camel and snake spellings as whole words covers all four:
    camel gets the op type and (via the ``aclnn`` prefix) the aclnn symbol,
    snake gets ``opFile``, ``opInterface``, the filename and the entry symbol.
    """

    camel: str
    snake: str
    renamed_camel: str
    renamed_snake: str
    # One vendor package per op, never shared. The installer inside the `.run`
    # does `rm -rf "$targetdir/vendors/<vendor>/<subdir>"/*` whenever an
    # incoming filename collides, and every single-op build ships its own
    # `libcust_opapi.so` -- so a shared vendor name means the second install
    # silently deletes the first op while printing SUCCESS.
    vendor: str
    # `recurrent_kda` ships a `torch_extension/` wrapper that only becomes
    # importable once the vendor-named wheel installs it; `chunk_kda_fwd` ships
    # none and is bound by this package's own aclnn bridge.
    needs_wheel: bool

    @property
    def aclnn(self) -> str:
        return "aclnn" + self.renamed_camel

    def rules(self) -> tuple[tuple[str, str], ...]:
        """The whole-word substitutions, camel before snake.

        Order is not significant -- neither spelling is a substring of the
        other at a word boundary -- but it is fixed so the manifest is stable.
        """
        return ((self.camel, self.renamed_camel), (self.snake, self.renamed_snake))


OPS = (
    Op(
        camel="ChunkKdaFwd",
        snake="chunk_kda_fwd",
        renamed_camel="RefOpsCannChunkKdaFwd",
        renamed_snake="ref_ops_cann_chunk_kda_fwd",
        vendor="refopschunk",
        needs_wheel=False,
    ),
    Op(
        camel="RecurrentKda",
        snake="recurrent_kda",
        renamed_camel="RefOpsCannRecurrentKda",
        renamed_snake="ref_ops_cann_recurrent_kda",
        vendor="refopsrecurrent",
        needs_wheel=True,
    ),
)

OPS_BY_SNAKE = {op.snake: op for op in OPS}

SOC = "ascend910b"
