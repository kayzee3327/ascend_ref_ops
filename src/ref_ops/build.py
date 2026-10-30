"""Fetch, extract, rename, build, install, verify -- ``python -m ref_ops.build``.

Six stages, all Python. ``subprocess`` is used only to invoke ``git``,
``build.sh``, the ``.run`` installer, ``pip`` and ``nm``; extraction, renaming
and manifesting are done here, so there is no shell script to port between
machines.

1. **fetch** -- ``git fetch --depth 1 <full-sha>`` into a bare cache per remote.
   Both gitcode and GitHub accept shallow fetch by SHA.
2. **extract** -- ``git ls-tree`` plus ``git cat-file`` to read blobs *from the
   pinned commit*. Never from working files: that is what makes a dirty local
   checkout unable to leak into the artifact. Then **patch**: the exact-text
   fixups in ``pins.py``, each required to match exactly once.
3. **rename** -- the two rules in ``pins.py``, over file contents and file
   names.
4. **build** -- ``build.sh`` once per op, as a custom vendor package.
5. **install** -- the emitted ``.run`` into the package's own ``_opp/``, plus
   the vendor wheel for the op whose Python entry point only exists there.
6. **verify** -- ``nm`` for the **renamed** aclnn symbol. Not the vendor
   directory existing, not a ``libcust_opapi.so`` being present: the sibling
   op's package satisfies both, and a first-match-wins dlsym scan then picks
   the wrong package, which looks available and fails at the first launch.

Then ``_manifest.json``: pin commits, every extracted blob's SHA256, the rename
rules, the vendor names and the built libraries' SHA256. ``ref_ops/__init__.py``
checks it on import, so a stale or half-rebuilt package is an error rather than
a wrong number.

Usage:

    python -m ref_ops.build                 # everything that is not current
    python -m ref_ops.build --check         # report status, build nothing
    python -m ref_ops.build --force         # rebuild even if current
    python -m ref_ops.build --only=cann     # one arm
    python -m ref_ops.build --ops=chunk_kda_fwd
    python -m ref_ops.build -j96

Run it with the venv's interpreter (``/data/wangkaize/.vsglang/bin/python``):
the wheel installs into ``sys.executable``, so the system python would put it
where nothing can see it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import OPP_ROOT
from .pins import OPS, OPS_BY_SNAKE, PINS, SOC, Op

_PACKAGE_DIR = Path(__file__).resolve().parent
#: Project root: the editable checkout holding pyproject.toml.
_ROOT = _PACKAGE_DIR.parent.parent

CACHE_DIR = _ROOT / ".cache"
WORK_DIR = _ROOT / "_work"
MANIFEST = _PACKAGE_DIR / "_manifest.json"
LOG = _ROOT / "build.log"

# ops-transformer builds through one shared `build/` tree, so two concurrent
# builds out of one extraction corrupt each other's objects. The lock is on the
# extraction, which is this package's own, so it does not need to be shared
# with anything outside it.
LOCK = WORK_DIR / ".build-lock"


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run`` with output captured and failures raised as SystemExit."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(f"{' '.join(cmd[:4])} ... failed:\n{detail}")
    return proc


# --- 1. fetch -----------------------------------------------------------


def fetch(name: str) -> Path:
    """The bare cache for pin ``name``, holding at least its commit.

    Shallow and by exact SHA, so this costs one commit's worth of objects
    rather than a clone, and cannot drift: there is no branch to be at the head
    of. Skipped entirely when the object is already present, which is what
    makes a rerun cheap.
    """
    pin = PINS[name]
    cache = CACHE_DIR / f"{name}.git"
    if not cache.is_dir():
        cache.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q", "--bare", str(cache)])
        run(["git", "-C", str(cache), "remote", "add", "origin", pin.url])

    have = subprocess.run(
        ["git", "-C", str(cache), "cat-file", "-t", pin.commit],
        capture_output=True, text=True,
    )
    if have.returncode == 0 and have.stdout.strip() == "commit":
        log(f"  {name}: {pin.commit[:12]} already cached")
        return cache

    log(f"  {name}: fetching {pin.commit[:12]} from {pin.url}")
    run(["git", "-C", str(cache), "fetch", "--depth", "1", "origin", pin.commit],
        timeout=1800)
    run(["git", "-C", str(cache), "cat-file", "-t", pin.commit])
    return cache


# --- 2. extract ---------------------------------------------------------


def _ls_tree(cache: Path, commit: str) -> list[tuple[str, str]]:
    """``(mode, blob_sha, path)`` -> ``(path, blob_sha)`` for every file in the commit."""
    out = run(
        ["git", "-C", str(cache), "ls-tree", "-r", "-z", commit],
        ).stdout
    entries = []
    for record in out.split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        mode, kind, sha = meta.split(" ", 2)
        if kind != "blob":
            continue
        entries.append((path, sha, mode))
    return entries


def extract(name: str, cache: Path, dest: Path) -> dict[str, str]:
    """Write pin ``name``'s selected subtrees into ``dest``, from the commit.

    Returns ``{path: blob_sha}``, which goes into the manifest -- the record of
    what was built, at blob granularity, independent of any checkout.

    Blobs are read with ``git cat-file`` rather than ``git checkout``, so no
    working tree exists for a stray edit to live in and the extraction is a
    pure function of the commit.
    """
    pin = PINS[name]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    wanted = []
    for path, sha, mode in _ls_tree(cache, pin.commit):
        if pin.paths and not any(
            path == prefix or path.startswith(prefix) for prefix in pin.paths
        ):
            continue
        wanted.append((path, sha, mode))

    blobs: dict[str, str] = {}
    for path, sha, mode in wanted:
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        content = subprocess.run(
            ["git", "-C", str(cache), "cat-file", "blob", sha],
            capture_output=True, check=True,
        ).stdout
        target.write_bytes(content)
        # Preserve the executable bit; `build.sh` is invoked through bash, but
        # the scripts it calls are not all.
        if mode == "100755":
            target.chmod(0o755)
        blobs[path] = sha

    log(f"  {name}: extracted {len(blobs)} files to {dest}")
    return blobs


# --- 2b. patch ----------------------------------------------------------


def patch(name: str, root: Path) -> list[dict]:
    """Apply pin ``name``'s patches to the extraction. Returns what was applied.

    Before the rename, so a patch is written against the upstream spelling it
    was read in, and requiring exactly one occurrence of the text it replaces:
    a patch that no longer applies at a bumped pin is a loud failure rather
    than a silent no-op that ships an unfixed kernel.
    """
    applied = []
    for item in PINS[name].patches:
        target = root / item.path
        if not target.is_file():
            raise SystemExit(
                f"patch for {name} names {item.path}, which the extraction "
                f"does not contain. Either the pin moved the file or `paths` "
                f"no longer covers it."
            )
        text = target.read_text(encoding="utf-8")
        count = text.count(item.old)
        if count != 1:
            raise SystemExit(
                f"patch for {name}:{item.path} matches {count} times, expected "
                f"exactly 1. The pinned source changed under the patch; "
                f"re-read it and update pins.py.\n  reason it exists: {item.why}"
            )
        target.write_text(text.replace(item.old, item.new), encoding="utf-8")
        log(f"  patched {item.path}: {item.why}")
        applied.append({"path": item.path, "why": item.why})
    return applied


# --- 3. rename ----------------------------------------------------------

# Text extensions the rename applies to. Binary blobs are left alone: nothing
# in these subtrees is a compiled artifact, but a stray image would otherwise
# be decoded as UTF-8 and fail.
_TEXT_SUFFIXES = {
    ".cpp", ".h", ".hpp", ".cc", ".cce", ".py", ".txt", ".cmake", ".sh",
    ".json", ".md", ".ini", ".info", ".yaml", ".yml", ".csv", ".ut",
}


def _substitute(text: str, rules: tuple[tuple[str, str], ...]) -> str:
    """Substitute every rule, as a plain substring, everywhere it occurs.

    Not whole-word, and that is the correction a first attempt here needed.
    Anchoring on ``\\b`` looks safer and is actually *worse*, because the
    boundary falls in the wrong places: ``aclnnChunkKdaFwd`` has no word break
    before the camel (``n`` to ``C`` is not one) and ``aclnn_chunk_kda_fwd.h``
    has none before the snake (``_`` is a word character), so the two aclnn
    identifiers -- the ones the dlsym search actually resolves -- kept their
    original spelling while ``ChunkKdaFwdTilingData`` next to them got renamed.
    A half-applied rename is the failure this whole exercise exists to avoid:
    two vendors exporting ``aclnnChunkKdaFwd`` means one silently wins even
    with distinct op types.

    Plain substitution is maximally *consistent* instead -- every occurrence of
    a spelling becomes the same new spelling, in definitions and uses alike --
    which is the property that keeps the result compilable. It renames more
    than the four identifiers that have to move (embedding names like
    ``npu_recurrent_kda`` in the docs move too), and that is harmless: nothing
    outside these two extracted subtrees reads any of them, and stage 6 fails
    loudly if a name that was load-bearing did not arrive.
    """
    for old, new in rules:
        text = text.replace(old, new)
    return text


def rename(root: Path, ops: tuple[Op, ...]) -> dict[str, list[str]]:
    """Apply every op's rename rules to file contents and file names under ``root``.

    Contents first, then names bottom-up, so a renamed directory does not
    invalidate the paths of files still to be visited.

    Returns the rules applied, for the manifest. The rename is the only novel
    mechanism in this pipeline, so what it did is recorded rather than
    reconstructed from the code that did it.
    """
    rules = tuple(rule for op in ops for rule in op.rules())

    touched = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        renamed = _substitute(text, rules)
        if renamed != text:
            path.write_text(renamed, encoding="utf-8")
            touched += 1

    moved = 0
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        new_name = _substitute(path.name, rules)
        if new_name != path.name:
            path.rename(path.with_name(new_name))
            moved += 1

    log(f"  renamed: {touched} files' contents, {moved} paths")
    return {old: new for old, new in rules}


# --- 4. build -----------------------------------------------------------


def acquire_lock(timeout: float = 7200.0) -> None:
    """Serialize builds out of this one extraction.

    A lock left behind by a dead process should not block forever, so a holder
    with no ``/proc`` entry is cleared rather than waited on.
    """
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            return
        except FileExistsError:
            holder = ""
            try:
                holder = LOCK.read_text().strip()
            except OSError:
                pass
            if holder.isdigit() and not Path(f"/proc/{holder}").exists():
                log(f"  removing stale build lock from pid {holder}")
                LOCK.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise SystemExit(f"timed out waiting for build lock {LOCK}")
            log(f"  waiting for build lock held by pid {holder or '?'} ...")
            time.sleep(30)


def release_lock() -> None:
    LOCK.unlink(missing_ok=True)


def _build_env(repo: Path) -> dict[str, str]:
    """``build.sh``'s environment, with the third-party deps pointed at locally.

    ``CANN_3RD_LIB_PATH`` is where every ``cmake/third_party/*.cmake`` looks
    before reaching for the network, so pointing it at the extracted catlass
    takes a fetch out of the middle of the build.
    """
    env = dict(os.environ)
    env["CANN_3RD_LIB_PATH"] = str(WORK_DIR / "third_party")
    return env


def build_op(repo: Path, op: Op, jobs: int) -> Path:
    """Build op ``op`` as a custom vendor package; return the emitted ``.run``.

    ``--opkernel`` is deliberately **absent**, despite its name suggesting it is
    what compiles the device kernel. It sets ``ENABLE_OPKERNEL``, which
    ``build.sh`` tests *before* the ``--ops``/``--vendor_name`` branch in its
    dispatch chain, so it wins, runs ``build_kernel`` alone, and exits before
    CPack. The build reports success and produces no ``.run`` at all. Without
    it the custom-op branch sets ``-DENABLE_OPS_KERNEL=ON`` itself and then
    packages, so the kernel binary is still built.

    ``--disable-check-compatible`` because the sources are 9.2.0 and the
    toolkit is 9.0.0. The repo's own gate is already forced off in
    ``cmake/config.cmake``, but pass it so this does not start failing if that
    line changes.
    """
    cmd = [
        "bash", "build.sh", "--pkg",
        f"--ops={op.renamed_snake}",
        f"--vendor_name={op.vendor}",
        f"--soc={SOC}",
        # Both operators live under `attention/`, and this extraction holds no
        # other module. Without the flag `should_add_module` defaults to true
        # for all of them and configure dies in a directory that was never
        # extracted; naming it also turns off `ENABLE_TILING_SINK` for
        # non-attention builds, which is not this one.
        "--module=attention",
        "--disable-check-compatible",
        f"-j{jobs}",
    ]
    log(f"  {op.renamed_snake}: {' '.join(cmd)}")
    log(f"  log: {LOG}")

    with LOG.open("a") as handle:
        handle.write(f"\n===== build {op.renamed_snake} =====\n")
        handle.flush()
        proc = subprocess.run(
            cmd, cwd=repo, stdout=handle, stderr=subprocess.STDOUT,
            text=True, env=_build_env(repo),
        )
    if proc.returncode != 0:
        tail = "\n".join(LOG.read_text(errors="replace").splitlines()[-40:])
        raise SystemExit(
            f"building {op.renamed_snake} failed (exit {proc.returncode}). "
            f"Last lines of {LOG}:\n{tail}"
        )

    package = run_package(repo, op)
    if package is None:
        raise SystemExit(
            f"{op.renamed_snake}: build reported success but produced no .run "
            f"package; see {LOG}"
        )
    return package


def run_package(repo: Path, op: Op) -> Path | None:
    """The newest ``.run`` this op's build produced.

    ``build/`` first because that is where ``--pkg`` leaves it: CPack's
    ``DIR_MOVE`` moves the package to ``CPACK_PACKAGE_DIRECTORY``, the CMake
    binary dir, and ``collect_rpm_deb_package`` returns early for
    ``PACKAGE_TYPE=run``. The others are globbed because other build modes do
    write there.
    """
    candidates = [
        *(repo / "build").glob("*.run"),
        *(repo / "build_out").glob("*.run"),
        *(repo / "output").glob("*.run"),
    ]
    runs = [p for p in candidates if op.vendor in p.name] or candidates
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


# --- 5. install ---------------------------------------------------------


def vendor_dir(op: Op) -> Path:
    return OPP_ROOT / "vendors" / f"{op.vendor}_transformer"


def install_run(package: Path, op: Op) -> None:
    """Install the vendor package into this package's own ``_opp/``.

    ``--install-path`` is an **opp-root, not a prefix**: its value becomes
    ``targetdir`` and ``vendors/<vendor>/`` is appended. So this lands
    ``_opp/vendors/<vendor>_transformer/``, inside the package and therefore
    inside the venv -- nothing is written to the CANN system path.

    ``--quiet`` plus a non-interactive stdin because the installer prompts when
    a vendor dir already holds files, which would hang an unattended build.
    """
    OPP_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"  {op.renamed_snake}: installing {package.name} -> {OPP_ROOT}")
    proc = subprocess.run(
        ["bash", str(package), "--quiet", f"--install-path={OPP_ROOT}"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"installing {package} failed:\n{proc.stdout}\n{proc.stderr}"
        )


def install_wheel(repo: Path, op: Op, jobs: int) -> None:
    """Build and install ``cann_ops_transformer_<vendor>``: the Python binding.

    Only for the op whose entry point exists nowhere else. ``setup.py``
    suffixes the package name whenever ``TORCH_EXTENSION_OPS`` is set -- which
    the per-op build always sets -- and copies the per-op ``torch_extension/``
    wrapper into ``ops/<cat>/<op>/`` with an entry point that resolves it. Pure
    Python, no kernel compile, so it is seconds.
    """
    log(f"  {op.renamed_snake}: building the vendor wheel")
    with LOG.open("a") as handle:
        handle.write(f"\n===== wheel {op.renamed_snake} =====\n")
        handle.flush()
        proc = subprocess.run(
            ["bash", "build.sh", "--torch_extension",
             f"--ops={op.renamed_snake}", f"--vendor_name={op.vendor}",
             f"-j{jobs}"],
            cwd=repo, stdout=handle, stderr=subprocess.STDOUT, text=True,
            env=_build_env(repo),
        )
    if proc.returncode != 0:
        tail = "\n".join(LOG.read_text(errors="replace").splitlines()[-40:])
        raise SystemExit(
            f"{op.renamed_snake}: wheel build failed. Last lines of {LOG}:\n{tail}"
        )

    package = f"cann_ops_transformer_{op.vendor}"
    wheels = sorted(
        [*(repo / "build_out").glob(f"{package}*.whl"),
         *(repo / "torch_extension" / "dist").glob(f"{package}*.whl")],
        key=lambda p: p.stat().st_mtime,
    )
    if not wheels:
        raise SystemExit(
            f"{op.renamed_snake}: wheel build reported success but produced no "
            f"{package} wheel; see {LOG}"
        )

    log(f"  {op.renamed_snake}: installing {wheels[-1].name}")
    # sys.executable, so this lands in whichever venv ran the build -- which is
    # the one that will import it.
    run([sys.executable, "-m", "pip", "install", str(wheels[-1]),
         "--force-reinstall", "--no-deps"])


def install_fla() -> None:
    """Install the pinned ``fla_npu`` wheel if it is not already the right one.

    There is nothing to build: the published wheel carries both operators and
    their vendor OPP inside ``site-packages``. ``ref_ops.fla_npu.check_pin``
    asserts the version, so this stage only has to make the assertion pass.
    """
    from .fla_npu import check_pin

    try:
        check_pin()
    except RuntimeError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            f"ref_ops does not build this wheel: it is not on PyPI and the "
            f"source install cannot work here. Install the matching release "
            f"zip or build it from the pinned checkout -- see\n"
            f"  {PINS['fla'].url} @ {PINS['fla'].commit[:12]}\n"
            f"and then rerun this."
        ) from None
    log(f"  fla: {PINS['fla'].wheel_version} installed")


# --- 6. verify ----------------------------------------------------------


def defined_symbols(lib: Path) -> set[str]:
    out = run(["nm", "-D", "--defined-only", str(lib)]).stdout
    return {line.split()[-1] for line in out.splitlines() if line.strip()}


def verify_op(op: Op) -> str:
    """Require the **renamed** aclnn symbol; return the library's SHA256.

    The weaker checks -- a vendor directory existing, a ``libcust_opapi.so``
    being present -- are satisfied by the sibling op's package, and the dlsym
    scan takes the first match, so passing on either would produce a package
    that looks available and fails at the first launch with "not in
    libopapi.so". Requiring this op's own symbol is also what catches a rename
    that missed an identifier: the build succeeds, the symbol is absent, this
    fails.
    """
    lib = vendor_dir(op) / "op_api" / "lib" / "libcust_opapi.so"
    if not lib.is_file():
        raise SystemExit(
            f"{op.renamed_snake}: install completed but there is no {lib}"
        )
    symbols = defined_symbols(lib)
    if op.aclnn not in symbols:
        kda = sorted(s for s in symbols if "Kda" in s)
        raise SystemExit(
            f"{op.renamed_snake}: {lib} does not export {op.aclnn}.\n"
            f"Kda symbols it does export: {kda or 'none'}.\n"
            f"If the old name is in that list, the rename missed the aclnn "
            f"interface; if the list is empty, the build skipped the op."
        )
    log(f"  {op.renamed_snake}: {op.aclnn} present in {lib.name}")
    return hashlib.sha256(lib.read_bytes()).hexdigest()


# --- manifest -----------------------------------------------------------


def write_manifest(entries: dict) -> None:
    """Merge ``entries`` into the manifest, so one arm's rebuild keeps the other's.

    Rewriting it wholesale would make ``--only=cann`` erase the fla record, and
    the import-time check reads the merged document.
    """
    current = {}
    if MANIFEST.is_file():
        current = json.loads(MANIFEST.read_text())
    for key, value in entries.items():
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key].update(value)
        else:
            current[key] = value
    MANIFEST.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    log(f"  manifest: {MANIFEST}")


# --- stages, assembled --------------------------------------------------


def build_cann(ops: tuple[Op, ...], jobs: int) -> None:
    """Stages 1-6 for the CANN arm, for the ops named."""
    log("fetch")
    cann_cache = fetch("cann")
    catlass_cache = fetch("catlass")

    log("extract")
    repo = WORK_DIR / "ops-transformer"
    blobs = extract("cann", cann_cache, repo)
    # catlass goes where `cmake/third_party/catlass.cmake` looks for it:
    # `${CANN_3RD_LIB_PATH}/catlass`, which `_build_env` points at `_work`.
    catlass = WORK_DIR / "third_party" / "catlass"
    catlass_blobs = extract("catlass", catlass_cache, catlass)

    log("patch")
    patches = patch("cann", repo)

    log("rename")
    # Every op's rules, not just the requested ops'. The extraction holds both
    # operator directories whichever one is being built, and `--ops=X` with a
    # half-renamed tree leaves the sibling under its upstream name -- which is
    # the collision this package exists to dissolve, reintroduced by
    # `--ops=chunk_kda_fwd`. Renaming both costs nothing; the vendor package
    # still contains only the op that was built.
    rules = rename(repo, OPS)

    acquire_lock()
    try:
        log("build")
        packages = {op.snake: build_op(repo, op, jobs) for op in ops}

        log("install")
        for op in ops:
            install_run(packages[op.snake], op)
            if op.needs_wheel:
                install_wheel(repo, op, jobs)
    finally:
        release_lock()

    log("verify")
    vendors = {
        f"{op.vendor}_transformer": {
            "op": op.snake,
            "renamed_op_type": op.renamed_camel,
            "renamed_op_file": op.renamed_snake,
            "aclnn_symbol": op.aclnn,
            "needs_wheel": op.needs_wheel,
            "libcust_opapi_sha256": verify_op(op),
        }
        for op in ops
    }

    write_manifest({
        "pins": {
            "cann": {"url": PINS["cann"].url, "commit": PINS["cann"].commit},
            "catlass": {
                "url": PINS["catlass"].url, "commit": PINS["catlass"].commit
            },
        },
        "rename_rules": rules,
        "patches": patches,
        "cann_source_root": str(repo),
        "cann_blobs": blobs,
        "catlass_blobs": catlass_blobs,
        "vendors": vendors,
    })


def build_fla() -> None:
    log("fla")
    install_fla()
    write_manifest({
        "pins": {
            "fla": {
                "url": PINS["fla"].url,
                "commit": PINS["fla"].commit,
                "wheel_version": PINS["fla"].wheel_version,
            },
        },
    })


def check() -> int:
    """Report what is current and what is not; build nothing. Exit status."""
    from . import cann as cann_mod
    from . import fla_npu as fla_mod

    print(f"work dir   : {WORK_DIR}")
    print(f"opp root   : {OPP_ROOT}")
    print(f"manifest   : {MANIFEST} ({'present' if MANIFEST.is_file() else 'absent'})")
    print()

    ok = True
    available, why = fla_mod.available()
    print(f"fla_npu.chunk_kda / recurrent_kda : "
          f"{'ok' if available else 'MISSING'}")
    if not available:
        print(f"  {why}")
        ok = False

    for op in OPS:
        available, why = cann_mod.available(op.snake)
        print(f"cann.{op.snake:<16} ({op.aclnn:<28}) : "
              f"{'ok' if available else 'MISSING'}")
        if not available:
            print(f"  {why}")
            ok = False

    if not ok:
        print()
        print("build it with: python -m ref_ops.build")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report status and exit without building")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the artifact is already current")
    parser.add_argument("--only", choices=("fla", "cann"),
                        help="build one arm instead of both")
    parser.add_argument("--ops", default="",
                        help="comma-separated CANN ops, by their upstream snake "
                             "name; default is all of them")
    parser.add_argument("-j", "--jobs", type=int,
                        default=max(1, (os.cpu_count() or 8) // 2),
                        help="parallel compile jobs")
    args = parser.parse_args(argv)

    if args.check:
        return check()

    requested = OPS
    if args.ops:
        names = [name.strip() for name in args.ops.split(",") if name.strip()]
        unknown = [name for name in names if name not in OPS_BY_SNAKE]
        if unknown:
            parser.error(f"unknown ops {unknown}; known: {sorted(OPS_BY_SNAKE)}")
        requested = tuple(OPS_BY_SNAKE[name] for name in names)

    if args.only != "cann":
        build_fla()

    if args.only != "fla":
        from . import cann as cann_mod

        todo = tuple(
            op for op in requested
            if args.force or not cann_mod.available(op.snake)[0]
        )
        if not todo:
            log(f"cann: {[op.snake for op in requested]} already built; "
                f"--force to rebuild")
        else:
            build_cann(todo, args.jobs)

    print()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
