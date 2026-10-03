#!/usr/bin/env python3
"""Build a local MLX wheel with the native group-split patch. No install, no push."""
import argparse, hashlib, json, os, subprocess, sys, sysconfig
from pathlib import Path

BASE = "0e3ff3643b1c3719f78814b98e0d222afbad867c"
PATCH_SHA = "e852ed31c27e711093d420de8913e95e6db6bae0c0e45ff58974e7f41cfd748b"
PATCH = Path(__file__).resolve().parents[1] / "omlx/patches/native_mlx_group_split.patch"
URL = "https://github.com/ml-explore/mlx.git"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd, cwd=None, env=None, log=None, check=True):
    r = subprocess.run([*map(str, cmd)], cwd=cwd, env=env,
                       stdout=log or subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if check and r.returncode:
        print(f"failed ({r.returncode}): {' '.join(map(str, cmd))}", file=sys.stderr)
        sys.exit(r.returncode)
    return r


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--work-dir", required=True)
    a.add_argument("--python", default=sys.executable)
    a.add_argument("--cpu-only", action="store_true")
    a.add_argument("--jobs", type=int, default=2)
    o = a.parse_args()
    if o.jobs < 1:
        sys.exit("--jobs must be >= 1")

    if sha(PATCH) != PATCH_SHA:
        sys.exit("patch hash mismatch")
    inc = subprocess.check_output([o.python, "-c", "import sysconfig;print(sysconfig.get_config_var('INCLUDEPY'))"],
                                  text=True).strip()
    if not (Path(inc) / "Python.h").is_file():
        sys.exit(f"Python.h missing in {inc}")
    if not o.cpu_only and run(["xcrun", "--find", "metal"], check=False).returncode:
        sys.exit("metal toolchain missing; use --cpu-only explicitly (CPU wheel has no GPU inference)")

    work = Path(o.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    src, venv, wheels = work / "mlx", work / "venv", work / "wheels"
    git = lambda *x, **k: run(["git", *x], cwd=src, **k)
    if not src.exists():
        src.mkdir()
        git("init")
        git("remote", "add", "origin", URL)
        git("fetch", "--depth", "1", "origin", BASE)
        git("checkout", "--detach", BASE)
    patch_arg = ["apply", "--check", "--reverse", PATCH]
    applied = git(*patch_arg, check=False).returncode == 0
    status = git("status", "--porcelain", "--untracked-files=all").stdout.strip()
    if status:
        if not applied:
            sys.exit("refuse: dirty MLX tree without exact patch applied")
        if any(l.startswith("??") for l in status.splitlines()):
            sys.exit("refuse: untracked files in MLX tree")
        if hashlib.sha256(git("diff", "--binary").stdout.encode()).hexdigest() != PATCH_SHA:
            sys.exit("refuse: unexpected MLX tree changes")
    head = git("rev-parse", "HEAD").stdout.strip()
    if head != BASE:
        if status:
            sys.exit("refuse: dirty tree on unexpected commit")
        git("fetch", "--depth", "1", "origin", BASE)
        git("checkout", "--detach", BASE)
    if not applied:
        git("apply", "--check", PATCH)
        git("apply", PATCH)
        git(*patch_arg)

    ver = "import sys;print('%d.%d' % sys.version_info[:2])"
    want = subprocess.check_output([o.python, "-c", ver], text=True).strip()
    if venv.exists():
        have = subprocess.run([venv / "bin/python", "-c", ver], capture_output=True, text=True).stdout.strip()
        if have != want:
            sys.exit(f"refuse: venv python {have or '?'} != {want}; remove {venv}")
    else:
        run([o.python, "-m", "venv", venv])
    wheels.mkdir(exist_ok=True)
    env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=str(o.jobs))
    env["CMAKE_ARGS"] = f"-DPython_INCLUDE_DIR={inc} -DMLX_BUILD_METAL={'OFF' if o.cpu_only else 'ON'}"
    with open(work / "build.log", "a") as log:
        run([venv / "bin/python", "-m", "pip", "wheel", "--no-cache-dir", "--wheel-dir", wheels,
             "--no-deps", src], env=env, log=log)

    print(json.dumps({"base": BASE, "patch_sha256": PATCH_SHA, "cpu_only": o.cpu_only,
                      "python": o.python,
                      "wheels": {str(w): sha(w) for w in sorted(wheels.glob("*.whl"))}}, indent=2))


if __name__ == "__main__":
    main()
