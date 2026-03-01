
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from utils import ensure_dir, save_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--blender_path", default="blender")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--scale", type=float, default=1.0)
    args = p.parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.scale <= 0:
        raise SystemExit("--scale must be > 0")
    if not os.path.isdir(args.in_dir):
        raise SystemExit(f"--in_dir does not exist: {args.in_dir}")

    ensure_dir(args.out_dir)
    script_path = Path(__file__).resolve().parent / "blender_fbx_to_obj.py"
    cmd = [
        args.blender_path,
        "--background",
        "--python", str(script_path),
        "--",
        "--in_dir", args.in_dir,
        "--out_dir", args.out_dir,
        "--scale", str(args.scale),
    ]
    if args.limit and args.limit > 0:
        cmd += ["--limit", str(args.limit)]

    print("Running Blender:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        print("[ERROR] Blender executable not found. Pass --blender_path to blender.exe")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Blender finished with non-zero exit code: {exc.returncode}")
        sys.exit(exc.returncode)

    save_json({
        "in_dir": os.path.abspath(args.in_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "limit": args.limit,
        "scale": args.scale
    }, os.path.join(args.out_dir, "conversion_meta.json"))

    print("Done:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
