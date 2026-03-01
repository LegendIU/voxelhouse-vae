import argparse
import os
import sys

import bpy


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_obj(out_path: str) -> None:
    # Blender 5 renamed enum "-Z" to "NEGATIVE_Z". Keep compatibility.
    kwargs = dict(
        filepath=out_path,
        use_selection=False,
        use_mesh_modifiers=True,
        use_materials=False,
    )
    try:
        bpy.ops.wm.obj_export(**kwargs, forward_axis="NEGATIVE_Z", up_axis="Y")
    except TypeError:
        bpy.ops.wm.obj_export(**kwargs, forward_axis="-Z", up_axis="Y")
    except AttributeError:
        bpy.ops.export_scene.obj(
            filepath=out_path,
            use_selection=False,
            axis_forward="-Z",
            axis_up="Y",
            use_materials=False,
        )


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="0 = convert all files")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    fbx_files: list[str] = []
    for root, _, files in os.walk(args.in_dir):
        for fn in files:
            if fn.lower().endswith(".fbx"):
                fbx_files.append(os.path.join(root, fn))
    fbx_files.sort()

    if args.limit and args.limit > 0:
        fbx_files = fbx_files[:args.limit]

    print(f"[INFO] Found {len(fbx_files)} FBX files under {args.in_dir}")

    ok, fail = 0, 0
    for i, fbx_path in enumerate(fbx_files, start=1):
        rel = os.path.relpath(fbx_path, args.in_dir)
        rel_noext = os.path.splitext(rel)[0]
        out_path = os.path.join(args.out_dir, rel_noext + ".obj")
        ensure_dir(os.path.dirname(out_path))

        try:
            reset_scene()
            bpy.ops.import_scene.fbx(filepath=fbx_path, global_scale=args.scale)

            for obj in bpy.context.scene.objects:
                if obj.type != "MESH":
                    continue
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
                obj.select_set(False)

            export_obj(out_path)
            ok += 1
            print(f"[OK] ({i}/{len(fbx_files)}) {fbx_path}")
        except Exception as exc:
            print(f"[WARN] export failed ({i}/{len(fbx_files)}): {fbx_path} | {exc}")
            fail += 1

    print(f"[DONE] ok={ok}, fail={fail}")


if __name__ == "__main__":
    main()
