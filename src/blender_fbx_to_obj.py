import bpy
import os
import sys
import argparse

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # удалить всё на всякий
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def export_obj(out_path: str):
    # В Blender 5 enum изменился: "-Z" -> "NEGATIVE_Z"
    # Делаем try для совместимости
    kwargs = dict(
        filepath=out_path,
        use_selection=False,
        use_mesh_modifiers=True,
        use_materials=False,
    )
    try:
        bpy.ops.wm.obj_export(**kwargs, forward_axis='NEGATIVE_Z', up_axis='Y')
    except TypeError:
        # старые версии blender (если вдруг)
        bpy.ops.wm.obj_export(**kwargs, forward_axis='-Z', up_axis='Y')

def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args(argv)

    in_dir = args.in_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)

    # соберём все FBX рекурсивно
    fbx_files = []
    for root, _, files in os.walk(in_dir):
        for fn in files:
            if fn.lower().endswith(".fbx"):
                fbx_files.append(os.path.join(root, fn))

    print(f"[INFO] Found {len(fbx_files)} FBX files under {in_dir}")

    ok, fail = 0, 0
    for fbx_path in fbx_files:
        rel = os.path.relpath(fbx_path, in_dir)
        rel_noext = os.path.splitext(rel)[0]
        out_path = os.path.join(out_dir, rel_noext + ".obj")
        ensure_dir(os.path.dirname(out_path))

        try:
            reset_scene()
            bpy.ops.import_scene.fbx(filepath=fbx_path, global_scale=args.scale)

            # применим трансформы, чтобы сетка норм была
            for obj in bpy.context.scene.objects:
                if obj.type == "MESH":
                    bpy.context.view_layer.objects.active = obj
                    obj.select_set(True)
                    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
                    obj.select_set(False)

            export_obj(out_path)
            ok += 1
        except Exception as e:
            print(f"[WARN] export failed: {fbx_path} | {e}")
            fail += 1

    print(f"[DONE] ok={ok}, fail={fail}")

if __name__ == "__main__":
    main()