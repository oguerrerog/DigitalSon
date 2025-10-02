import os
import json
import time
import hashlib
import zipfile

def sha256_of_file(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def export_mind(data_dir="data", out_path=None):
    """
    Package the entire data directory into a zip and return path + sha256.
    Use to migrate a mind to another machine.
    """
    out_path = out_path or f"mind_export_{int(time.time())}.zip"
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(data_dir):
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, os.path.dirname(data_dir))
                z.write(full, arcname=rel)
    digest = sha256_of_file(out_path)
    # write metadata for convenience
    meta = {"exported_at": time.time(), "zip": out_path, "sha256": digest}
    meta_path = out_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_path, digest, meta_path
