"""Verify downloaded public Path A/B/C 100-m and 30-m archives."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = json.loads((ROOT / "manifests/public_sample_files.json").read_text())
id_by_path = {x["path"]: x["id"] for x in files}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for group in "ABC":
        for altitude in ("100", "30"):
            base = ROOT / "raw/public_paths" / group / altitude
            pose_path = base / "6DOF.csv"
            pose = list(csv.DictReader(pose_path.open()))
            frame_names = {r["ImageName"] for r in pose}
            for kind, ext in (("images", ".JPG"), ("annotations", ".png")):
                path = base / (kind + ".zip")
                with zipfile.ZipFile(path) as z:
                    bad = z.testzip()
                    names = {Path(x).stem for x in z.namelist() if x.endswith(ext)}
                if bad or names != frame_names:
                    raise RuntimeError(f"Archive mismatch: {path}, bad={bad}, count={len(names)}")
                source = f"Train and Val/{kind}/Agamim/Path/{group}/{altitude}.zip"
                rows.append({"group": group, "altitude": altitude, "kind": kind,
                             "drive_id": id_by_path[source], "path": str(path),
                             "bytes": path.stat().st_size, "sha256": digest(path),
                             "frame_count": len(names), "crc": "pass"})
            source = f"Train and Val/6DOF/Agamim/Path/{group}/{altitude}/6DOF.csv"
            rows.append({"group": group, "altitude": altitude, "kind": "pose",
                         "drive_id": id_by_path[source], "path": str(pose_path),
                         "bytes": pose_path.stat().st_size, "sha256": digest(pose_path),
                         "frame_count": len(frame_names), "crc": "n/a"})
            print(group, altitude, len(frame_names), flush=True)
    out = ROOT / "manifests/public_path_inventory.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
