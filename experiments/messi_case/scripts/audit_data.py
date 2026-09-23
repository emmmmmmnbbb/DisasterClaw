"""Validate downloaded MESSI ZIP/CSV correspondence and record provenance."""
import csv
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDS = {
    '100_0001': ('19cYDM69p9aJaz_DE09Dc-wAwbnFv8Cqy','1zHDhkPeqKXi1D2UJauk2nQ4oFWKlNN38','1U8K8JxNVXFW1wiXoPxoDtrS3IOEnsNtd'),
    '100_0002': ('1es70lB8fmmGong2LObrE_xGXjQIxAnie','11zUQOirPmSQ0AcoOpAruoK19DTpKu4u9','138Z8tKtzLoauPJ2VN50yU5Nx9cJOSiyw'),
    '100_0003': ('11v4ETuSJ5DW0bWLk5Hz-5UIf-jNn7vZw','1kEB6-uJlxu6VHbadmaEhDkGUg718uVKY','1Um5X9gGnqfQiNc4wIaZQF7BJ3VgpEO08'),
    '100_0004': ('1Pc0P4CD3nHPihp3pqrPazXzgwkcCopER','1Jndf4bCX1_vSUoZ8nFvEGPM4yvkd0Q_A','1byYAsq6qiXPaQHdio5wculYbBrj39XxV'),
    '100_0042': ('1utp1OtRtSSFlzTgYNMvOU21XMKMqUXyP','1HOQXHoR2TxkmVy7mGQk1QLY8Ktc7QrsJ','1Mo3LaLoq6kjXnlklkZlKURlwUAnMtHNe'),
    '100_0043': ('1SIeleAYaPhWhk2uUMILrKS2oTNP9Xq9V','1Mf3RzLezaw0ua7gNW3fMOAQymGVYlqoJ','1G0gDwoBO2n212NIvlmDPRWR6iw5fw2O6'),
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def main():
    rows=[]
    for seq,ids in IDS.items():
        raw=ROOT/'raw'/seq
        poses=list(csv.DictReader((raw/'6DOF.csv').open()))
        pose_names={r['ImageName'] for r in poses}
        names={}
        for kind,ext in [('images','.JPG'),('annotations','.png')]:
            path=raw/f'{kind}.zip'
            with zipfile.ZipFile(path) as z:
                bad=z.testzip()
                members={Path(n).stem for n in z.namelist() if n.endswith(ext)}
            if bad:raise ValueError(f'ZIP CRC failed: {path}:{bad}')
            names[kind]=members
            rows.append({'sequence':seq,'kind':kind,'drive_file_id':ids[0 if kind=='images' else 1],
                         'filename':path.name,'bytes':path.stat().st_size,
                         'sha256':digest(path),'count':len(members),'zip_crc':'pass'})
        path=raw/'6DOF.csv'
        rows.append({'sequence':seq,'kind':'pose','drive_file_id':ids[2],
                     'filename':path.name,'bytes':path.stat().st_size,
                     'sha256':digest(path),'count':len(poses),'zip_crc':'n/a'})
        if names['images']!=names['annotations'] or names['images']!=pose_names:
            raise ValueError(f'Frame set mismatch: {seq}')
        print(seq,'paired',len(pose_names),flush=True)
    out=ROOT/'manifests'/'data_inventory.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(out)


if __name__=='__main__':main()
