"""Download the public MESSI descend sequences listed in audit_data.IDS."""
import argparse
from pathlib import Path

import gdown

from audit_data import IDS, ROOT


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sequences',nargs='+',choices=sorted(IDS),default=sorted(IDS))
    args=ap.parse_args()
    for seq in args.sequences:
        out=ROOT/'raw'/seq
        out.mkdir(parents=True,exist_ok=True)
        for kind,fid in zip(('images.zip','annotations.zip','6DOF.csv'),IDS[seq]):
            target=out/kind
            if target.exists():
                print('EXISTS',target)
                continue
            print('DOWNLOADING',target,flush=True)
            if gdown.download(id=fid,output=str(target),quiet=False) is None:
                raise RuntimeError(f'Download failed: {fid}')


if __name__=='__main__':main()
