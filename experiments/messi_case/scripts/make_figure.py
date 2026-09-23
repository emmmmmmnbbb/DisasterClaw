"""Render paired MESSI help and harm examples from frozen outputs."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1] / 'shuanglan' / 'polished' / 'figures'


def main():
    rows = {r['sample_id']:r for r in csv.DictReader((ROOT/'results'/'per_sample.csv').open())}
    cases = [('100_0042_r16','Correction with low image'),
             ('100_0002_r02','Harm with low image')]
    fig,axes = plt.subplots(2,3,figsize=(11.6,6.8),layout='constrained')
    for i,(sid,label) in enumerate(cases):
        r=rows[sid]
        for j,(branch,typ,height) in enumerate([('A','high_marked','high'),
                                                ('B','high_crop','high'),
                                                ('C','low_crop','low')]):
            p=ROOT/'derived'/'inputs'/f'{sid}_{typ}.png'
            ax=axes[i,j]
            ax.imshow(Image.open(p))
            ax.set_xticks([]);ax.set_yticks([])
            answer=r[f'{branch}_answer'].upper()
            correct=r[f'{branch}_correct']=='1'
            ax.set_title(f'{branch}: {"repeat" if branch=="A" else "crop" if branch=="B" else "new image"}\n'
                         f'{answer} ({"correct" if correct else "incorrect"})',
                         fontsize=10,color='#147d52' if correct else '#b33030')
            if j==0:
                ax.set_ylabel(f'{sid}\n{label}\nGT: {r["truth"].upper()}',
                              fontsize=10,rotation=90,labelpad=9)
    PAPER.mkdir(parents=True,exist_ok=True)
    out=PAPER/'messi_case_pairs.png'
    fig.savefig(out,dpi=240,bbox_inches='tight',facecolor='white')
    print(out)


if __name__=='__main__': main()
