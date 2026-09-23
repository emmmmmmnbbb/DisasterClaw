"""Record every inspected grid candidate omitted from the formal case."""
import csv
import json

from audit_data import ROOT


def main():
    chosen={x['candidate_id'] for x in json.loads((ROOT/'manifests'/'formal_selection.json').read_text())}
    rows=[]
    for seq in ('100_0002','100_0003','100_0042'):
        for line in (ROOT/'derived'/seq/'vegetation'/'candidates.jsonl').read_text().splitlines():
            x=json.loads(line)
            if x['id'] in chosen:continue
            rows.append({'sequence_id':seq,'high_frame_id':x['high_frame'],
                         'low_frame_id':x['low_frame'],'roi_id':x['id'],
                         'reason':('ambiguous_or_discordant_label_fraction' if x['status']=='review'
                                   else 'outside_preselected_four_per_sequence'),
                         'candidate_status':x['status']})
    out=ROOT/'manifests'/'exclusions.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(len(rows),out)


if __name__=='__main__':main()
