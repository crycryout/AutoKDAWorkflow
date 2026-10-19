#!/usr/bin/env python3
"""Inventory the metrics actually present in a Thor NCU report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for path in sorted(Path('/opt/nvidia/nsight-compute').glob('*/extras/python'),reverse=True):
        if (path/'ncu_report.py').is_file():sys.path.insert(0,str(path));break
    import ncu_report
    report=ncu_report.load_report(str(a.input));actions=[]
    for ri in range(report.num_ranges()):
        region=report.range_by_idx(ri)
        for ai in range(region.num_actions()):
            action=region.action_by_idx(ai);metrics={}
            for name in action.metric_names():
                metric=action[name]
                try: metrics[name]=dict(value=metric.value(),unit=metric.unit())
                except Exception as e: metrics[name]=dict(value=None,error=str(e))
            actions.append(dict(range_index=ri,action_index=ai,name=action.name(),metrics=metrics))
    result=dict(input=str(a.input),sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),actions=actions,
                note='Report metrics are profiler observations. Missing/unreadable values remain unknown; instrumented duration is not paired performance evidence.')
    if not actions:raise RuntimeError('No kernel actions in NCU report')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,default=str)
    print(json.dumps(dict(output=str(a.output),actions=len(actions),metric_counts=[len(x['metrics']) for x in actions])))

if __name__=='__main__':main()
