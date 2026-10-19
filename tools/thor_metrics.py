#!/usr/bin/env python3
"""Thor workload FLOPs, unified-memory inventory, and paired E2E comparison."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics


def compute(rows, elapsed_ms=None, dense_tflops=None):
    result=[]
    for row in rows:
        dims = row.get('shape', [row.get(k) for k in ('m','n','k')])
        if len(dims)!=3 or any(not isinstance(v,int) or v <= 0 for v in dims):
            raise ValueError('Every row needs positive integer M/N/K')
        count=row.get('calls_per_request')
        if count is not None and (not isinstance(count,int) or count <= 0):
            raise ValueError('calls_per_request must be a positive integer or absent')
        m,n,k=dims
        result.append(dict(id=row['id'],shape=dims,gemm_flops_per_call=2*m*n*k,
                           bias_adds_per_call=m*n if row.get('bias') else 0,calls_per_request=count))
    total=sum(r['gemm_flops_per_call']*r['calls_per_request'] for r in result) if all(r['calls_per_request'] is not None for r in result) else None
    if elapsed_ms is not None and elapsed_ms <= 0: raise ValueError('elapsed_ms must be positive')
    if dense_tflops is not None and dense_tflops <= 0: raise ValueError('dense_tflops must be positive')
    return dict(rows=result,gemm_flops_per_request=total,
                achieved_gemm_tflops=total/(elapsed_ms*1e9) if total is not None and elapsed_ms else None,
                gemm_mfu=total/(elapsed_ms*1e9*dense_tflops) if total is not None and elapsed_ms and dense_tflops else None,
                scope='GEMM subset only. Supply observed call counts and synchronized elapsed time; overlapping durations are not a denominator. Dense peak must match dtype and power state.')


def compare(data):
    contracts=data.get('contracts',{})
    if not contracts.get('baseline') or contracts.get('baseline') != contracts.get('candidate'):
        raise ValueError('Matching frozen workload/precision/graph/timing contract hashes are required')
    pairs=data['pairs']
    valid=[p for p in pairs if p.get('valid') is True and p.get('profiled') is False]
    logs=[]
    for row in valid:
        a,b=row['baseline_e2e_ms'],row['candidate_e2e_ms']
        if not math.isfinite(a) or not math.isfinite(b) or a<=0 or b<=0: raise ValueError('Timings must be finite and positive')
        logs.append(math.log(b/a))
    if len(logs)<2: raise ValueError('At least two valid unprofiled pairs are needed')
    rng=random.Random(20261008)
    boots=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(5000))
    interval=[boots[125],boots[4874]]
    return dict(valid_pairs=len(logs),excluded_pairs=len(pairs)-len(logs),
        candidate_over_baseline=math.exp(statistics.mean(logs)),ratio_ci95=interval,
        timing_signal='improvement' if interval[1]<1 else 'regression' if interval[0]>1 else 'inconclusive',
        minimum_20_pairs_met=len(logs)>=20,
        promotion='Requires matched platform/A-A, numerical, graph/stream, NCU and actual-model quality evidence. This timing report alone does not promote a candidate.')


def capacity(snapshot):
    memory=snapshot['meminfo_kib']
    return dict(platform='Thor unified CPU/GPU memory',total_gib=memory['MemTotal']/1024**2,
                available_gib=memory['MemAvailable']/1024**2,
                processes=snapshot.get('gpu_processes'),
                note='MemAvailable is a point-in-time system budget. GPU/process memory shares the same RAM and must not be added to MemTotal. KV capacity requires actual backend cache layout and post-graph allocation logs.')


def main():
    p=argparse.ArgumentParser(description=__doc__); s=p.add_subparsers(dest='mode',required=True)
    for name in ('compute','compare','capacity'):
        q=s.add_parser(name);q.add_argument('--input',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        if name=='compute': q.add_argument('--elapsed-ms',type=float);q.add_argument('--dense-tflops',type=float)
    a=p.parse_args(); data=json.loads(a.input.read_text())
    if a.mode=='compute': result=compute(data if isinstance(data,list) else data['workloads'],a.elapsed_ms,a.dense_tflops)
    elif a.mode=='compare': result=compare(data)
    else: result=capacity(data)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
