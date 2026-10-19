#!/usr/bin/env python3
"""Export Nsight SQLite GPU/runtime/NVTX records for the upstream trace tools.

This is a derived view, not a reconstructed Torch stack trace. Raw Nsight IDs
and integer timestamps remain in args; the source SQLite is opened read-only.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

GEMM_REGEX=r'(?i)^(?!.*(?:convol|conv2d|winograd|attention|prefill|fmha|merge_states)).*(?:gemm|gemv|matmul|nvjet|splitkreduce|warp_splitk|small_k\()'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def union_ns(intervals):
    total = 0
    right = None
    for start, end in sorted(intervals):
        if right is None or start > right:
            total += end - start
        elif end > right:
            total += end - right
        right = max(right or end, end)
    return total


def convert(source, output, *, framework='phyai', request_regex=None,
            gemm_regex=GEMM_REGEX,
            window_regex=None, window_index=0):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.exists():
        raise ValueError('Use a new output path; retain the original trace')
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings = dict(db.execute('SELECT id,value FROM StringIds')) if 'StringIds' in tables else {}
        def rows(table):
            return [dict(r) for r in db.execute('SELECT * FROM ' + table)] if table in tables else []
        kernels = rows('CUPTI_ACTIVITY_KIND_KERNEL')
        runtimes = rows('CUPTI_ACTIVITY_KIND_RUNTIME')
        nvtx = rows('NVTX_EVENTS')
        copies = rows('CUPTI_ACTIVITY_KIND_MEMCPY')
        memsets = rows('CUPTI_ACTIVITY_KIND_MEMSET')
    if not kernels:
        raise ValueError('No GPU kernels in this SQLite export')
    selected_window=None
    if window_regex:
        pattern=re.compile(window_regex)
        windows=sorted([r for r in nvtx if r.get('end') is not None and
                        pattern.search(r.get('text') or strings.get(r.get('textId'),''))],key=lambda r:r['start'])
        if window_index < 0 or window_index >= len(windows):
            raise ValueError('Selected NVTX window does not exist')
        selected_window=windows[window_index]
        left,right=selected_window['start'],selected_window['end']
        if any(r['start']<right and r['end']>left and (r['start']<left or r['end']>right) for r in kernels):
            raise ValueError('GPU kernel crosses selected NVTX boundary; verify a synchronized window')
        def within(items):
            return [r for r in items if r.get('end') is not None and r['start']>=left and r['end']<=right]
        kernels,runtimes,nvtx,copies,memsets=map(within,(kernels,runtimes,nvtx,copies,memsets))
        if not kernels: raise ValueError('No GPU kernels in selected window')
    valid = [r for r in kernels + runtimes + nvtx + copies + memsets
             if r.get('start') is not None and r.get('end') is not None and r['end'] >= r['start']]
    origin = min(r['start'] for r in valid)
    events = []
    for category, items in [('kernel', kernels), ('cuda_runtime', runtimes),
                            ('user_annotation', nvtx), ('gpu_memcpy', copies), ('gpu_memset', memsets)]:
        for row in items:
            if row.get('end') is None or row['end'] < row['start']:
                continue
            args = dict(row)
            gpu = category in ('kernel', 'gpu_memcpy', 'gpu_memset')
            if category == 'kernel':
                name = strings.get(row.get('demangledName'), strings.get(row.get('shortName'), '<unknown kernel>'))
            elif category == 'cuda_runtime':
                name = strings.get(row.get('nameId'), '<unknown CUDA runtime>')
            elif category == 'user_annotation':
                name = row.get('text') or strings.get(row.get('textId'), '<unknown NVTX>')
            else:
                name = category
            if gpu:
                args.update(device=row['deviceId'], stream=row['streamId'], correlation=row.get('correlationId'))
                pid, tid = f"GPU {row['deviceId']} process {row.get('globalPid', 'unknown')}", row['streamId']
            else:
                pid, tid = 'CPU / Nsight global thread IDs', row.get('globalTid', 0)
            events.append(dict(ph='X', cat=category, name=name, pid=pid, tid=tid,
                               ts=(row['start']-origin)/1000, dur=(row['end']-row['start'])/1000,
                               args=args))
    document = dict(traceEvents=events, displayTimeUnit='ms',
                    thor_provenance=dict(source=str(source), source_sha256=sha256(source),
                        framework=framework, origin_ns=origin, selected_window=selected_window,
                        note='Derived Nsight view. CUDA runtime/NVTX are not Python source attribution.'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(document, f, separators=(',', ':'))
    names = Counter(strings.get(r.get('demangledName'), '<unknown>') for r in kernels)
    summary = dict(source=str(source), source_sha256=document['thor_provenance']['source_sha256'],
                   output=str(output), output_sha256=sha256(output), kernel_count=len(kernels),
                   origin_ns=origin, runtime_count=len(runtimes), nvtx_count=len(nvtx),
                   unique_kernel_count=len(names), gemm_classifier_regex=gemm_regex, request_windows=[])
    if request_regex:
        pattern, gemm = re.compile(request_regex), re.compile(gemm_regex)
        for scope in [e for e in events if e['cat']=='user_annotation' and pattern.search(e['name'])]:
            start, end = scope['args']['start'], scope['args']['end']
            overlapping = [e for e in events if e['cat']=='kernel' and e['args']['start'] < end and e['args']['end'] > start]
            contained = [e for e in overlapping if e['args']['start'] >= start and e['args']['end'] <= end]
            intervals = [(e['args']['start'], e['args']['end']) for e in contained]
            gemm_ns = sum(b-a for e,(a,b) in zip(contained,intervals) if gemm.search(e['name']))
            wall = end-start
            summary['request_windows'].append(dict(name=scope['name'], nvtx_wall_ms=wall/1e6,
                kernel_count=len(contained), boundary_crossing_kernels=len(overlapping)-len(contained),
                kernel_sum_ms=sum(b-a for a,b in intervals)/1e6, gpu_busy_union_ms=union_ns(intervals)/1e6,
                gpu_span_ms=(max(b for a,b in intervals)-min(a for a,b in intervals))/1e6 if intervals else 0,
                gemm_sum_ms=gemm_ns/1e6, gemm_sum_over_nvtx_pct=100*gemm_ns/wall if wall else None,
                qualification='NVTX host interval; synchronized model E2E requires callsite verification'))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--framework',choices=['phyai','flashrt','thor'],default='phyai')
    p.add_argument('--request-regex'); p.add_argument('--gemm-regex',default=GEMM_REGEX)
    p.add_argument('--window-regex');p.add_argument('--window-index',type=int,default=0)
    a=p.parse_args()
    result=convert(a.input,a.output,framework=a.framework,request_regex=a.request_regex,gemm_regex=a.gemm_regex,
                   window_regex=a.window_regex,window_index=a.window_index)
    report=a.output.with_suffix(a.output.suffix+'.source.json')
    with report.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
