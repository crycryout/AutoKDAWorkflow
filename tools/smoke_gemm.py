#!/usr/bin/env python3
"""Verify a manifest-defined BF16 GEMM C ABI on Thor."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import platform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--capture-only',action='store_true')
    p.add_argument('--trace',type=Path);a=p.parse_args()
    root=Path(os.environ['KDA_THOR_ROOT']).resolve()
    if platform.machine()!='aarch64': raise RuntimeError('Thor ARM64 execution required')
    import torch
    if torch.cuda.get_device_capability()!=(11,0): raise RuntimeError('SM110 required')
    torch.backends.cuda.matmul.allow_tf32=False
    from thor_kda.semantics import load
    manifest=load(a.candidate)
    if manifest['layout']!='row_row': raise ValueError('This verifier supports the declared row_row ABI only; supply a layout-specific verifier for other ABIs')
    so=(root/manifest['output']).resolve()
    if not so.is_relative_to(root):raise ValueError('Binary must be task-local')
    tolerance=manifest.get('tolerance_relative_l2')
    if not isinstance(tolerance,(float,int)) or isinstance(tolerance,bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Set a finite positive tolerance_relative_l2 from the task numerical contract')
    tolerance=float(tolerance)
    lib=C.CDLL(str(so));P,I=C.c_void_p,C.c_int
    lib.ct_error.restype=C.c_char_p;lib.ct_create.argtypes=[I,I,I,P,P,P,P,P];lib.ct_create.restype=P
    lib.ct_run.argtypes=[P,P,P,P,P,P];lib.ct_run.restype=I;lib.ct_destroy.argtypes=[P]
    stream=torch.cuda.Stream();rows=[]
    def checked(out,reference):
        torch.cuda.synchronize()
        relative=float((out.float()-reference).norm()/reference.norm().clamp_min(1e-12))
        return dict(finite=bool(torch.isfinite(out).all()),relative_l2=relative,
                    max_abs=float((out.float()-reference).abs().max()),passed=bool(torch.isfinite(out).all()) and relative<tolerance)
    with torch.inference_mode(),torch.cuda.stream(stream):
        torch.manual_seed(123)
        primary=[manifest['m'],manifest['n'],manifest['k']]
        shapes=[primary] if a.capture_only else [primary]+[s for s in manifest.get('validation_shapes',[]) if s != primary]
        for m,n,k in shapes:
            x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
            w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)
            b=torch.randn(n,device='cuda',dtype=torch.bfloat16) if manifest['bias'] else None
            out=torch.full((m,n),float('nan'),device='cuda',dtype=torch.bfloat16)
            plan=lib.ct_create(m,n,k,x.data_ptr(),w.data_ptr(),b.data_ptr() if b is not None else 0,out.data_ptr(),stream.cuda_stream)
            if not plan: raise RuntimeError(lib.ct_error().decode())
            def call():
                code=lib.ct_run(plan,x.data_ptr(),w.data_ptr(),b.data_ptr() if b is not None else 0,out.data_ptr(),torch.cuda.current_stream().cuda_stream)
                if code: raise RuntimeError(lib.ct_error().decode())
            try:
                call()
                if a.capture_only:
                    torch.cuda.synchronize();continue
                ref=x.float()@w.float().T+(b.float() if b is not None else 0);initial=checked(out,ref)
                previous=out.clone()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): call()
                x.add_(0.25);out.fill_(float('nan'))
                graph.replay()
                ref=x.float()@w.float().T+(b.float() if b is not None else 0);replay=checked(out,ref)
                changed=not torch.equal(previous,out)
                rows.append(dict(shape=[m,n,k],nondefault_stream=stream.cuda_stream!=torch.cuda.default_stream().cuda_stream,
                                 initial=initial,changed_input_graph_replay=replay,output_changed=changed))
                del graph
                if a.trace and [m,n,k]==primary:
                    if not a.trace.resolve().is_relative_to(root) or a.trace.exists(): raise ValueError('New task-local trace required')
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=True,with_stack=True) as prof:
                        with torch.profiler.record_function('ThorKDA/GEMM/smoke'):
                            call();torch.cuda.synchronize()
                    a.trace.parent.mkdir(parents=True,exist_ok=True);prof.export_chrome_trace(str(a.trace))
            finally: lib.ct_destroy(plan)
    result=dict(scope='Kernel ABI verification, not a new optimization or full-model validation',candidate=manifest['name'],
                binary_sha256=hashlib.sha256(so.read_bytes()).hexdigest(),device=torch.cuda.get_device_name(),rows=rows,
                passed=all(r['initial']['passed'] and r['changed_input_graph_replay']['passed'] and r['output_changed'] and r['nondefault_stream'] for r in rows) if rows else None)
    if a.output:
        if not a.output.resolve().is_relative_to(root): raise ValueError('Output must be task-local')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    return 0 if a.capture_only or result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
