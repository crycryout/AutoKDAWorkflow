import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from nsys_to_chrome import convert,sha256,union_ns
from thor_metrics import compute,compare,capacity
import remote_run
import new_task


class TraceTests(unittest.TestCase):
    def fixture(self,base):
        p=base/'sample.sqlite'
        with sqlite3.connect(p) as c:
            c.execute('CREATE TABLE StringIds (id INTEGER,value TEXT)')
            c.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'gemm'),(2,'cudaLaunchKernel')])
            c.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER)')
            c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)',[(1000,4000,0,7,1,11,1),(2000,5000,0,8,1,12,1),(11000,12000,0,7,1,13,1)])
            c.execute('CREATE TABLE NVTX_EVENTS (start INTEGER,end INTEGER,text TEXT,globalTid INTEGER)')
            c.executemany('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)',[(0,6000,'request:phyai:0',5),(10000,13000,'request:flashrt:0',5)])
            c.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER,end INTEGER,nameId INTEGER,globalTid INTEGER,correlationId INTEGER)')
            c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0,500,2,5,11)')
        return p

    def test_read_only_window_units_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=self.fixture(base);before=sha256(source);out=base/'view.json'
            r=convert(source,out,window_regex='request:phyai',request_regex='request')
            self.assertEqual(sha256(source),before)
            self.assertEqual(r['kernel_count'],2)
            row=r['request_windows'][0]
            self.assertEqual(row['kernel_sum_ms'],.006)
            self.assertEqual(row['gpu_busy_union_ms'],.004)
            self.assertEqual(row['nvtx_wall_ms'],.006)
            d=json.loads(out.read_text());kernels=[e for e in d['traceEvents'] if e['cat']=='kernel']
            self.assertEqual(kernels[0]['ts'],1.0);self.assertEqual(kernels[0]['dur'],3.0)
            self.assertEqual(kernels[0]['args']['start'],1000)
            self.assertFalse(any(e['cat']=='python_function' for e in d['traceEvents']))

    def test_cross_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=self.fixture(base)
            with sqlite3.connect(source) as c:c.execute('UPDATE NVTX_EVENTS SET end=3000 WHERE text="request:phyai:0"')
            with self.assertRaisesRegex(ValueError,'crosses'):convert(source,base/'view.json',window_regex='request:phyai')

    def test_output_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.fixture(Path(tmp))
            with self.assertRaises(ValueError):convert(p,p)

    def test_union_overlapping_and_disjoint(self):
        self.assertEqual(union_ns([(8,12),(0,5),(3,7)]),11)


class MetricsTests(unittest.TestCase):
    def test_no_invented_call_counts_or_peak(self):
        rows=[dict(id='g',shape=[2,3,4],bias=True)]
        r=compute(rows,1,10);self.assertIsNone(r['gemm_flops_per_request']);self.assertIsNone(r['gemm_mfu'])
        rows[0]['calls_per_request']=2;r=compute(rows,1)
        self.assertEqual(r['gemm_flops_per_request'],96);self.assertIsNone(r['gemm_mfu'])
        self.assertAlmostEqual(compute(rows,1,10)['gemm_mfu'],9.6e-9)

    def test_compare_requires_same_contract_and_excludes_profile(self):
        d=dict(contracts=dict(baseline='abc',candidate='xyz'),pairs=[])
        with self.assertRaises(ValueError):compare(d)
        d['contracts']['candidate']='abc'
        d['pairs']=[dict(baseline_e2e_ms=100,candidate_e2e_ms=90,valid=True,profiled=False) for _ in range(20)]
        d['pairs'].append(dict(baseline_e2e_ms=100,candidate_e2e_ms=1,valid=True,profiled=True))
        r=compare(d);self.assertEqual(r['valid_pairs'],20);self.assertEqual(r['excluded_pairs'],1)
        self.assertAlmostEqual(r['candidate_over_baseline'],.9);self.assertIn('alone does not promote',r['promotion'])

    def test_unified_memory_is_not_double_counted(self):
        r=capacity(dict(meminfo_kib=dict(MemTotal=8*1024**2,MemAvailable=3*1024**2),gpu_processes='2 GiB'))
        self.assertEqual(r['total_gib'],8);self.assertEqual(r['available_gib'],3)


if __name__=='__main__':unittest.main()
