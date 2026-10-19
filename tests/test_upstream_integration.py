import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]

@unittest.skipUnless((ROOT/'infra/skills/llm-torch-profiler-analysis').exists(), 'Run tools/bootstrap_corpus.py for upstream integration checks')
class ProfilerBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(ROOT/'infra/skills/llm-torch-profiler-analysis/scripts'))
        import profile_common
        import analyze_llm_torch_profile
        cls.common=profile_common;cls.cli=analyze_llm_torch_profile

    def test_backend_choices_and_source_paths(self):
        for framework in ['phyai','flashrt','thor']:
            args=self.cli.build_triage_parser().parse_args(['--framework',framework,'--input','x'])
            self.assertEqual(self.common.resolve_framework(args.framework),framework)
        self.assertEqual(self.common.normalize_repo_relative_path('/workspace/pkg/phyai/model.py'),'phyai/model.py')

    def test_offline_backends_never_issue_http_capture(self):
        for framework in ['phyai','flashrt','thor']:
            with patch.object(self.common.request,'urlopen') as request:
                with self.assertRaisesRegex(ValueError,'task-local'):
                    self.common.run_profiler('http://unused',None,1,True,False,None,0,'x',1,0,framework=framework)
                request.assert_not_called()

    def test_thor_manglings_keep_tiles_and_classify_norm(self):
        import triage_kernel_helpers as kernels
        import triage_overlap_helpers as overlap
        for module in (kernels,overlap):
            self.assertEqual(module.canonicalize_name('nvjet_sm110_tst_320x192_64x4'), 'nvjet_sm110_tst_320x192_64x4')
            self.assertEqual(module.canonicalize_name('object_at_0xdeadbeef'),'object_at_0xADDR')
        self.assertEqual(kernels.classify_kernel('kernel_cutlass_flashinfernormkernelslayernormLayerNormKernel'),'norm')
        self.assertEqual(kernels.classify_kernel('void pytorch_flash::flash_fwd_kernel<...>'),'attention')
        self.assertEqual(kernels.classify_kernel('warp_splitk(cutlass::bfloat16_t const*, int, int)'),'gemm')

@unittest.skipUnless((ROOT/'infra/skills/llm-pipeline-analysis').exists(), 'Run tools/bootstrap_corpus.py for upstream integration checks')
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(ROOT/'infra/skills/llm-pipeline-analysis/scripts'))
        import model_profiles
        import layer_timeline_analyzer
        cls.profiles=model_profiles;cls.timeline=layer_timeline_analyzer

    def test_thor_requires_evidence_and_terminal(self):
        gpu=[dict(name='anchor',ts=0),dict(name='down_gemm',ts=1),dict(name='expert',ts=2)]
        with self.assertRaisesRegex(ValueError,'terminal-index'):self.profiles.bounded_component(gpu,required=True)
        with self.assertRaisesRegex(ValueError,'evidence'):self.profiles.bounded_component(gpu,1)
        bounded=self.profiles.bounded_component(gpu,1,'verified down_proj call',True)
        self.assertEqual(len(bounded),2);self.assertEqual(len(gpu),3)
        self.assertEqual(self.profiles.checked_anchor_indices(bounded,'anchor',1,1),[0,2])

    def test_thor_categories_do_not_infer_gemm_dtype(self):
        profile=self.profiles.get_profile('thor')
        for name,category in [('cutlass::gemm::ThorCompactNativeGemmUniversal','gemm'),('nvjet_sm110','gemm'),('flashinfernormkernelslayernormLayerNormKernel','rmsnorm'),('pytorch_flash::flash_fwd_kernel','attention')]:
            self.assertEqual(self.timeline.classify_kernel(name,profile),category)
        self.assertEqual(self.timeline.prefix_total(dict(gemm=2,gemm_bf16=3,attention=8),'gemm'),5)

if __name__=='__main__':unittest.main()
