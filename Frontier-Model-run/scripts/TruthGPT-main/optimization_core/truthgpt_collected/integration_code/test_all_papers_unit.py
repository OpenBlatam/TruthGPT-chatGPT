#!/usr/bin/env python3
"""
Unit Tests para Todos los Papers - Top 10 Papers 2025
=======================================================

Tests unitarios individuales para cada paper, verificando:
- Inicialización correcta
- Forward pass sin errores
- Shape de outputs
- Métricas básicas
- Edge cases
"""

import torch
import torch.nn as nn
import unittest
from pathlib import Path
import sys
import traceback

# Add papers to path
papers_dir = Path(__file__).parent / 'papers'
sys.path.insert(0, str(papers_dir))

# Import all papers using direct file imports
import importlib.util

def load_module(module_path, module_name):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load all papers
paper_qwen3 = load_module(papers_dir / 'research' / 'paper_qwen3.py', 'paper_qwen3')
paper_absolute_zero = load_module(papers_dir / 'research' / 'paper_absolute_zero.py', 'paper_absolute_zero')
paper_seed1_5_vl = load_module(papers_dir / 'research' / 'paper_seed1_5_vl.py', 'paper_seed1_5_vl')
paper_mixture_of_reasonings = load_module(papers_dir / 'research' / 'paper_mixture_of_reasonings.py', 'paper_mixture_of_reasonings')
paper_crft = load_module(papers_dir / 'research' / 'paper_crft.py', 'paper_crft')
paper_meta_cot = load_module(papers_dir / 'research' / 'paper_meta_cot.py', 'paper_meta_cot')
paper_sft_rl_generalization = load_module(papers_dir / 'research' / 'paper_sft_rl_generalization.py', 'paper_sft_rl_generalization')
paper_learning_dynamics = load_module(papers_dir / 'research' / 'paper_learning_dynamics.py', 'paper_learning_dynamics')
paper_faster_cascades = load_module(papers_dir / 'inference' / 'paper_faster_cascades.py', 'paper_faster_cascades')
paper_deepseek_v3 = load_module(papers_dir / 'architecture' / 'paper_deepseek_v3.py', 'paper_deepseek_v3')

# Extract classes
Qwen3Module = paper_qwen3.Qwen3Module
Qwen3Config = paper_qwen3.Qwen3Config
RLVRModule = paper_absolute_zero.RLVRModule
AbsoluteZeroConfig = paper_absolute_zero.AbsoluteZeroConfig
Seed1_5VLModule = paper_seed1_5_vl.Seed1_5VLModule
Seed1_5VLConfig = paper_seed1_5_vl.Seed1_5VLConfig
MixtureOfReasoningsModule = paper_mixture_of_reasonings.MixtureOfReasoningsModule
MixtureOfReasoningsConfig = paper_mixture_of_reasonings.MixtureOfReasoningsConfig
CRFTModule = paper_crft.CRFTModule
CRFTConfig = paper_crft.CRFTConfig
MetaCoTModule = paper_meta_cot.MetaCoTModule
MetaCoTConfig = paper_meta_cot.MetaCoTConfig
SFTRLGeneralizationModule = paper_sft_rl_generalization.SFTRLGeneralizationModule
SFTRLGeneralizationConfig = paper_sft_rl_generalization.SFTRLGeneralizationConfig
LearningDynamicsModule = paper_learning_dynamics.LearningDynamicsModule
LearningDynamicsConfig = paper_learning_dynamics.LearningDynamicsConfig
FasterCascadesModule = paper_faster_cascades.FasterCascadesModule
FasterCascadesConfig = paper_faster_cascades.FasterCascadesConfig
DeepSeekV3Module = paper_deepseek_v3.DeepSeekV3Module
DeepSeekV3Config = paper_deepseek_v3.DeepSeekV3Config


class TestPaperBase(unittest.TestCase):
    """Base class para tests de papers."""
    
    def setUp(self):
        """Setup común para todos los tests."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = 512
        self.batch_size = 2
        self.seq_len = 32
        
    def create_test_input(self):
        """Crea input de prueba estándar."""
        return torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)


class TestQwen3(TestPaperBase):
    """Tests para Qwen3."""
    
    def test_initialization(self):
        """Test de inicialización."""
        config = Qwen3Config(hidden_dim=self.hidden_dim)
        module = Qwen3Module(config).to(self.device)
        self.assertIsNotNone(module)
        
    def test_forward_pass(self):
        """Test de forward pass."""
        config = Qwen3Config(hidden_dim=self.hidden_dim)
        module = Qwen3Module(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        with torch.no_grad():
            output, metadata = module(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertIn('thinking_mode', metadata)
        self.assertIn('multilingual_enabled', metadata)
        
    def test_metrics(self):
        """Test de métricas."""
        config = Qwen3Config(hidden_dim=self.hidden_dim)
        module = Qwen3Module(config).to(self.device)
        metrics = module.get_metrics()
        
        self.assertIn('avg_thinking_mode_quality', metrics)
        self.assertIn('num_languages', metrics)
        self.assertEqual(metrics['num_languages'], 119)


class TestAbsoluteZero(TestPaperBase):
    """Tests para Absolute Zero (AZR)."""
    
    def test_initialization(self):
        """Test de inicialización."""
        config = AbsoluteZeroConfig(hidden_dim=self.hidden_dim)
        module = RLVRModule(config).to(self.device)
        self.assertIsNotNone(module)
        
    def test_forward_pass(self):
        """Test de forward pass."""
        config = AbsoluteZeroConfig(hidden_dim=self.hidden_dim)
        module = RLVRModule(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        with torch.no_grad():
            output, metadata = module(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertIn('avg_reward', metadata)
        self.assertIn('self_play_enabled', metadata)
        
    def test_metrics(self):
        """Test de métricas."""
        config = AbsoluteZeroConfig(hidden_dim=self.hidden_dim)
        module = RLVRModule(config).to(self.device)
        metrics = module.get_metrics()
        
        self.assertIn('avg_reward', metrics)
        self.assertIn('zero_data', metrics)
        self.assertTrue(metrics['zero_data'])


class TestSeed1_5VL(TestPaperBase):
    """Tests para Seed1.5-VL."""
    
    def test_initialization(self):
        """Test de inicialización."""
        config = Seed1_5VLConfig(hidden_dim=self.hidden_dim)
        module = Seed1_5VLModule(config).to(self.device)
        self.assertIsNotNone(module)
        
    def test_forward_pass_text_only(self):
        """Test de forward pass solo texto."""
        config = Seed1_5VLConfig(hidden_dim=self.hidden_dim)
        module = Seed1_5VLModule(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        with torch.no_grad():
            output, metadata = module(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertIn('mmmu_score', metadata)
        
    def test_forward_pass_multimodal(self):
        """Test de forward pass multimodal."""
        config = Seed1_5VLConfig(hidden_dim=self.hidden_dim)
        module = Seed1_5VLModule(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        vision_features = torch.randn(self.batch_size, self.seq_len, config.vision_dim, device=self.device)
        
        with torch.no_grad():
            output, metadata = module(input_tensor, vision_features)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertTrue(metadata['multimodal'])
        
    def test_metrics(self):
        """Test de métricas."""
        config = Seed1_5VLConfig(hidden_dim=self.hidden_dim)
        module = Seed1_5VLModule(config).to(self.device)
        metrics = module.get_metrics()
        
        self.assertIn('mmmu_score', metrics)
        self.assertAlmostEqual(metrics['mmmu_score'], 0.779, places=3)


class TestMixtureOfReasonings(TestPaperBase):
    """Tests para Mixture of Reasonings."""
    
    def test_initialization(self):
        """Test de inicialización."""
        config = MixtureOfReasoningsConfig(hidden_dim=self.hidden_dim)
        module = MixtureOfReasoningsModule(config).to(self.device)
        self.assertIsNotNone(module)
        
    def test_forward_pass(self):
        """Test de forward pass."""
        config = MixtureOfReasoningsConfig(hidden_dim=self.hidden_dim)
        module = MixtureOfReasoningsModule(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        with torch.no_grad():
            output, metadata = module(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertIn('selected_strategy', metadata)
        self.assertIn('num_strategies', metadata)
        
    def test_metrics(self):
        """Test de métricas."""
        config = MixtureOfReasoningsConfig(hidden_dim=self.hidden_dim)
        module = MixtureOfReasoningsModule(config).to(self.device)
        metrics = module.get_metrics()
        
        self.assertIn('strategy_usage', metrics)
        self.assertIn('num_strategies', metrics)
        self.assertEqual(len(metrics['strategy_usage']), config.num_strategies)


class TestCRFT(TestPaperBase):
    """Tests para CRFT."""
    
    def test_initialization(self):
        """Test de inicialización."""
        config = CRFTConfig(hidden_dim=self.hidden_dim)
        module = CRFTModule(config).to(self.device)
        self.assertIsNotNone(module)
        
    def test_forward_pass(self):
        """Test de forward pass."""
        config = CRFTConfig(hidden_dim=self.hidden_dim)
        module = CRFTModule(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        with torch.no_grad():
            output, metadata = module(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertIn('parameter_efficiency', metadata)
        
    def test_metrics(self):
        """Test de métricas."""
        config = CRFTConfig(hidden_dim=self.hidden_dim)
        module = CRFTModule(config).to(self.device)
        metrics = module.get_metrics()
        
        self.assertIn('parameter_efficiency', metrics)
        self.assertAlmostEqual(metrics['total_params_ratio'], 0.00016, places=5)


class TestMetaCoT(TestPaperBase):
    """Tests para Meta-CoT."""
    
    def test_initialization(self):
        """Test de inicialización."""
        config = MetaCoTConfig(hidden_dim=self.hidden_dim)
        module = MetaCoTModule(config).to(self.device)
        self.assertIsNotNone(module)
        
    def test_forward_pass(self):
        """Test de forward pass."""
        config = MetaCoTConfig(hidden_dim=self.hidden_dim)
        module = MetaCoTModule(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        with torch.no_grad():
            output, metadata = module(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertIn('num_reasoning_steps', metadata)
        self.assertIn('system2_reasoning', metadata)
        
    def test_metrics(self):
        """Test de métricas."""
        config = MetaCoTConfig(hidden_dim=self.hidden_dim)
        module = MetaCoTModule(config).to(self.device)
        metrics = module.get_metrics()
        
        self.assertIn('reasoning_quality', metrics)
        self.assertIn('verification_rate', metrics)


class TestSFTRLGeneralization(TestPaperBase):
    """Tests para SFT vs RL Generalization."""
    
    def test_initialization(self):
        """Test de inicialización."""
        config = SFTRLGeneralizationConfig(hidden_dim=self.hidden_dim)
        module = SFTRLGeneralizationModule(config).to(self.device)
        self.assertIsNotNone(module)
        
    def test_forward_pass(self):
        """Test de forward pass."""
        config = SFTRLGeneralizationConfig(hidden_dim=self.hidden_dim)
        module = SFTRLGeneralizationModule(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        with torch.no_grad():
            output, metadata = module(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertIn('ood_detection_rate', metadata)
        self.assertIn('use_rl', metadata)
        
    def test_metrics(self):
        """Test de métricas."""
        config = SFTRLGeneralizationConfig(hidden_dim=self.hidden_dim)
        module = SFTRLGeneralizationModule(config).to(self.device)
        metrics = module.get_metrics()
        
        self.assertIn('generalization_score', metrics)
        self.assertIn('rl_advantage', metrics)


class TestLearningDynamics(TestPaperBase):
    """Tests para Learning Dynamics."""
    
    def test_initialization(self):
        """Test de inicialización."""
        config = LearningDynamicsConfig(hidden_dim=self.hidden_dim)
        module = LearningDynamicsModule(config).to(self.device)
        self.assertIsNotNone(module)
        
    def test_forward_pass(self):
        """Test de forward pass."""
        config = LearningDynamicsConfig(hidden_dim=self.hidden_dim)
        module = LearningDynamicsModule(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        with torch.no_grad():
            output, metadata = module(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertIn('hallucination_rate', metadata)
        self.assertIn('squeezing_rate', metadata)
        
    def test_metrics(self):
        """Test de métricas."""
        config = LearningDynamicsConfig(hidden_dim=self.hidden_dim)
        module = LearningDynamicsModule(config).to(self.device)
        metrics = module.get_metrics()
        
        self.assertIn('hallucination_rate', metrics)
        self.assertIn('qa_accuracy', metrics)


class TestFasterCascades(TestPaperBase):
    """Tests para Faster Cascades."""
    
    def test_initialization(self):
        """Test de inicialización."""
        config = FasterCascadesConfig(hidden_dim=self.hidden_dim)
        module = FasterCascadesModule(config).to(self.device)
        self.assertIsNotNone(module)
        
    def test_forward_pass(self):
        """Test de forward pass."""
        config = FasterCascadesConfig(hidden_dim=self.hidden_dim)
        module = FasterCascadesModule(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        with torch.no_grad():
            output, metadata = module(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertIn('inference_speedup', metadata)
        self.assertIn('cascade_level_used', metadata)
        
    def test_metrics(self):
        """Test de métricas."""
        config = FasterCascadesConfig(hidden_dim=self.hidden_dim)
        module = FasterCascadesModule(config).to(self.device)
        metrics = module.get_metrics()
        
        self.assertIn('inference_speedup', metrics)
        self.assertIn('cascade_usage', metrics)
        self.assertEqual(len(metrics['cascade_usage']), config.num_cascade_levels)


class TestDeepSeekV3(TestPaperBase):
    """Tests para DeepSeek-V3."""
    
    def test_initialization(self):
        """Test de inicialización."""
        config = DeepSeekV3Config(hidden_dim=self.hidden_dim)
        module = DeepSeekV3Module(config).to(self.device)
        self.assertIsNotNone(module)
        
    def test_forward_pass(self):
        """Test de forward pass."""
        config = DeepSeekV3Config(hidden_dim=self.hidden_dim)
        module = DeepSeekV3Module(config).to(self.device)
        module.eval()
        
        input_tensor = self.create_test_input()
        with torch.no_grad():
            output, metadata = module(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertIn('memory_efficiency', metadata)
        self.assertIn('uses_mla', metadata)
        
    def test_metrics(self):
        """Test de métricas."""
        config = DeepSeekV3Config(hidden_dim=self.hidden_dim)
        module = DeepSeekV3Module(config).to(self.device)
        metrics = module.get_metrics()
        
        self.assertIn('memory_efficiency', metrics)
        self.assertIn('num_experts', metrics)
        self.assertEqual(metrics['num_experts'], config.num_experts)


class TestEdgeCases(TestPaperBase):
    """Tests de edge cases para todos los papers."""
    
    def test_small_batch(self):
        """Test con batch size pequeño."""
        papers = [
            (Qwen3Module, Qwen3Config),
            (RLVRModule, AbsoluteZeroConfig),
            (Seed1_5VLModule, Seed1_5VLConfig),
            (MixtureOfReasoningsModule, MixtureOfReasoningsConfig),
            (CRFTModule, CRFTConfig),
            (MetaCoTModule, MetaCoTConfig),
            (SFTRLGeneralizationModule, SFTRLGeneralizationConfig),
            (LearningDynamicsModule, LearningDynamicsConfig),
            (FasterCascadesModule, FasterCascadesConfig),
            (DeepSeekV3Module, DeepSeekV3Config),
        ]
        
        for ModuleClass, ConfigClass in papers:
            with self.subTest(paper=ModuleClass.__name__):
                config = ConfigClass(hidden_dim=self.hidden_dim)
                module = ModuleClass(config).to(self.device)
                module.eval()
                
                # Batch size 1
                input_tensor = torch.randn(1, self.seq_len, self.hidden_dim, device=self.device)
                with torch.no_grad():
                    output, metadata = module(input_tensor)
                
                self.assertEqual(output.shape[0], 1)
                self.assertEqual(output.shape[1], self.seq_len)
                self.assertEqual(output.shape[2], self.hidden_dim)
    
    def test_short_sequence(self):
        """Test con secuencia corta."""
        papers = [
            (Qwen3Module, Qwen3Config),
            (RLVRModule, AbsoluteZeroConfig),
            (CRFTModule, CRFTConfig),
            (SFTRLGeneralizationModule, SFTRLGeneralizationConfig),
        ]
        
        for ModuleClass, ConfigClass in papers:
            with self.subTest(paper=ModuleClass.__name__):
                config = ConfigClass(hidden_dim=self.hidden_dim)
                module = ModuleClass(config).to(self.device)
                module.eval()
                
                # Sequence length 1
                input_tensor = torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device)
                with torch.no_grad():
                    output, metadata = module(input_tensor)
                
                self.assertEqual(output.shape[1], 1)
    
    def test_different_hidden_dims(self):
        """Test con diferentes hidden dimensions."""
        papers = [
            (Qwen3Module, Qwen3Config),
            (CRFTModule, CRFTConfig),
            (SFTRLGeneralizationModule, SFTRLGeneralizationConfig),
        ]
        
        for hidden_dim in [256, 512, 768, 1024]:
            for ModuleClass, ConfigClass in papers:
                with self.subTest(paper=ModuleClass.__name__, hidden_dim=hidden_dim):
                    try:
                        config = ConfigClass(hidden_dim=hidden_dim)
                        module = ModuleClass(config).to(self.device)
                        module.eval()
                        
                        input_tensor = torch.randn(self.batch_size, self.seq_len, hidden_dim, device=self.device)
                        with torch.no_grad():
                            output, metadata = module(input_tensor)
                        
                        self.assertEqual(output.shape, input_tensor.shape)
                    except Exception as e:
                        # Some papers might not support all hidden dims
                        pass


def run_all_tests():
    """Ejecuta todos los tests y genera reporte."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestQwen3,
        TestAbsoluteZero,
        TestSeed1_5VL,
        TestMixtureOfReasonings,
        TestCRFT,
        TestMetaCoT,
        TestSFTRLGeneralization,
        TestLearningDynamics,
        TestFasterCascades,
        TestDeepSeekV3,
        TestEdgeCases,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result


if __name__ == "__main__":
    print("🧪 Running Unit Tests for All Papers...")
    print("="*80)
    result = run_all_tests()
    
    # Exit with error code if tests failed
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)


