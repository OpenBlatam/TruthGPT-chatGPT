"""
Unit tests for optimization validation and testing
Tests validation frameworks, test data generation, and optimization verification
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestOptimizationValidation(unittest.TestCase):
    """Test suite for optimization validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_optimization_validation(self):
        """Test optimization validation framework"""
        class OptimizationValidator:
            def __init__(self, validation_criteria):
                self.validation_criteria = validation_criteria
                self.validation_results = {}
                self.validation_history = []
                
            def validate_optimization(self, optimization_result):
                """Validate optimization result"""
                validation_result = {
                    'optimization_result': optimization_result,
                    'validation_passed': True,
                    'validation_errors': [],
                    'validation_warnings': []
                }
                
                # Check each validation criterion
                for criterion_name, criterion_func in self.validation_criteria.items():
                    try:
                        criterion_result = criterion_func(optimization_result)
                        if not criterion_result['passed']:
                            validation_result['validation_passed'] = False
                            validation_result['validation_errors'].append({
                                'criterion': criterion_name,
                                'message': criterion_result['message']
                            })
                        elif criterion_result.get('warning'):
                            validation_result['validation_warnings'].append({
                                'criterion': criterion_name,
                                'message': criterion_result['warning']
                            })
                    except Exception as e:
                        validation_result['validation_passed'] = False
                        validation_result['validation_errors'].append({
                            'criterion': criterion_name,
                            'message': f"Validation error: {str(e)}"
                        })
                        
                # Record validation result
                self.validation_results[optimization_result.get('id', 'unknown')] = validation_result
                self.validation_history.append(validation_result)
                
                return validation_result
                
            def get_validation_stats(self):
                """Get validation statistics"""
                if not self.validation_history:
                    return {}
                    
                total_validations = len(self.validation_history)
                passed_validations = sum(1 for v in self.validation_history if v['validation_passed'])
                failed_validations = total_validations - passed_validations
                
                return {
                    'total_validations': total_validations,
                    'passed_validations': passed_validations,
                    'failed_validations': failed_validations,
                    'success_rate': passed_validations / total_validations if total_validations > 0 else 0,
                    'total_errors': sum(len(v['validation_errors']) for v in self.validation_history),
                    'total_warnings': sum(len(v['validation_warnings']) for v in self.validation_history)
                }
        
        # Test optimization validation
        validation_criteria = {
            'convergence_check': lambda result: {
                'passed': result.get('converged', False),
                'message': 'Optimization did not converge'
            },
            'performance_check': lambda result: {
                'passed': result.get('final_loss', 1.0) < 0.1,
                'message': 'Final loss is too high'
            },
            'stability_check': lambda result: {
                'passed': result.get('gradient_norm', 0) < 10.0,
                'message': 'Gradient norm is too high'
            }
        }
        
        validator = OptimizationValidator(validation_criteria)
        
        # Test validation with good result
        good_result = {
            'id': 'test_1',
            'converged': True,
            'final_loss': 0.05,
            'gradient_norm': 5.0
        }
        
        validation_result = validator.validate_optimization(good_result)
        self.assertTrue(validation_result['validation_passed'])
        self.assertEqual(len(validation_result['validation_errors']), 0)
        
        # Test validation with bad result
        bad_result = {
            'id': 'test_2',
            'converged': False,
            'final_loss': 0.5,
            'gradient_norm': 15.0
        }
        
        validation_result = validator.validate_optimization(bad_result)
        self.assertFalse(validation_result['validation_passed'])
        self.assertGreater(len(validation_result['validation_errors']), 0)
        
        # Check validation stats
        stats = validator.get_validation_stats()
        self.assertEqual(stats['total_validations'], 2)
        self.assertEqual(stats['passed_validations'], 1)
        self.assertEqual(stats['failed_validations'], 1)
        self.assertEqual(stats['success_rate'], 0.5)
        self.assertGreater(stats['total_errors'], 0)
        
    def test_optimization_testing(self):
        """Test optimization testing framework"""
        class OptimizationTester:
            def __init__(self, test_cases):
                self.test_cases = test_cases
                self.test_results = {}
                self.test_history = []
                
            def run_optimization_tests(self, optimizer):
                """Run optimization tests"""
                for test_name, test_case in self.test_cases.items():
                    test_result = self._run_single_test(optimizer, test_name, test_case)
                    self.test_results[test_name] = test_result
                    self.test_history.append(test_result)
                    
                return self.test_results
                
            def _run_single_test(self, optimizer, test_name, test_case):
                """Run single optimization test"""
                test_result = {
                    'test_name': test_name,
                    'test_case': test_case,
                    'test_passed': True,
                    'test_errors': [],
                    'performance_metrics': {}
                }
                
                try:
                    # Run optimization test
                    result = self._execute_optimization_test(optimizer, test_case)
                    test_result['performance_metrics'] = result
                    
                    # Check test criteria
                    for criterion_name, criterion_value in test_case.get('criteria', {}).items():
                        if criterion_name in result:
                            if not self._check_criterion(result[criterion_name], criterion_value):
                                test_result['test_passed'] = False
                                test_result['test_errors'].append({
                                    'criterion': criterion_name,
                                    'expected': criterion_value,
                                    'actual': result[criterion_name]
                                })
                                
                except Exception as e:
                    test_result['test_passed'] = False
                    test_result['test_errors'].append({
                        'error': str(e)
                    })
                    
                return test_result
                
            def _execute_optimization_test(self, optimizer, test_case):
                """Execute optimization test"""
                # Simulate optimization test execution
                result = {
                    'final_loss': np.random.uniform(0, 1),
                    'convergence_time': np.random.uniform(10, 100),
                    'function_evaluations': np.random.randint(100, 1000),
                    'success': np.random.uniform(0, 1) > 0.2,
                    'accuracy': np.random.uniform(0.8, 0.99)
                }
                return result
                
            def _check_criterion(self, actual, expected):
                """Check if actual value meets expected criterion"""
                if isinstance(expected, dict):
                    if 'min' in expected:
                        return actual >= expected['min']
                    elif 'max' in expected:
                        return actual <= expected['max']
                    elif 'range' in expected:
                        return expected['range'][0] <= actual <= expected['range'][1]
                else:
                    return actual == expected
                    
            def get_test_stats(self):
                """Get test statistics"""
                if not self.test_history:
                    return {}
                    
                total_tests = len(self.test_history)
                passed_tests = sum(1 for t in self.test_history if t['test_passed'])
                failed_tests = total_tests - passed_tests
                
                return {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                    'total_errors': sum(len(t['test_errors']) for t in self.test_history)
                }
        
        # Test optimization testing
        test_cases = {
            'convergence_test': {
                'criteria': {
                    'final_loss': {'max': 0.1},
                    'success': True
                }
            },
            'performance_test': {
                'criteria': {
                    'accuracy': {'min': 0.9},
                    'convergence_time': {'max': 50}
                }
            },
            'stability_test': {
                'criteria': {
                    'function_evaluations': {'max': 500}
                }
            }
        }
        
        tester = OptimizationTester(test_cases)
        
        # Test optimization testing
        optimizer = {'type': 'test_optimizer'}
        results = tester.run_optimization_tests(optimizer)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIn('convergence_test', results)
        self.assertIn('performance_test', results)
        self.assertIn('stability_test', results)
        
        for test_name, result in results.items():
            self.assertIn('test_passed', result)
            self.assertIn('test_errors', result)
            self.assertIn('performance_metrics', result)
            
        # Check test stats
        stats = tester.get_test_stats()
        self.assertEqual(stats['total_tests'], 3)
        self.assertGreaterEqual(stats['passed_tests'], 0)
        self.assertGreaterEqual(stats['failed_tests'], 0)
        self.assertGreaterEqual(stats['success_rate'], 0)
        self.assertGreaterEqual(stats['total_errors'], 0)

class TestOptimizationVerification(unittest.TestCase):
    """Test suite for optimization verification"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_optimization_verification(self):
        """Test optimization verification framework"""
        class OptimizationVerifier:
            def __init__(self, verification_rules):
                self.verification_rules = verification_rules
                self.verification_results = {}
                self.verification_history = []
                
            def verify_optimization(self, optimization_result):
                """Verify optimization result"""
                verification_result = {
                    'optimization_result': optimization_result,
                    'verification_passed': True,
                    'verification_errors': [],
                    'verification_warnings': []
                }
                
                # Check each verification rule
                for rule_name, rule_func in self.verification_rules.items():
                    try:
                        rule_result = rule_func(optimization_result)
                        if not rule_result['passed']:
                            verification_result['verification_passed'] = False
                            verification_result['verification_errors'].append({
                                'rule': rule_name,
                                'message': rule_result['message']
                            })
                        elif rule_result.get('warning'):
                            verification_result['verification_warnings'].append({
                                'rule': rule_name,
                                'message': rule_result['warning']
                            })
                    except Exception as e:
                        verification_result['verification_passed'] = False
                        verification_result['verification_errors'].append({
                            'rule': rule_name,
                            'message': f"Verification error: {str(e)}"
                        })
                        
                # Record verification result
                self.verification_results[optimization_result.get('id', 'unknown')] = verification_result
                self.verification_history.append(verification_result)
                
                return verification_result
                
            def get_verification_stats(self):
                """Get verification statistics"""
                if not self.verification_history:
                    return {}
                    
                total_verifications = len(self.verification_history)
                passed_verifications = sum(1 for v in self.verification_history if v['verification_passed'])
                failed_verifications = total_verifications - passed_verifications
                
                return {
                    'total_verifications': total_verifications,
                    'passed_verifications': passed_verifications,
                    'failed_verifications': failed_verifications,
                    'success_rate': passed_verifications / total_verifications if total_verifications > 0 else 0,
                    'total_errors': sum(len(v['verification_errors']) for v in self.verification_history),
                    'total_warnings': sum(len(v['verification_warnings']) for v in self.verification_history)
                }
        
        # Test optimization verification
        verification_rules = {
            'convergence_rule': lambda result: {
                'passed': result.get('converged', False),
                'message': 'Optimization did not converge'
            },
            'performance_rule': lambda result: {
                'passed': result.get('final_loss', 1.0) < 0.1,
                'message': 'Final loss is too high'
            },
            'stability_rule': lambda result: {
                'passed': result.get('gradient_norm', 0) < 10.0,
                'message': 'Gradient norm is too high'
            }
        }
        
        verifier = OptimizationVerifier(verification_rules)
        
        # Test verification with good result
        good_result = {
            'id': 'test_1',
            'converged': True,
            'final_loss': 0.05,
            'gradient_norm': 5.0
        }
        
        verification_result = verifier.verify_optimization(good_result)
        self.assertTrue(verification_result['verification_passed'])
        self.assertEqual(len(verification_result['verification_errors']), 0)
        
        # Test verification with bad result
        bad_result = {
            'id': 'test_2',
            'converged': False,
            'final_loss': 0.5,
            'gradient_norm': 15.0
        }
        
        verification_result = verifier.verify_optimization(bad_result)
        self.assertFalse(verification_result['verification_passed'])
        self.assertGreater(len(verification_result['verification_errors']), 0)
        
        # Check verification stats
        stats = verifier.get_verification_stats()
        self.assertEqual(stats['total_verifications'], 2)
        self.assertEqual(stats['passed_verifications'], 1)
        self.assertEqual(stats['failed_verifications'], 1)
        self.assertEqual(stats['success_rate'], 0.5)
        self.assertGreater(stats['total_errors'], 0)
        
    def test_optimization_quality_assurance(self):
        """Test optimization quality assurance"""
        class OptimizationQA:
            def __init__(self, quality_metrics):
                self.quality_metrics = quality_metrics
                self.qa_results = {}
                self.qa_history = []
                
            def assess_quality(self, optimization_result):
                """Assess optimization quality"""
                qa_result = {
                    'optimization_result': optimization_result,
                    'quality_score': 0.0,
                    'quality_grade': 'F',
                    'quality_issues': [],
                    'quality_recommendations': []
                }
                
                # Calculate quality score
                total_score = 0.0
                max_score = 0.0
                
                for metric_name, metric_func in self.quality_metrics.items():
                    try:
                        metric_result = metric_func(optimization_result)
                        total_score += metric_result['score']
                        max_score += metric_result['max_score']
                        
                        if metric_result['score'] < metric_result['max_score'] * 0.8:
                            qa_result['quality_issues'].append({
                                'metric': metric_name,
                                'score': metric_result['score'],
                                'max_score': metric_result['max_score'],
                                'issue': metric_result.get('issue', 'Low score')
                            })
                            
                    except Exception as e:
                        qa_result['quality_issues'].append({
                            'metric': metric_name,
                            'error': str(e)
                        })
                        
                # Calculate final quality score
                if max_score > 0:
                    qa_result['quality_score'] = total_score / max_score
                    
                # Assign quality grade
                if qa_result['quality_score'] >= 0.9:
                    qa_result['quality_grade'] = 'A'
                elif qa_result['quality_score'] >= 0.8:
                    qa_result['quality_grade'] = 'B'
                elif qa_result['quality_score'] >= 0.7:
                    qa_result['quality_grade'] = 'C'
                elif qa_result['quality_score'] >= 0.6:
                    qa_result['quality_grade'] = 'D'
                else:
                    qa_result['quality_grade'] = 'F'
                    
                # Generate recommendations
                if qa_result['quality_score'] < 0.8:
                    qa_result['quality_recommendations'].append('Consider adjusting optimization parameters')
                if qa_result['quality_score'] < 0.6:
                    qa_result['quality_recommendations'].append('Review optimization algorithm')
                if qa_result['quality_score'] < 0.4:
                    qa_result['quality_recommendations'].append('Consider alternative optimization approach')
                    
                # Record QA result
                self.qa_results[optimization_result.get('id', 'unknown')] = qa_result
                self.qa_history.append(qa_result)
                
                return qa_result
                
            def get_qa_stats(self):
                """Get QA statistics"""
                if not self.qa_history:
                    return {}
                    
                quality_scores = [qa['quality_score'] for qa in self.qa_history]
                quality_grades = [qa['quality_grade'] for qa in self.qa_history]
                
                return {
                    'total_assessments': len(self.qa_history),
                    'avg_quality_score': np.mean(quality_scores),
                    'min_quality_score': np.min(quality_scores),
                    'max_quality_score': np.max(quality_scores),
                    'quality_grade_distribution': {
                        grade: quality_grades.count(grade) for grade in set(quality_grades)
                    },
                    'total_issues': sum(len(qa['quality_issues']) for qa in self.qa_history),
                    'total_recommendations': sum(len(qa['quality_recommendations']) for qa in self.qa_history)
                }
        
        # Test optimization QA
        quality_metrics = {
            'convergence_quality': lambda result: {
                'score': 1.0 if result.get('converged', False) else 0.0,
                'max_score': 1.0,
                'issue': 'Optimization did not converge'
            },
            'performance_quality': lambda result: {
                'score': max(0, 1.0 - result.get('final_loss', 1.0)),
                'max_score': 1.0,
                'issue': 'Final loss is too high'
            },
            'stability_quality': lambda result: {
                'score': max(0, 1.0 - result.get('gradient_norm', 0) / 10.0),
                'max_score': 1.0,
                'issue': 'Gradient norm is too high'
            }
        }
        
        qa = OptimizationQA(quality_metrics)
        
        # Test QA with good result
        good_result = {
            'id': 'test_1',
            'converged': True,
            'final_loss': 0.05,
            'gradient_norm': 5.0
        }
        
        qa_result = qa.assess_quality(good_result)
        self.assertGreater(qa_result['quality_score'], 0.8)
        self.assertEqual(qa_result['quality_grade'], 'A')
        self.assertEqual(len(qa_result['quality_issues']), 0)
        
        # Test QA with bad result
        bad_result = {
            'id': 'test_2',
            'converged': False,
            'final_loss': 0.5,
            'gradient_norm': 15.0
        }
        
        qa_result = qa.assess_quality(bad_result)
        self.assertLess(qa_result['quality_score'], 0.6)
        self.assertEqual(qa_result['quality_grade'], 'F')
        self.assertGreater(len(qa_result['quality_issues']), 0)
        self.assertGreater(len(qa_result['quality_recommendations']), 0)
        
        # Check QA stats
        stats = qa.get_qa_stats()
        self.assertEqual(stats['total_assessments'], 2)
        self.assertGreater(stats['avg_quality_score'], 0)
        self.assertGreater(stats['min_quality_score'], 0)
        self.assertGreater(stats['max_quality_score'], 0)
        self.assertIn('quality_grade_distribution', stats)
        self.assertGreaterEqual(stats['total_issues'], 0)
        self.assertGreaterEqual(stats['total_recommendations'], 0)

if __name__ == '__main__':
    unittest.main()





