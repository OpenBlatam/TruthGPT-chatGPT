#!/usr/bin/env python3
"""
Test Suite for Library Recommender
Comprehensive tests for the library recommendation system
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import os
import json
import random

# Import library recommender components
import sys
sys.path.append('..')
from modules.base.core_system.core.library_recommender import LibraryRecommender

class TestLibraryRecommenderComprehensive(unittest.TestCase):
    """Comprehensive tests for library recommender."""
    
    def setUp(self):
        self.recommender = LibraryRecommender()
    
    def test_library_recommender_initialization(self):
        """Test library recommender initialization."""
        self.assertIsNotNone(self.recommender)
        self.assertIsInstance(self.recommender.logger, logging.Logger)
    
    def test_recommend_libraries_basic(self):
        """Test basic library recommendation."""
        requirements = {
            'task': 'optimization',
            'framework': 'pytorch',
            'performance': 'high'
        }
        
        recommendations = self.recommender.recommend_libraries(requirements)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        for rec in recommendations:
            self.assertIn('library', rec)
            self.assertIn('score', rec)
            self.assertIn('reason', rec)
    
    def test_recommend_libraries_advanced(self):
        """Test advanced library recommendation."""
        requirements = {
            'task': 'neural_architecture_search',
            'framework': 'pytorch',
            'performance': 'ultra_high',
            'memory_efficiency': True,
            'gpu_acceleration': True,
            'distributed_training': True
        }
        
        recommendations = self.recommender.recommend_libraries(requirements)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        for rec in recommendations:
            self.assertIn('library', rec)
            self.assertIn('score', rec)
            self.assertIn('reason', rec)
            self.assertIn('features', rec)
            self.assertIn('compatibility', rec)
    
    def test_recommend_libraries_with_constraints(self):
        """Test library recommendation with constraints."""
        requirements = {
            'task': 'optimization',
            'framework': 'pytorch',
            'performance': 'high'
        }
        
        constraints = {
            'max_dependencies': 10,
            'min_version': '1.8.0',
            'license': 'MIT',
            'maintenance': 'active'
        }
        
        recommendations = self.recommender.recommend_libraries(
            requirements, constraints=constraints
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        for rec in recommendations:
            self.assertIn('library', rec)
            self.assertIn('score', rec)
            self.assertIn('reason', rec)
            self.assertIn('constraints_met', rec)
    
    def test_recommend_libraries_performance(self):
        """Test library recommendation performance."""
        requirements = {
            'task': 'optimization',
            'framework': 'pytorch',
            'performance': 'high'
        }
        
        start_time = time.time()
        recommendations = self.recommender.recommend_libraries(requirements)
        end_time = time.time()
        
        recommendation_time = end_time - start_time
        self.assertLess(recommendation_time, 2.0)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    def test_recommend_libraries_scalability(self):
        """Test library recommendation scalability."""
        task_types = ['optimization', 'neural_architecture_search', 'hyperparameter_tuning', 'quantum_optimization']
        
        for task in task_types:
            requirements = {
                'task': task,
                'framework': 'pytorch',
                'performance': 'high'
            }
            
            start_time = time.time()
            recommendations = self.recommender.recommend_libraries(requirements)
            end_time = time.time()
            
            recommendation_time = end_time - start_time
            self.assertLess(recommendation_time, 3.0)
            self.assertIsInstance(recommendations, list)
            self.assertGreater(len(recommendations), 0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()

