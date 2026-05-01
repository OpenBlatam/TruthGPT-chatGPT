"""
Gaussian Process Model
======================

Gaussian Process implementation for Bayesian optimization.
"""
import logging
import numpy as np
from typing import Tuple, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from .config import BayesianOptimizationConfig
from .enums import KernelType

logger = logging.getLogger(__name__)

class GaussianProcessModel:
    """Gaussian Process model implementation"""
    
    def __init__(self, config: BayesianOptimizationConfig):
        self.config = config
        self.gp_model = None
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        logger.info("✅ Gaussian Process Model initialized")
    
    def create_kernel(self):
        """Create kernel based on configuration"""
        if self.config.kernel_type == KernelType.RBF:
            kernel = RBF(length_scale=1.0)
        elif self.config.kernel_type == KernelType.MATERN:
            kernel = Matern(length_scale=1.0, nu=2.5)
        elif self.config.kernel_type == KernelType.WHITE:
            kernel = WhiteKernel(noise_level=1.0)
        elif self.config.kernel_type == KernelType.CONSTANT:
            kernel = ConstantKernel(constant_value=1.0)
        else:
            kernel = RBF(length_scale=1.0)
        
        # Add white noise kernel
        kernel += WhiteKernel(noise_level=self.config.gp_alpha)
        
        return kernel
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Gaussian Process model"""
        logger.info("🔧 Fitting Gaussian Process model")
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Create kernel
        kernel = self.create_kernel()
        
        # Create GP model
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.gp_alpha,
            n_restarts_optimizer=self.config.gp_n_restarts,
            normalize_y=self.config.gp_normalize_y,
            random_state=42
        )
        
        # Fit model
        self.gp_model.fit(X, y)
        self.is_fitted = True
        
        logger.info("✅ Gaussian Process model fitted")
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict using Gaussian Process model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if return_std:
            mean, std = self.gp_model.predict(X, return_std=True)
            return mean, std
        else:
            mean = self.gp_model.predict(X, return_std=False)
            return mean, None
    
    def sample_y(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from Gaussian Process model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
        
        return self.gp_model.sample_y(X, n_samples=n_samples)

