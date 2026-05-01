"""
Pretext Tasks
=============

Pretext task models and training logic.
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Any
from .config import SSLConfig
from .enums import PretextTaskType

logger = logging.getLogger(__name__)

class PretextTaskModel:
    """Pretext task model implementation"""
    
    def __init__(self, config: SSLConfig):
        self.config = config
        self.task_models = {}
        self.training_history = []
        logger.info("✅ Pretext Task Model initialized")
    
    def create_rotation_prediction_model(self) -> nn.Module:
        """Create rotation prediction model"""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 4)  # 4 rotation classes
        )
        return model
    
    def create_colorization_model(self) -> nn.Module:
        """Create colorization model"""
        model = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, 1),
            nn.Tanh()
        )
        return model
    
    def create_inpainting_model(self) -> nn.Module:
        """Create inpainting model"""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, 1),
            nn.Tanh()
        )
        return model
    
    def train_pretext_task(self, task_type: PretextTaskType, 
                          data: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, Any]:
        """Train pretext task"""
        logger.info(f"🎯 Training pretext task: {task_type.value}")
        
        if task_type == PretextTaskType.ROTATION_PREDICTION:
            return self._train_rotation_prediction(data, labels)
        elif task_type == PretextTaskType.COLORIZATION:
            return self._train_colorization(data, labels)
        elif task_type == PretextTaskType.INPAINTING:
            return self._train_inpainting(data, labels)
        else:
            return self._train_generic_pretext_task(data, labels)

    def _train_rotation_prediction(self, data: torch.Tensor, 
                                 labels: torch.Tensor) -> Dict[str, Any]:
        """Train rotation prediction task"""
        if 'rotation_prediction' not in self.task_models:
            self.task_models['rotation_prediction'] = self.create_rotation_prediction_model()
        
        model = self.task_models['rotation_prediction']
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Generate rotation labels
        rotation_labels = torch.randint(0, 4, (data.shape[0],))
        
        # Apply rotations
        rotated_data = []
        for i, rotation in enumerate(rotation_labels):
            if rotation == 0:
                rotated_data.append(data[i])
            elif rotation == 1:
                rotated_data.append(torch.rot90(data[i], 1, dims=[1, 2]))
            elif rotation == 2:
                rotated_data.append(torch.rot90(data[i], 2, dims=[1, 2]))
            else:
                rotated_data.append(torch.rot90(data[i], 3, dims=[1, 2]))
        
        rotated_data = torch.stack(rotated_data)
        
        # Training loop
        model.train()
        total_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            optimizer.zero_grad()
            
            outputs = model(rotated_data)
            loss = criterion(outputs, rotation_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
        
        training_result = {
            'task_type': PretextTaskType.ROTATION_PREDICTION.value,
            'total_loss': total_loss,
            'epochs': self.config.num_epochs,
            'status': 'success'
        }
        
        return training_result

    def _train_colorization(self, data: torch.Tensor, 
                           labels: torch.Tensor = None) -> Dict[str, Any]:
        """Train colorization task"""
        if 'colorization' not in self.task_models:
            self.task_models['colorization'] = self.create_colorization_model()
        
        model = self.task_models['colorization']
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to grayscale
        grayscale_data = torch.mean(data, dim=1, keepdim=True)
        
        # Training loop
        model.train()
        total_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            optimizer.zero_grad()
            
            outputs = model(grayscale_data)
            loss = criterion(outputs, data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
        
        training_result = {
            'task_type': PretextTaskType.COLORIZATION.value,
            'total_loss': total_loss,
            'epochs': self.config.num_epochs,
            'status': 'success'
        }
        
        return training_result

    def _train_inpainting(self, data: torch.Tensor, 
                         labels: torch.Tensor = None) -> Dict[str, Any]:
        """Train inpainting task"""
        if 'inpainting' not in self.task_models:
            self.task_models['inpainting'] = self.create_inpainting_model()
        
        model = self.task_models['inpainting']
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # Create masked data
        masked_data = data.clone()
        mask = torch.rand_like(data) > 0.5
        masked_data[mask] = 0.0
        
        # Training loop
        model.train()
        total_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            optimizer.zero_grad()
            
            outputs = model(masked_data)
            loss = criterion(outputs, data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
        
        training_result = {
            'task_type': PretextTaskType.INPAINTING.value,
            'total_loss': total_loss,
            'epochs': self.config.num_epochs,
            'status': 'success'
        }
        
        return training_result
    
    def _train_generic_pretext_task(self, data: torch.Tensor, 
                                  labels: torch.Tensor = None) -> Dict[str, Any]:
        """Train generic pretext task"""
        logger.info("🎯 Training generic pretext task")
        
        # Simple reconstruction task
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        total_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
        
        training_result = {
            'task_type': 'generic_reconstruction',
            'total_loss': total_loss,
            'epochs': self.config.num_epochs,
            'status': 'success'
        }
        
        return training_result

