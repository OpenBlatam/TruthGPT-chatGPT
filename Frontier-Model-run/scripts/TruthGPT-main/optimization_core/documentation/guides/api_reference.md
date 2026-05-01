# Referencia de API - TruthGPT

Esta guía proporciona una referencia completa de todas las APIs disponibles en TruthGPT.

## 📋 Tabla de Contenidos

1. [API Principal](#api-principal)
2. [Optimizadores](#optimizadores)
3. [Modelos](#modelos)
4. [Utilidades](#utilidades)
5. [Configuración](#configuración)
6. [Ejemplos de Uso](#ejemplos-de-uso)

## 🚀 API Principal

### ModernTruthGPTOptimizer

```python
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class ModernTruthGPTOptimizer:
    """
    Optimizador principal de TruthGPT
    
    Args:
        config (TruthGPTConfig): Configuración del optimizador
    """
    
    def __init__(self, config: TruthGPTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.initialize()
    
    def initialize(self):
        """Inicializar el optimizador"""
        pass
    
    def generate(self, input_text: str, max_length: int = 100, 
                temperature: float = 0.7, top_p: float = 0.9, 
                top_k: int = 50, repetition_penalty: float = 1.0) -> str:
        """
        Generar texto optimizado
        
        Args:
            input_text (str): Texto de entrada
            max_length (int): Longitud máxima de generación
            temperature (float): Temperatura de muestreo (0.0-1.0)
            top_p (float): Nucleus sampling (0.0-1.0)
            top_k (int): Top-k sampling
            repetition_penalty (float): Penalización por repetición
            
        Returns:
            str: Texto generado
        """
        pass
    
    def optimize(self, optimization_config: Dict[str, Any]) -> 'ModernTruthGPTOptimizer':
        """
        Aplicar optimizaciones adicionales
        
        Args:
            optimization_config (Dict): Configuración de optimización
            
        Returns:
            ModernTruthGPTOptimizer: Optimizador optimizado
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo
        
        Returns:
            Dict: Información del modelo
        """
        pass
    
    def save_model(self, path: str):
        """
        Guardar modelo
        
        Args:
            path (str): Ruta donde guardar
        """
        pass
    
    def load_model(self, path: str):
        """
        Cargar modelo
        
        Args:
            path (str): Ruta del modelo
        """
        pass
```

### TruthGPTConfig

```python
from optimization_core import TruthGPTConfig

class TruthGPTConfig:
    """
    Configuración de TruthGPT
    
    Args:
        model_name (str): Nombre del modelo
        use_mixed_precision (bool): Usar precisión mixta
        use_gradient_checkpointing (bool): Usar gradient checkpointing
        use_flash_attention (bool): Usar Flash Attention
        device (str): Dispositivo a usar ('cuda', 'cpu')
        batch_size (int): Tamaño de lote
        max_length (int): Longitud máxima
        temperature (float): Temperatura por defecto
        debug (bool): Modo debug
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 use_mixed_precision: bool = True,
                 use_gradient_checkpointing: bool = False,
                 use_flash_attention: bool = False,
                 device: str = "auto",
                 batch_size: int = 1,
                 max_length: int = 100,
                 temperature: float = 0.7,
                 debug: bool = False):
        self.model_name = model_name
        self.use_mixed_precision = use_mixed_precision
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_flash_attention = use_flash_attention
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.temperature = temperature
        self.debug = debug
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir configuración a diccionario
        
        Returns:
            Dict: Configuración como diccionario
        """
        pass
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """
        Cargar configuración desde diccionario
        
        Args:
            config_dict (Dict): Configuración como diccionario
        """
        pass
    
    def validate(self) -> bool:
        """
        Validar configuración
        
        Returns:
            bool: True si es válida
        """
        pass
```

## ⚡ Optimizadores

### create_ultra_optimization_core

```python
from optimization_core import create_ultra_optimization_core

def create_ultra_optimization_core(config: Dict[str, Any]) -> 'UltraOptimizer':
    """
    Crear optimizador ultra avanzado
    
    Args:
        config (Dict): Configuración de optimización
            - use_quantization (bool): Usar cuantización
            - use_kernel_fusion (bool): Usar fusión de kernels
            - use_memory_pooling (bool): Usar pool de memoria
            - use_adaptive_precision (bool): Usar precisión adaptativa
            - use_parallel_processing (bool): Usar procesamiento paralelo
            
    Returns:
        UltraOptimizer: Optimizador ultra
    """
    pass

class UltraOptimizer:
    """
    Optimizador ultra avanzado
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimizations = []
        self.performance_metrics = {}
    
    def optimize(self, optimizer: ModernTruthGPTOptimizer) -> ModernTruthGPTOptimizer:
        """
        Aplicar optimizaciones ultra
        
        Args:
            optimizer (ModernTruthGPTOptimizer): Optimizador base
            
        Returns:
            ModernTruthGPTOptimizer: Optimizador optimizado
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Obtener métricas de rendimiento
        
        Returns:
            Dict: Métricas de rendimiento
        """
        pass
    
    def benchmark(self, test_data: List[str]) -> Dict[str, Any]:
        """
        Ejecutar benchmark
        
        Args:
            test_data (List[str]): Datos de prueba
            
        Returns:
            Dict: Resultados del benchmark
        """
        pass
```

### create_memory_optimizer

```python
from optimization_core import create_memory_optimizer

def create_memory_optimizer(config: Dict[str, Any]) -> 'MemoryOptimizer':
    """
    Crear optimizador de memoria
    
    Args:
        config (Dict): Configuración de memoria
            - use_gradient_checkpointing (bool): Usar gradient checkpointing
            - use_activation_checkpointing (bool): Usar activation checkpointing
            - use_memory_efficient_attention (bool): Usar atención eficiente
            - use_offload (bool): Usar offload
            - memory_threshold (float): Umbral de memoria
            
    Returns:
        MemoryOptimizer: Optimizador de memoria
    """
    pass

class MemoryOptimizer:
    """
    Optimizador de memoria
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_usage = {}
        self.optimization_strategies = []
    
    def optimize(self, optimizer: ModernTruthGPTOptimizer) -> ModernTruthGPTOptimizer:
        """
        Aplicar optimizaciones de memoria
        
        Args:
            optimizer (ModernTruthGPTOptimizer): Optimizador base
            
        Returns:
            ModernTruthGPTOptimizer: Optimizador optimizado
        """
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Obtener uso de memoria
        
        Returns:
            Dict: Uso de memoria por componente
        """
        pass
    
    def optimize_memory_allocation(self) -> Dict[str, Any]:
        """
        Optimizar asignación de memoria
        
        Returns:
            Dict: Resultados de optimización
        """
        pass
```

### create_gpu_accelerator

```python
from optimization_core import create_gpu_accelerator

def create_gpu_accelerator(config: Dict[str, Any]) -> 'GPUAccelerator':
    """
    Crear acelerador GPU
    
    Args:
        config (Dict): Configuración de GPU
            - cuda_device (int): Dispositivo CUDA
            - use_mixed_precision (bool): Usar precisión mixta
            - use_tensor_cores (bool): Usar Tensor Cores
            - use_cuda_graphs (bool): Usar CUDA Graphs
            - memory_fraction (float): Fracción de memoria GPU
            
    Returns:
        GPUAccelerator: Acelerador GPU
    """
    pass

class GPUAccelerator:
    """
    Acelerador GPU
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gpu_info = {}
        self.acceleration_techniques = []
    
    def optimize(self, optimizer: ModernTruthGPTOptimizer) -> ModernTruthGPTOptimizer:
        """
        Aplicar aceleración GPU
        
        Args:
            optimizer (ModernTruthGPTOptimizer): Optimizador base
            
        Returns:
            ModernTruthGPTOptimizer: Optimizador acelerado
        """
        pass
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Obtener información de GPU
        
        Returns:
            Dict: Información de GPU
        """
        pass
    
    def benchmark_gpu_performance(self) -> Dict[str, float]:
        """
        Benchmark de rendimiento GPU
        
        Returns:
            Dict: Métricas de rendimiento GPU
        """
        pass
```

## 🧠 Modelos

### AdvancedTransformerModel

```python
from optimization_core import AdvancedTransformerModel

class AdvancedTransformerModel:
    """
    Modelo Transformer avanzado
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizations = []
    
    def load_model(self, model_name: str):
        """
        Cargar modelo
        
        Args:
            model_name (str): Nombre del modelo
        """
        pass
    
    def apply_optimizations(self, optimizations: List[str]):
        """
        Aplicar optimizaciones
        
        Args:
            optimizations (List[str]): Lista de optimizaciones
        """
        pass
    
    def generate(self, input_text: str, **kwargs) -> str:
        """
        Generar texto
        
        Args:
            input_text (str): Texto de entrada
            **kwargs: Parámetros adicionales
            
        Returns:
            str: Texto generado
        """
        pass
    
    def fine_tune(self, training_data: List[Dict[str, str]], 
                  epochs: int = 3, learning_rate: float = 5e-5):
        """
        Fine-tuning del modelo
        
        Args:
            training_data (List[Dict]): Datos de entrenamiento
            epochs (int): Número de épocas
            learning_rate (float): Tasa de aprendizaje
        """
        pass
```

### LoRALinear

```python
from optimization_core import LoRALinear

class LoRALinear:
    """
    Capa LoRA (Low-Rank Adaptation)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 16, alpha: float = 32.0):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.lora_a = None
        self.lora_b = None
        self.scaling = alpha / rank
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_parameters(self) -> List[torch.nn.Parameter]:
        """
        Obtener parámetros LoRA
        
        Returns:
            List[torch.nn.Parameter]: Parámetros LoRA
        """
        pass
    
    def merge_weights(self):
        """
        Fusionar pesos LoRA con pesos base
        """
        pass
```

### FlashAttentionModel

```python
from optimization_core import FlashAttentionModel

class FlashAttentionModel:
    """
    Modelo con Flash Attention
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.flash_attention = None
    
    def enable_flash_attention(self):
        """
        Habilitar Flash Attention
        """
        pass
    
    def optimize_attention(self, attention_config: Dict[str, Any]):
        """
        Optimizar atención
        
        Args:
            attention_config (Dict): Configuración de atención
        """
        pass
    
    def get_attention_metrics(self) -> Dict[str, float]:
        """
        Obtener métricas de atención
        
        Returns:
            Dict: Métricas de atención
        """
        pass
```

## 🛠️ Utilidades

### ModelRegistry

```python
from optimization_core import ModelRegistry

class ModelRegistry:
    """
    Registro de modelos
    """
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
    
    def register_model(self, name: str, model: Any, metadata: Dict[str, Any]):
        """
        Registrar modelo
        
        Args:
            name (str): Nombre del modelo
            model (Any): Instancia del modelo
            metadata (Dict): Metadatos del modelo
        """
        pass
    
    def get_model(self, name: str) -> Any:
        """
        Obtener modelo
        
        Args:
            name (str): Nombre del modelo
            
        Returns:
            Any: Instancia del modelo
        """
        pass
    
    def list_models(self) -> List[str]:
        """
        Listar modelos disponibles
        
        Returns:
            List[str]: Lista de nombres de modelos
        """
        pass
    
    def get_model_metadata(self, name: str) -> Dict[str, Any]:
        """
        Obtener metadatos del modelo
        
        Args:
            name (str): Nombre del modelo
            
        Returns:
            Dict: Metadatos del modelo
        """
        pass
```

### OptimizationEngine

```python
from optimization_core import OptimizationEngine

class OptimizationEngine:
    """
    Motor de optimización
    """
    
    def __init__(self):
        self.optimizations = []
        self.performance_tracker = None
    
    def add_optimization(self, optimization: str, config: Dict[str, Any]):
        """
        Agregar optimización
        
        Args:
            optimization (str): Nombre de la optimización
            config (Dict): Configuración de la optimización
        """
        pass
    
    def apply_optimizations(self, model: Any) -> Any:
        """
        Aplicar optimizaciones
        
        Args:
            model (Any): Modelo a optimizar
            
        Returns:
            Any: Modelo optimizado
        """
        pass
    
    def benchmark_optimizations(self, model: Any, test_data: List[str]) -> Dict[str, Any]:
        """
        Benchmark de optimizaciones
        
        Args:
            model (Any): Modelo a probar
            test_data (List[str]): Datos de prueba
            
        Returns:
            Dict: Resultados del benchmark
        """
        pass
```

### AdvancedTrainer

```python
from optimization_core import AdvancedTrainer

class AdvancedTrainer:
    """
    Entrenador avanzado
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = {}
    
    def setup_training(self, model: Any, training_config: Dict[str, Any]):
        """
        Configurar entrenamiento
        
        Args:
            model (Any): Modelo a entrenar
            training_config (Dict): Configuración de entrenamiento
        """
        pass
    
    def train(self, training_data: List[Dict[str, str]], 
              validation_data: List[Dict[str, str]] = None):
        """
        Entrenar modelo
        
        Args:
            training_data (List[Dict]): Datos de entrenamiento
            validation_data (List[Dict]): Datos de validación
        """
        pass
    
    def evaluate(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Evaluar modelo
        
        Args:
            test_data (List[Dict]): Datos de prueba
            
        Returns:
            Dict: Métricas de evaluación
        """
        pass
    
    def save_checkpoint(self, path: str):
        """
        Guardar checkpoint
        
        Args:
            path (str): Ruta del checkpoint
        """
        pass
    
    def load_checkpoint(self, path: str):
        """
        Cargar checkpoint
        
        Args:
            path (str): Ruta del checkpoint
        """
        pass
```

## 📊 Configuración

### Configuración Básica

```python
# Configuración básica
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True,
    device="cuda"
)

optimizer = ModernTruthGPTOptimizer(config)
```

### Configuración Avanzada

```python
# Configuración avanzada
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-large",
    use_mixed_precision=True,
    use_gradient_checkpointing=True,
    use_flash_attention=True,
    device="cuda",
    batch_size=4,
    max_length=200,
    temperature=0.7,
    debug=False
)

optimizer = ModernTruthGPTOptimizer(config)
```

### Configuración de Optimización

```python
# Configuración de optimización
optimization_config = {
    "use_quantization": True,
    "use_kernel_fusion": True,
    "use_memory_pooling": True,
    "use_adaptive_precision": True,
    "use_parallel_processing": True
}

ultra_optimizer = create_ultra_optimization_core(optimization_config)
optimized_optimizer = ultra_optimizer.optimize(optimizer)
```

## 💻 Ejemplos de Uso

### Ejemplo 1: Uso Básico

```python
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

# Configuración
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

# Crear optimizador
optimizer = ModernTruthGPTOptimizer(config)

# Generar texto
text = optimizer.generate(
    input_text="Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)

print(f"Texto generado: {text}")
```

### Ejemplo 2: Optimización Avanzada

```python
from optimization_core import (
    ModernTruthGPTOptimizer, 
    TruthGPTConfig,
    create_ultra_optimization_core,
    create_memory_optimizer,
    create_gpu_accelerator
)

# Configuración base
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)

# Optimización ultra
ultra_config = {
    "use_quantization": True,
    "use_kernel_fusion": True,
    "use_memory_pooling": True
}
ultra_optimizer = create_ultra_optimization_core(ultra_config)
optimizer = ultra_optimizer.optimize(optimizer)

# Optimización de memoria
memory_config = {
    "use_gradient_checkpointing": True,
    "use_activation_checkpointing": True
}
memory_optimizer = create_memory_optimizer(memory_config)
optimizer = memory_optimizer.optimize(optimizer)

# Aceleración GPU
gpu_config = {
    "cuda_device": 0,
    "use_mixed_precision": True,
    "use_tensor_cores": True
}
gpu_accelerator = create_gpu_accelerator(gpu_config)
optimizer = gpu_accelerator.optimize(optimizer)

# Generar texto optimizado
text = optimizer.generate(
    input_text="Explica la inteligencia artificial",
    max_length=200,
    temperature=0.7
)

print(f"Texto optimizado: {text}")
```

### Ejemplo 3: Fine-tuning

```python
from optimization_core import AdvancedTrainer, AdvancedTransformerModel

# Crear modelo
model = AdvancedTransformerModel({
    "model_name": "microsoft/DialoGPT-medium",
    "use_mixed_precision": True
})

# Crear entrenador
trainer = AdvancedTrainer({
    "learning_rate": 5e-5,
    "batch_size": 4,
    "epochs": 3
})

# Configurar entrenamiento
trainer.setup_training(model, {
    "optimizer": "adamw",
    "scheduler": "cosine",
    "warmup_steps": 100
})

# Datos de entrenamiento
training_data = [
    {"input": "Hola", "output": "Hola, ¿cómo estás?"},
    {"input": "¿Qué tal?", "output": "Muy bien, gracias por preguntar."},
    {"input": "Buenos días", "output": "Buenos días, ¿en qué puedo ayudarte?"}
]

# Entrenar
trainer.train(training_data)

# Evaluar
test_data = [
    {"input": "Hola", "output": "Hola, ¿cómo estás?"}
]
metrics = trainer.evaluate(test_data)
print(f"Métricas: {metrics}")
```

### Ejemplo 4: Registro de Modelos

```python
from optimization_core import ModelRegistry, ModernTruthGPTOptimizer, TruthGPTConfig

# Crear registro
registry = ModelRegistry()

# Crear modelos
config1 = TruthGPTConfig(model_name="microsoft/DialoGPT-small")
model1 = ModernTruthGPTOptimizer(config1)

config2 = TruthGPTConfig(model_name="microsoft/DialoGPT-medium")
model2 = ModernTruthGPTOptimizer(config2)

# Registrar modelos
registry.register_model("small_model", model1, {
    "description": "Modelo pequeño para desarrollo",
    "parameters": "117M",
    "performance": "fast"
})

registry.register_model("medium_model", model2, {
    "description": "Modelo mediano para producción",
    "parameters": "345M",
    "performance": "balanced"
})

# Listar modelos
models = registry.list_models()
print(f"Modelos disponibles: {models}")

# Obtener modelo
model = registry.get_model("medium_model")
metadata = registry.get_model_metadata("medium_model")
print(f"Metadatos: {metadata}")
```

## 🎯 Próximos Pasos

### 1. Explorar APIs
```python
# Explorar APIs disponibles
from optimization_core import *

# Listar todas las funciones disponibles
print(dir())
```

### 2. Experimentar con Configuraciones
```python
# Experimentar con diferentes configuraciones
configs = [
    {"model_name": "microsoft/DialoGPT-small", "use_mixed_precision": True},
    {"model_name": "microsoft/DialoGPT-medium", "use_mixed_precision": True},
    {"model_name": "microsoft/DialoGPT-large", "use_mixed_precision": True}
]

for config in configs:
    optimizer = ModernTruthGPTOptimizer(TruthGPTConfig(**config))
    # Probar con diferentes configuraciones
```

### 3. Optimizar Rendimiento
```python
# Optimizar rendimiento
optimization_configs = [
    {"use_quantization": True},
    {"use_kernel_fusion": True},
    {"use_memory_pooling": True}
]

for opt_config in optimization_configs:
    ultra_optimizer = create_ultra_optimization_core(opt_config)
    # Probar optimizaciones
```

---

*¡Con esta referencia de API tienes todo lo necesario para usar TruthGPT de manera avanzada! 🚀✨*


