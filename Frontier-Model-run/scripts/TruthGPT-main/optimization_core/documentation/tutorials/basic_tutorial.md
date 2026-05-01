# Tutorial Básico de TruthGPT

Este tutorial te guiará paso a paso para empezar con TruthGPT desde cero.

## 📋 Tabla de Contenidos

1. [Preparación del Entorno](#preparación-del-entorno)
2. [Primera Configuración](#primera-configuración)
3. [Primer Uso](#primer-uso)
4. [Generación de Texto](#generación-de-texto)
5. [Entrenamiento Básico](#entrenamiento-básico)
6. [Optimizaciones Iniciales](#optimizaciones-iniciales)

## 🚀 Preparación del Entorno

### Paso 1: Verificar Requisitos

```bash
# Verificar Python
python --version
# Debe ser 3.8 o superior

# Verificar PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Verificar CUDA (opcional pero recomendado)
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

### Paso 2: Instalar Dependencias

```bash
# Crear entorno virtual
python -m venv truthgpt_env

# Activar entorno virtual
# En Windows:
truthgpt_env\Scripts\activate
# En Linux/Mac:
source truthgpt_env/bin/activate

# Instalar dependencias
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install -r requirements_modern.txt
```

### Paso 3: Verificar Instalación

```python
# test_installation.py
try:
    from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig
    print("✅ TruthGPT instalado correctamente")
except ImportError as e:
    print(f"❌ Error de instalación: {e}")
    exit(1)

# Verificar PyTorch
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
```

## ⚙️ Primera Configuración

### Paso 1: Crear Configuración Básica

```python
# config_basic.py
from optimization_core import TruthGPTConfig, ModernTruthGPTOptimizer

# Configuración básica
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",  # Modelo base
    use_mixed_precision=True,                # Usar precisión mixta
    use_gradient_checkpointing=True,        # Ahorrar memoria
    use_flash_attention=True                # Atención optimizada
)

print("Configuración creada:")
print(f"Modelo: {config.model_name}")
print(f"Precisión mixta: {config.use_mixed_precision}")
print(f"Gradient checkpointing: {config.use_gradient_checkpointing}")
print(f"Flash attention: {config.use_flash_attention}")
```

### Paso 2: Inicializar Optimizador

```python
# initialize_optimizer.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

# Crear configuración
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True,
    use_gradient_checkpointing=True,
    use_flash_attention=True
)

# Inicializar optimizador
try:
    optimizer = ModernTruthGPTOptimizer(config)
    print("✅ Optimizador TruthGPT inicializado correctamente")
except Exception as e:
    print(f"❌ Error al inicializar: {e}")
    exit(1)
```

### Paso 3: Verificar Funcionamiento

```python
# test_basic_functionality.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

# Configuración
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

# Inicializar
optimizer = ModernTruthGPTOptimizer(config)

# Probar generación básica
try:
    test_text = optimizer.generate(
        input_text="Hola",
        max_length=50,
        temperature=0.7
    )
    print(f"✅ Generación exitosa: {test_text}")
except Exception as e:
    print(f"❌ Error en generación: {e}")
```

## 🎯 Primer Uso

### Paso 1: Generación Simple

```python
# first_generation.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

# Configuración
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

# Inicializar
optimizer = ModernTruthGPTOptimizer(config)

# Generar texto
input_text = "Hola, ¿cómo estás?"
generated_text = optimizer.generate(
    input_text=input_text,
    max_length=100,
    temperature=0.7
)

print(f"Input: {input_text}")
print(f"Generated: {generated_text}")
```

### Paso 2: Experimentar con Parámetros

```python
# experiment_parameters.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

# Configuración
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

# Inicializar
optimizer = ModernTruthGPTOptimizer(config)

# Probar diferentes temperaturas
input_text = "Explica la inteligencia artificial"
temperatures = [0.3, 0.7, 1.0]

for temp in temperatures:
    generated = optimizer.generate(
        input_text=input_text,
        max_length=150,
        temperature=temp
    )
    print(f"Temperatura {temp}: {generated}\n")
```

### Paso 3: Generación en Lote

```python
# batch_generation.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

# Configuración
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

# Inicializar
optimizer = ModernTruthGPTOptimizer(config)

# Lista de inputs
inputs = [
    "¿Qué es la inteligencia artificial?",
    "Explica el machine learning",
    "¿Cómo funciona un transformer?",
    "Describe el deep learning"
]

# Generar para cada input
for i, input_text in enumerate(inputs):
    generated = optimizer.generate(
        input_text=input_text,
        max_length=200,
        temperature=0.7
    )
    print(f"Input {i+1}: {input_text}")
    print(f"Generated: {generated}\n")
```

## 📝 Generación de Texto

### Paso 1: Chat Simple

```python
# simple_chat.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class SimpleChat:
    def __init__(self):
        # Configuración
        config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        
        # Inicializar optimizador
        self.optimizer = ModernTruthGPTOptimizer(config)
        self.conversation_history = []
    
    def chat(self, user_input):
        # Agregar input del usuario
        self.conversation_history.append(f"Usuario: {user_input}")
        
        # Crear contexto (últimos 3 mensajes)
        context = "\n".join(self.conversation_history[-3:])
        
        # Generar respuesta
        response = self.optimizer.generate(
            input_text=context,
            max_length=200,
            temperature=0.7
        )
        
        # Extraer respuesta del bot
        bot_response = response.split("Usuario:")[-1].strip()
        if "Bot:" in bot_response:
            bot_response = bot_response.split("Bot:")[-1].strip()
        
        # Agregar respuesta a la historia
        self.conversation_history.append(f"Bot: {bot_response}")
        
        return bot_response

# Usar el chat
chat = SimpleChat()

# Conversación
print("=== Chat con TruthGPT ===")
print("Escribe 'salir' para terminar\n")

while True:
    user_input = input("Tú: ")
    if user_input.lower() == 'salir':
        break
    
    response = chat.chat(user_input)
    print(f"Bot: {response}\n")
```

### Paso 2: Generador de Código

```python
# code_generator.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class CodeGenerator:
    def __init__(self):
        # Configuración
        config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        
        # Inicializar optimizador
        self.optimizer = ModernTruthGPTOptimizer(config)
    
    def generate_function(self, description):
        prompt = f"""
        Escribe una función Python que:
        {description}
        
        Código:
        """
        
        code = self.optimizer.generate(
            input_text=prompt,
            max_length=300,
            temperature=0.3  # Baja temperatura para código
        )
        
        return code
    
    def explain_code(self, code):
        prompt = f"""
        Explica este código Python:
        
        {code}
        
        Explicación:
        """
        
        explanation = self.optimizer.generate(
            input_text=prompt,
            max_length=300,
            temperature=0.3
        )
        
        return explanation

# Usar generador de código
generator = CodeGenerator()

# Generar función
description = "calcule el factorial de un número"
function_code = generator.generate_function(description)
print(f"Descripción: {description}")
print(f"Código generado:\n{function_code}")

# Explicar código
code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
explanation = generator.explain_code(code)
print(f"\nCódigo: {code}")
print(f"Explicación: {explanation}")
```

### Paso 3: Generador de Contenido

```python
# content_generator.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class ContentGenerator:
    def __init__(self):
        # Configuración
        config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        
        # Inicializar optimizador
        self.optimizer = ModernTruthGPTOptimizer(config)
    
    def generate_blog_post(self, topic):
        prompt = f"""
        Escribe un blog post sobre: {topic}
        
        Título:
        """
        
        blog_post = self.optimizer.generate(
            input_text=prompt,
            max_length=500,
            temperature=0.7
        )
        
        return blog_post
    
    def generate_social_media_post(self, topic):
        prompt = f"""
        Escribe un post para redes sociales sobre: {topic}
        
        Post:
        """
        
        social_post = self.optimizer.generate(
            input_text=prompt,
            max_length=200,
            temperature=0.8
        )
        
        return social_post

# Usar generador de contenido
generator = ContentGenerator()

# Generar blog post
blog_post = generator.generate_blog_post("inteligencia artificial")
print("Blog post:")
print(blog_post)

# Generar post de redes sociales
social_post = generator.generate_social_media_post("machine learning")
print("\nPost de redes sociales:")
print(social_post)
```

## 🎓 Entrenamiento Básico

### Paso 1: Preparar Datos

```python
# prepare_training_data.py
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {"text": self.texts[idx]}

# Datos de ejemplo
train_texts = [
    "La inteligencia artificial es el futuro",
    "El machine learning es fascinante",
    "Los transformers son increíbles",
    "La tecnología avanza rápidamente",
    "Los algoritmos son inteligentes",
    "El deep learning es poderoso"
]

# Crear dataset
train_dataset = SimpleDataset(train_texts)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

print(f"Dataset creado con {len(train_dataset)} ejemplos")
print(f"Batch size: {train_loader.batch_size}")
```

### Paso 2: Configurar Entrenamiento

```python
# setup_training.py
from optimization_core import create_training_pipeline, TruthGPTConfig

# Configuración de entrenamiento
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True,
    use_gradient_checkpointing=True
)

# Crear pipeline de entrenamiento
pipeline = create_training_pipeline(
    model_name="microsoft/DialoGPT-medium",
    experiment_name="mi_primer_entrenamiento",
    use_wandb=False  # Desactivar WandB para simplicidad
)

print("Pipeline de entrenamiento creado")
print(f"Modelo: {config.model_name}")
print(f"Experimento: mi_primer_entrenamiento")
```

### Paso 3: Ejecutar Entrenamiento

```python
# run_training.py
from optimization_core import create_training_pipeline, TruthGPTConfig
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {"text": self.texts[idx]}

# Datos de entrenamiento
train_texts = [
    "La inteligencia artificial es el futuro",
    "El machine learning es fascinante",
    "Los transformers son increíbles",
    "La tecnología avanza rápidamente"
]

# Crear dataset y dataloader
train_dataset = SimpleDataset(train_texts)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Crear pipeline
pipeline = create_training_pipeline(
    model_name="microsoft/DialoGPT-medium",
    experiment_name="mi_primer_entrenamiento",
    use_wandb=False
)

# Entrenar
print("Iniciando entrenamiento...")
try:
    results = pipeline.train(train_loader, None)
    print("✅ Entrenamiento completado")
    print(f"Resultados: {results}")
except Exception as e:
    print(f"❌ Error en entrenamiento: {e}")
```

## ⚡ Optimizaciones Iniciales

### Paso 1: Optimización de Memoria

```python
# memory_optimization.py
from optimization_core import (
    ModernTruthGPTOptimizer, 
    TruthGPTConfig,
    create_memory_optimizer
)

# Configuración con optimizaciones de memoria
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True,
    use_gradient_checkpointing=True,
    use_flash_attention=True
)

# Inicializar optimizador
optimizer = ModernTruthGPTOptimizer(config)

# Configuración de optimización de memoria
memory_config = {
    "use_gradient_checkpointing": True,
    "use_activation_checkpointing": True,
    "use_memory_efficient_attention": True
}

# Crear optimizador de memoria
memory_optimizer = create_memory_optimizer(memory_config)

# Aplicar optimizaciones
optimized_optimizer = memory_optimizer.optimize(optimizer)

# Probar generación optimizada
generated_text = optimized_optimizer.generate(
    input_text="Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)

print(f"Texto generado con optimización de memoria: {generated_text}")
```

### Paso 2: Optimización de Velocidad

```python
# speed_optimization.py
from optimization_core import (
    ModernTruthGPTOptimizer, 
    TruthGPTConfig,
    create_ultra_fast_optimizer
)

# Configuración
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

# Inicializar optimizador
optimizer = ModernTruthGPTOptimizer(config)

# Configuración de optimización de velocidad
speed_config = {
    "use_parallel_processing": True,
    "use_batch_optimization": True,
    "use_kernel_fusion": True
}

# Crear optimizador ultra rápido
speed_optimizer = create_ultra_fast_optimizer(speed_config)

# Aplicar optimizaciones
fast_optimizer = speed_optimizer.optimize(optimizer)

# Probar velocidad
import time

start_time = time.time()
generated_text = fast_optimizer.generate(
    input_text="Explica la inteligencia artificial",
    max_length=200,
    temperature=0.7
)
end_time = time.time()

print(f"Tiempo de generación: {end_time - start_time:.2f} segundos")
print(f"Texto generado: {generated_text}")
```

### Paso 3: Optimización de GPU

```python
# gpu_optimization.py
from optimization_core import (
    ModernTruthGPTOptimizer, 
    TruthGPTConfig,
    create_gpu_accelerator
)

# Verificar CUDA
import torch
if not torch.cuda.is_available():
    print("CUDA no disponible, usando CPU")
    device = "cpu"
else:
    print(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
    device = "cuda"

# Configuración
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

# Inicializar optimizador
optimizer = ModernTruthGPTOptimizer(config)

# Configuración de aceleración GPU
gpu_config = {
    "cuda_device": 0,
    "use_mixed_precision": True,
    "use_tensor_cores": True,
    "memory_fraction": 0.8
}

# Crear acelerador GPU
gpu_accelerator = create_gpu_accelerator(gpu_config)

# Aplicar aceleración GPU
gpu_optimizer = gpu_accelerator.optimize(optimizer)

# Probar generación con GPU
generated_text = gpu_optimizer.generate(
    input_text="Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)

print(f"Texto generado con GPU: {generated_text}")
```

## 🎯 Próximos Pasos

### Paso 1: Experimentar con Diferentes Modelos

```python
# experiment_models.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

# Lista de modelos para probar
models = [
    "microsoft/DialoGPT-small",
    "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-large"
]

input_text = "Explica la inteligencia artificial"

for model_name in models:
    print(f"\n=== Probando {model_name} ===")
    
    try:
        config = TruthGPTConfig(
            model_name=model_name,
            use_mixed_precision=True
        )
        
        optimizer = ModernTruthGPTOptimizer(config)
        
        generated = optimizer.generate(
            input_text=input_text,
            max_length=150,
            temperature=0.7
        )
        
        print(f"Generated: {generated}")
        
    except Exception as e:
        print(f"Error con {model_name}: {e}")
```

### Paso 2: Crear Tu Propio Caso de Uso

```python
# custom_use_case.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class MyCustomUseCase:
    def __init__(self):
        # Configuración
        config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        
        # Inicializar optimizador
        self.optimizer = ModernTruthGPTOptimizer(config)
    
    def my_custom_function(self, input_text):
        # Implementar tu lógica personalizada
        prompt = f"""
        Procesa este texto: {input_text}
        
        Resultado:
        """
        
        result = self.optimizer.generate(
            input_text=prompt,
            max_length=200,
            temperature=0.7
        )
        
        return result

# Usar tu caso de uso personalizado
my_use_case = MyCustomUseCase()

# Probar
input_text = "Hola, ¿cómo estás?"
result = my_use_case.my_custom_function(input_text)
print(f"Input: {input_text}")
print(f"Result: {result}")
```

### Paso 3: Explorar Optimizaciones Avanzadas

```python
# explore_advanced_optimizations.py
from optimization_core import (
    ModernTruthGPTOptimizer, 
    TruthGPTConfig,
    create_ultra_optimization_core,
    create_memory_optimizer,
    create_gpu_accelerator
)

# Configuración
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

# Inicializar optimizador
optimizer = ModernTruthGPTOptimizer(config)

# Configuración de optimizaciones avanzadas
ultra_config = {
    "use_quantization": True,
    "use_kernel_fusion": True,
    "use_memory_pooling": True,
    "use_adaptive_precision": True
}

# Crear optimizador ultra
ultra_optimizer = create_ultra_optimization_core(ultra_config)

# Aplicar optimizaciones
optimized_optimizer = ultra_optimizer.optimize(optimizer)

# Probar generación optimizada
generated_text = optimized_optimizer.generate(
    input_text="Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)

print(f"Texto generado con optimizaciones avanzadas: {generated_text}")
```

## 🎉 ¡Felicidades!

Has completado el tutorial básico de TruthGPT. Ahora tienes:

- ✅ Entorno configurado
- ✅ Primer uso funcionando
- ✅ Generación de texto
- ✅ Entrenamiento básico
- ✅ Optimizaciones iniciales

### Próximos Pasos Recomendados:

1. **Explora** los ejemplos avanzados
2. **Experimenta** con diferentes modelos
3. **Crea** tus propios casos de uso
4. **Optimiza** según tu hardware
5. **Entrena** modelos personalizados

---

*¡Ahora estás listo para usar TruthGPT de manera efectiva! Consulta la documentación avanzada para casos de uso más complejos.*


