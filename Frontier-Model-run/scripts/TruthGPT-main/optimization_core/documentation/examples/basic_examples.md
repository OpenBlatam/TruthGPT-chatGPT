# Ejemplos Básicos de TruthGPT

Esta sección contiene ejemplos prácticos y simples para empezar con TruthGPT.

## 📋 Tabla de Contenidos

1. [Instalación y Configuración](#instalación-y-configuración)
2. [Uso Básico](#uso-básico)
3. [Generación de Texto](#generación-de-texto)
4. [Entrenamiento Simple](#entrenamiento-simple)
5. [Optimizaciones Básicas](#optimizaciones-básicas)

## 🚀 Instalación y Configuración

### Ejemplo 1: Instalación Básica

```python
# Instalar dependencias
pip install torch transformers accelerate
pip install -r requirements_modern.txt

# Verificar instalación
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
```

### Ejemplo 2: Configuración Inicial

```python
from optimization_core import TruthGPTConfig, ModernTruthGPTOptimizer

# Configuración básica
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True,
    use_gradient_checkpointing=True,
    use_flash_attention=True
)

# Inicializar optimizador
optimizer = ModernTruthGPTOptimizer(config)
print("TruthGPT inicializado correctamente")
```

## 🎯 Uso Básico

### Ejemplo 1: Generación Simple

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
input_text = "Hola, ¿cómo estás?"
generated_text = optimizer.generate(
    input_text=input_text,
    max_length=100,
    temperature=0.7
)

print(f"Input: {input_text}")
print(f"Generated: {generated_text}")
```

### Ejemplo 2: Generación con Parámetros

```python
# Generación con parámetros personalizados
generated_text = optimizer.generate(
    input_text="Explica la inteligencia artificial",
    max_length=200,
    temperature=0.5,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True
)

print(f"Generated: {generated_text}")
```

### Ejemplo 3: Generación en Lote

```python
# Generar múltiples textos
input_texts = [
    "¿Qué es la inteligencia artificial?",
    "Explica el machine learning",
    "¿Cómo funciona un transformer?"
]

generated_texts = []
for text in input_texts:
    generated = optimizer.generate(
        input_text=text,
        max_length=150,
        temperature=0.7
    )
    generated_texts.append(generated)
    print(f"Input: {text}")
    print(f"Generated: {generated}\n")
```

## 📝 Generación de Texto

### Ejemplo 1: Chat Simple

```python
class SimpleChat:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.conversation_history = []
    
    def chat(self, user_input):
        # Agregar input del usuario
        self.conversation_history.append(f"Usuario: {user_input}")
        
        # Crear contexto de conversación
        context = "\n".join(self.conversation_history[-5:])  # Últimos 5 mensajes
        
        # Generar respuesta
        response = self.optimizer.generate(
            input_text=context,
            max_length=200,
            temperature=0.7
        )
        
        # Extraer solo la respuesta del bot
        bot_response = response.split("Usuario:")[-1].strip()
        if "Bot:" in bot_response:
            bot_response = bot_response.split("Bot:")[-1].strip()
        
        # Agregar respuesta a la historia
        self.conversation_history.append(f"Bot: {bot_response}")
        
        return bot_response

# Usar el chat
chat = SimpleChat(optimizer)

# Conversación
user_input = "Hola, ¿cómo estás?"
response = chat.chat(user_input)
print(f"Usuario: {user_input}")
print(f"Bot: {response}")

user_input = "¿Qué puedes hacer?"
response = chat.chat(user_input)
print(f"Usuario: {user_input}")
print(f"Bot: {response}")
```

### Ejemplo 2: Generación de Código

```python
def generate_code(optimizer, description):
    """Genera código basado en una descripción"""
    
    prompt = f"""
    Descripción: {description}
    
    Código Python:
    """
    
    generated_code = optimizer.generate(
        input_text=prompt,
        max_length=300,
        temperature=0.3,  # Baja temperatura para código más determinístico
        top_p=0.9
    )
    
    return generated_code

# Ejemplos de generación de código
descriptions = [
    "Una función que calcule el factorial de un número",
    "Una clase para manejar una lista de tareas",
    "Una función que ordene una lista de números"
]

for desc in descriptions:
    code = generate_code(optimizer, desc)
    print(f"Descripción: {desc}")
    print(f"Código generado:\n{code}\n")
```

### Ejemplo 3: Generación de Texto Creativo

```python
def generate_creative_text(optimizer, prompt, style="story"):
    """Genera texto creativo en diferentes estilos"""
    
    if style == "story":
        full_prompt = f"Escribe una historia corta sobre: {prompt}"
    elif style == "poem":
        full_prompt = f"Escribe un poema sobre: {prompt}"
    elif style == "article":
        full_prompt = f"Escribe un artículo sobre: {prompt}"
    else:
        full_prompt = f"Escribe sobre: {prompt}"
    
    generated_text = optimizer.generate(
        input_text=full_prompt,
        max_length=400,
        temperature=0.8,  # Alta temperatura para creatividad
        top_p=0.95
    )
    
    return generated_text

# Ejemplos de texto creativo
prompts = [
    "un robot que aprende a soñar",
    "una ciudad flotante en las nubes",
    "el último árbol en la Tierra"
]

styles = ["story", "poem", "article"]

for prompt in prompts:
    for style in styles:
        text = generate_creative_text(optimizer, prompt, style)
        print(f"Estilo: {style}")
        print(f"Prompt: {prompt}")
        print(f"Texto generado:\n{text}\n")
```

## 🎓 Entrenamiento Simple

### Ejemplo 1: Fine-tuning Básico

```python
from optimization_core import create_training_pipeline
import torch
from torch.utils.data import DataLoader, Dataset

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
    "La tecnología avanza rápidamente"
]

# Crear dataset
train_dataset = SimpleDataset(train_texts)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Crear pipeline de entrenamiento
pipeline = create_training_pipeline(
    model_name="microsoft/DialoGPT-medium",
    experiment_name="mi_primer_entrenamiento",
    use_wandb=False  # Desactivar WandB para simplicidad
)

# Entrenar
results = pipeline.train(train_loader, None)
print("Entrenamiento completado")
print(f"Resultados: {results}")
```

### Ejemplo 2: Entrenamiento con Validación

```python
# Datos de entrenamiento y validación
train_texts = [
    "La IA es increíble",
    "El machine learning es poderoso",
    "Los algoritmos son inteligentes"
]

val_texts = [
    "La tecnología es genial",
    "Los datos son importantes"
]

# Crear datasets
train_dataset = SimpleDataset(train_texts)
val_dataset = SimpleDataset(val_texts)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Crear pipeline con validación
pipeline = create_training_pipeline(
    model_name="microsoft/DialoGPT-medium",
    experiment_name="entrenamiento_con_validacion",
    use_wandb=False
)

# Entrenar con validación
results = pipeline.train(train_loader, val_loader)
print("Entrenamiento con validación completado")
print(f"Resultados: {results}")
```

### Ejemplo 3: Entrenamiento con LoRA

```python
from optimization_core import create_lora_optimizer

# Configuración LoRA
lora_config = {
    "rank": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "use_peft": True
}

# Crear optimizador LoRA
lora_optimizer = create_lora_optimizer(lora_config)

# Crear pipeline con LoRA
pipeline = create_training_pipeline(
    model_name="microsoft/DialoGPT-medium",
    experiment_name="entrenamiento_lora",
    use_lora=True,
    lora_config=lora_config
)

# Entrenar con LoRA
results = pipeline.train(train_loader, val_loader)
print("Entrenamiento con LoRA completado")
print(f"Resultados: {results}")
```

## ⚡ Optimizaciones Básicas

### Ejemplo 1: Optimización de Memoria

```python
from optimization_core import create_memory_optimizer

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

# Usar optimizador optimizado
generated_text = optimized_optimizer.generate(
    input_text="Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)

print(f"Texto generado con optimización de memoria: {generated_text}")
```

### Ejemplo 2: Optimización de Velocidad

```python
from optimization_core import create_ultra_fast_optimizer

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

# Usar optimizador rápido
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

### Ejemplo 3: Optimización de GPU

```python
from optimization_core import create_gpu_accelerator

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

# Usar optimizador con GPU
generated_text = gpu_optimizer.generate(
    input_text="Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)

print(f"Texto generado con GPU: {generated_text}")
```

## 🔧 Casos de Uso Prácticos

### Ejemplo 1: Asistente de Código

```python
class CodeAssistant:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
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
    
    def debug_code(self, code, error):
        prompt = f"""
        Este código tiene un error:
        
        {code}
        
        Error: {error}
        
        Solución:
        """
        
        solution = self.optimizer.generate(
            input_text=prompt,
            max_length=300,
            temperature=0.3
        )
        
        return solution
    
    def generate_function(self, description):
        prompt = f"""
        Escribe una función Python que:
        {description}
        
        Código:
        """
        
        code = self.optimizer.generate(
            input_text=prompt,
            max_length=300,
            temperature=0.3
        )
        
        return code

# Usar asistente de código
assistant = CodeAssistant(optimizer)

# Explicar código
code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
explanation = assistant.explain_code(code)
print(f"Código: {code}")
print(f"Explicación: {explanation}")

# Generar función
description = "calcule el área de un círculo"
function_code = assistant.generate_function(description)
print(f"Descripción: {description}")
print(f"Función generada: {function_code}")
```

### Ejemplo 2: Generador de Contenido

```python
class ContentGenerator:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
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
    
    def generate_email(self, purpose, recipient):
        prompt = f"""
        Escribe un email {purpose} para {recipient}
        
        Email:
        """
        
        email = self.optimizer.generate(
            input_text=prompt,
            max_length=300,
            temperature=0.6
        )
        
        return email

# Usar generador de contenido
generator = ContentGenerator(optimizer)

# Generar blog post
blog_post = generator.generate_blog_post("inteligencia artificial")
print(f"Blog post: {blog_post}")

# Generar post de redes sociales
social_post = generator.generate_social_media_post("machine learning")
print(f"Post social: {social_post}")

# Generar email
email = generator.generate_email("de presentación", "un cliente potencial")
print(f"Email: {email}")
```

### Ejemplo 3: Tutor de Aprendizaje

```python
class LearningTutor:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.learning_history = []
    
    def explain_concept(self, concept):
        prompt = f"""
        Explica el concepto de {concept} de manera simple y clara:
        
        Explicación:
        """
        
        explanation = self.optimizer.generate(
            input_text=prompt,
            max_length=400,
            temperature=0.5
        )
        
        self.learning_history.append(f"Concepto: {concept}")
        return explanation
    
    def generate_quiz(self, topic):
        prompt = f"""
        Crea un quiz sobre {topic} con 3 preguntas:
        
        Quiz:
        """
        
        quiz = self.optimizer.generate(
            input_text=prompt,
            max_length=300,
            temperature=0.6
        )
        
        return quiz
    
    def provide_examples(self, concept):
        prompt = f"""
        Proporciona ejemplos prácticos de {concept}:
        
        Ejemplos:
        """
        
        examples = self.optimizer.generate(
            input_text=prompt,
            max_length=400,
            temperature=0.7
        )
        
        return examples

# Usar tutor de aprendizaje
tutor = LearningTutor(optimizer)

# Explicar concepto
concept = "machine learning"
explanation = tutor.explain_concept(concept)
print(f"Concepto: {concept}")
print(f"Explicación: {explanation}")

# Generar quiz
quiz = tutor.generate_quiz("inteligencia artificial")
print(f"Quiz: {quiz}")

# Proporcionar ejemplos
examples = tutor.provide_examples("algoritmos de clasificación")
print(f"Ejemplos: {examples}")
```

## 🎯 Próximos Pasos

1. **Experimenta** con diferentes parámetros de generación
2. **Prueba** diferentes modelos base
3. **Optimiza** según tu hardware
4. **Crea** tus propios casos de uso
5. **Explora** las optimizaciones avanzadas

---

*¡Estos ejemplos te dan una base sólida para empezar con TruthGPT! Experimenta y adapta según tus necesidades.*


