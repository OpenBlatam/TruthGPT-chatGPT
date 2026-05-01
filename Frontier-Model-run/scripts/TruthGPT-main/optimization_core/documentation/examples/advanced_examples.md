# Ejemplos Avanzados de TruthGPT

Esta sección contiene ejemplos avanzados y casos de uso complejos para TruthGPT.

## 📋 Tabla de Contenidos

1. [Modelos Personalizados Avanzados](#modelos-personalizados-avanzados)
2. [Optimizaciones Complejas](#optimizaciones-complejas)
3. [Entrenamiento Distribuido](#entrenamiento-distribuido)
4. [Casos de Uso Especializados](#casos-de-uso-especializados)
5. [Integración con Sistemas Externos](#integración-con-sistemas-externos)

## 🧠 Modelos Personalizados Avanzados

### Ejemplo 1: Modelo Multimodal

```python
import torch
import torch.nn as nn
from optimization_core import TruthGPTConfig, ModernTruthGPTOptimizer

class MultimodalTruthGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text encoder
        self.text_encoder = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, config.hidden_size)
        )
        
        # Fusion layer
        self.fusion_layer = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_attention_heads
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.hidden_size,
                config.num_attention_heads
            ) for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids, images=None, attention_mask=None):
        # Text encoding
        text_embeds = self.text_encoder(input_ids)
        
        # Vision encoding
        if images is not None:
            vision_embeds = self.vision_encoder(images)
            # Repeat vision embeddings to match sequence length
            vision_embeds = vision_embeds.unsqueeze(1).repeat(1, text_embeds.size(1), 1)
            
            # Fusion
            fused_embeds, _ = self.fusion_layer(
                text_embeds, vision_embeds, vision_embeds
            )
        else:
            fused_embeds = text_embeds
        
        # Transformer layers
        for layer in self.transformer_layers:
            fused_embeds = layer(fused_embeds, src_key_padding_mask=attention_mask)
        
        # Output
        logits = self.lm_head(fused_embeds)
        return logits

# Configuración
config = TruthGPTConfig(
    model_name="custom_multimodal",
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    vocab_size=50257
)

# Crear modelo
model = MultimodalTruthGPTModel(config)

# Crear optimizador
optimizer = ModernTruthGPTOptimizer(config)
optimizer.model = model

# Generar con texto e imagen
text_input = "Describe esta imagen"
image_input = torch.randn(1, 3, 224, 224)  # Imagen de ejemplo

generated = optimizer.generate(
    input_text=text_input,
    images=image_input,
    max_length=200,
    temperature=0.7
)

print(f"Generated: {generated}")
```

### Ejemplo 2: Modelo con Memoria Externa

```python
class MemoryAugmentedTruthGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base model
        self.base_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                config.hidden_size,
                config.num_attention_heads
            ),
            num_layers=config.num_layers
        )
        
        # Memory bank
        self.memory_bank = nn.Parameter(
            torch.randn(config.memory_size, config.hidden_size)
        )
        
        # Memory attention
        self.memory_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads
        )
        
        # Output layer
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids, memory_query=None):
        # Base model forward
        base_output = self.base_model(input_ids)
        
        # Memory attention
        if memory_query is not None:
            memory_output, _ = self.memory_attention(
                memory_query, self.memory_bank, self.memory_bank
            )
            # Combine with base output
            combined_output = base_output + memory_output
        else:
            combined_output = base_output
        
        # Output
        logits = self.lm_head(combined_output)
        return logits

# Configuración
config = TruthGPTConfig(
    model_name="memory_augmented",
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    vocab_size=50257,
    memory_size=1000
)

# Crear modelo
model = MemoryAugmentedTruthGPT(config)

# Crear optimizador
optimizer = ModernTruthGPTOptimizer(config)
optimizer.model = model

# Generar con memoria
text_input = "Recuerda que hablamos sobre IA"
memory_query = torch.randn(1, 1, 768)  # Query de memoria

generated = optimizer.generate(
    input_text=text_input,
    memory_query=memory_query,
    max_length=200,
    temperature=0.7
)

print(f"Generated with memory: {generated}")
```

### Ejemplo 3: Modelo con Arquitectura Híbrida

```python
class HybridTruthGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder (BERT-style)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.hidden_size,
                config.num_attention_heads
            ),
            num_layers=config.num_encoder_layers
        )
        
        # Decoder (GPT-style)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                config.hidden_size,
                config.num_attention_heads
            ),
            num_layers=config.num_decoder_layers
        )
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads
        )
        
        # Output layer
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids, encoder_input=None):
        # Encoder
        if encoder_input is not None:
            encoder_output = self.encoder(encoder_input)
        else:
            encoder_output = None
        
        # Decoder
        decoder_output = self.decoder(input_ids)
        
        # Cross-attention if encoder output available
        if encoder_output is not None:
            attended_output, _ = self.cross_attention(
                decoder_output, encoder_output, encoder_output
            )
            final_output = attended_output
        else:
            final_output = decoder_output
        
        # Output
        logits = self.lm_head(final_output)
        return logits

# Configuración
config = TruthGPTConfig(
    model_name="hybrid_model",
    hidden_size=768,
    num_attention_heads=12,
    vocab_size=50257,
    num_encoder_layers=6,
    num_decoder_layers=6
)

# Crear modelo
model = HybridTruthGPTModel(config)

# Crear optimizador
optimizer = ModernTruthGPTOptimizer(config)
optimizer.model = model

# Generar con contexto
text_input = "Genera una respuesta"
context_input = torch.randint(0, 50257, (1, 100))  # Contexto

generated = optimizer.generate(
    input_text=text_input,
    encoder_input=context_input,
    max_length=200,
    temperature=0.7
)

print(f"Generated with context: {generated}")
```

## ⚡ Optimizaciones Complejas

### Ejemplo 1: Optimización Ultra Completa

```python
from optimization_core import (
    create_ultra_optimization_core,
    create_memory_optimizer,
    create_gpu_accelerator,
    create_quantization_optimizer,
    create_kernel_fusion_optimizer
)

def create_ultra_optimized_model(model, config):
    """Crear modelo ultra optimizado"""
    
    # Configuración ultra
    ultra_config = {
        "use_quantization": True,
        "use_kernel_fusion": True,
        "use_memory_pooling": True,
        "use_adaptive_precision": True,
        "use_dynamic_kernel_fusion": True,
        "use_intelligent_memory_manager": True
    }
    
    # Configuración de memoria
    memory_config = {
        "use_gradient_checkpointing": True,
        "use_activation_checkpointing": True,
        "use_memory_efficient_attention": True,
        "use_offload": True,
        "offload_device": "cpu"
    }
    
    # Configuración GPU
    gpu_config = {
        "cuda_device": 0,
        "use_mixed_precision": True,
        "use_tensor_cores": True,
        "use_cuda_graphs": True,
        "memory_fraction": 0.9
    }
    
    # Configuración de quantización
    quantization_config = {
        "quantization_type": "int8",
        "use_dynamic_quantization": True,
        "use_static_quantization": True,
        "use_qat": True
    }
    
    # Configuración de kernel fusion
    fusion_config = {
        "use_fused_layernorm_linear": True,
        "use_fused_attention_mlp": True,
        "use_fused_gelu": True,
        "fusion_level": "aggressive"
    }
    
    # Crear optimizadores
    ultra_optimizer = create_ultra_optimization_core(ultra_config)
    memory_optimizer = create_memory_optimizer(memory_config)
    gpu_accelerator = create_gpu_accelerator(gpu_config)
    quantization_optimizer = create_quantization_optimizer(quantization_config)
    fusion_optimizer = create_kernel_fusion_optimizer(fusion_config)
    
    # Aplicar optimizaciones en secuencia
    optimized_model = ultra_optimizer.optimize(model)
    optimized_model = memory_optimizer.optimize(optimized_model)
    optimized_model = gpu_accelerator.optimize(optimized_model)
    optimized_model = quantization_optimizer.optimize(optimized_model)
    optimized_model = fusion_optimizer.optimize(optimized_model)
    
    return optimized_model

# Usar optimización ultra
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)
ultra_optimized_optimizer = create_ultra_optimized_model(optimizer, config)

# Generar con modelo ultra optimizado
generated = ultra_optimized_optimizer.generate(
    input_text="Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)

print(f"Generated with ultra optimization: {generated}")
```

### Ejemplo 2: Optimización Adaptativa

```python
from optimization_core import create_adaptive_optimizer

class AdaptiveOptimizationManager:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimization_history = []
        self.current_optimization_level = "basic"
    
    def adapt_optimization(self, performance_metrics):
        """Adaptar optimización basada en métricas de rendimiento"""
        
        # Analizar métricas
        if performance_metrics["memory_usage"] > 0.8:
            self.current_optimization_level = "memory_optimized"
        elif performance_metrics["speed"] < 0.5:
            self.current_optimization_level = "speed_optimized"
        elif performance_metrics["accuracy"] < 0.9:
            self.current_optimization_level = "accuracy_optimized"
        else:
            self.current_optimization_level = "balanced"
        
        # Aplicar optimización adaptativa
        if self.current_optimization_level == "memory_optimized":
            from optimization_core import create_memory_optimizer
            memory_config = {
                "use_gradient_checkpointing": True,
                "use_activation_checkpointing": True,
                "use_memory_efficient_attention": True
            }
            optimizer = create_memory_optimizer(memory_config)
            self.model = optimizer.optimize(self.model)
        
        elif self.current_optimization_level == "speed_optimized":
            from optimization_core import create_ultra_fast_optimizer
            speed_config = {
                "use_parallel_processing": True,
                "use_batch_optimization": True,
                "use_kernel_fusion": True
            }
            optimizer = create_ultra_fast_optimizer(speed_config)
            self.model = optimizer.optimize(self.model)
        
        elif self.current_optimization_level == "accuracy_optimized":
            from optimization_core import create_quantization_optimizer
            quantization_config = {
                "quantization_type": "fp16",
                "use_mixed_precision": True
            }
            optimizer = create_quantization_optimizer(quantization_config)
            self.model = optimizer.optimize(self.model)
        
        # Registrar optimización
        self.optimization_history.append({
            "level": self.current_optimization_level,
            "metrics": performance_metrics,
            "timestamp": time.time()
        })
    
    def get_optimization_report(self):
        """Obtener reporte de optimización"""
        return {
            "current_level": self.current_optimization_level,
            "history": self.optimization_history,
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self):
        """Obtener recomendaciones de optimización"""
        if len(self.optimization_history) < 2:
            return ["Recopilar más datos para recomendaciones"]
        
        recent_metrics = self.optimization_history[-1]["metrics"]
        recommendations = []
        
        if recent_metrics["memory_usage"] > 0.9:
            recommendations.append("Considerar quantización adicional")
        
        if recent_metrics["speed"] < 0.3:
            recommendations.append("Considerar optimización de velocidad")
        
        if recent_metrics["accuracy"] < 0.85:
            recommendations.append("Considerar optimización de precisión")
        
        return recommendations

# Usar optimización adaptativa
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)
adaptive_manager = AdaptiveOptimizationManager(optimizer, config)

# Simular métricas de rendimiento
performance_metrics = {
    "memory_usage": 0.85,
    "speed": 0.4,
    "accuracy": 0.92
}

# Adaptar optimización
adaptive_manager.adapt_optimization(performance_metrics)

# Obtener reporte
report = adaptive_manager.get_optimization_report()
print(f"Optimization report: {report}")
```

### Ejemplo 3: Optimización con Compilación

```python
from optimization_core import (
    create_compiler_core,
    create_jit_compiler,
    create_aot_compiler,
    create_mlir_compiler
)

def create_compiled_optimization(model, config):
    """Crear optimización con compilación"""
    
    # Configuración de compilación
    compiler_config = {
        "target": "cuda",
        "optimization_level": "O3",
        "use_fusion": True,
        "use_quantization": True
    }
    
    # Configuración JIT
    jit_config = {
        "use_jit": True,
        "use_torchscript": True,
        "use_tracing": True,
        "optimization_level": "O3"
    }
    
    # Configuración AOT
    aot_config = {
        "use_aot": True,
        "target": "cuda",
        "optimization_level": "O3",
        "use_fusion": True
    }
    
    # Configuración MLIR
    mlir_config = {
        "use_mlir": True,
        "dialect": "torch",
        "optimization_passes": ["canonicalize", "cse", "loop-fusion"],
        "target": "cuda"
    }
    
    # Crear compiladores
    compiler = create_compiler_core(compiler_config)
    jit_compiler = create_jit_compiler(jit_config)
    aot_compiler = create_aot_compiler(aot_config)
    mlir_compiler = create_mlir_compiler(mlir_config)
    
    # Compilar modelo
    compiled_model = compiler.compile(model)
    jit_compiled_model = jit_compiler.compile(compiled_model)
    aot_compiled_model = aot_compiler.compile(jit_compiled_model)
    mlir_compiled_model = mlir_compiler.compile(aot_compiled_model)
    
    return mlir_compiled_model

# Usar compilación
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)
compiled_optimizer = create_compiled_optimization(optimizer, config)

# Generar con modelo compilado
generated = compiled_optimizer.generate(
    input_text="Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)

print(f"Generated with compilation: {generated}")
```

## 🌐 Entrenamiento Distribuido

### Ejemplo 1: Distributed Data Parallel

```python
from optimization_core import create_distributed_optimizer
import torch.distributed as dist

def setup_distributed_training():
    """Configurar entrenamiento distribuido"""
    
    # Inicializar proceso distribuido
    dist.init_process_group(backend='nccl')
    
    # Configuración distribuida
    distributed_config = {
        "num_nodes": 2,
        "gpus_per_node": 4,
        "strategy": "ddp",
        "use_gradient_accumulation": True,
        "accumulation_steps": 4,
        "use_sync_batchnorm": True
    }
    
    # Crear optimizador distribuido
    distributed_optimizer = create_distributed_optimizer(distributed_config)
    
    return distributed_optimizer

def train_distributed_model(model, train_loader, config):
    """Entrenar modelo distribuido"""
    
    # Configurar entrenamiento distribuido
    distributed_optimizer = setup_distributed_training()
    
    # Aplicar optimizaciones distribuidas
    distributed_model = distributed_optimizer.optimize(model)
    
    # Configurar entrenamiento
    training_config = {
        "num_epochs": 3,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "use_mixed_precision": True,
        "use_gradient_checkpointing": True
    }
    
    # Entrenar
    results = distributed_optimizer.train(
        distributed_model,
        train_loader,
        training_config
    )
    
    return results

# Usar entrenamiento distribuido
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)
train_loader = create_train_loader()  # Tu dataloader

results = train_distributed_model(optimizer, train_loader, config)
print(f"Distributed training results: {results}")
```

### Ejemplo 2: Pipeline Parallelism

```python
def setup_pipeline_parallelism():
    """Configurar pipeline parallelism"""
    
    # Configuración de pipeline
    pipeline_config = {
        "num_stages": 4,
        "micro_batch_size": 2,
        "use_interleaved_pipeline": True,
        "use_1f1b_scheduling": True,
        "use_gradient_checkpointing": True
    }
    
    # Crear optimizador de pipeline
    pipeline_optimizer = create_distributed_optimizer(pipeline_config)
    
    return pipeline_optimizer

def train_pipeline_parallel_model(model, train_loader, config):
    """Entrenar modelo con pipeline parallelism"""
    
    # Configurar pipeline parallelism
    pipeline_optimizer = setup_pipeline_parallelism()
    
    # Aplicar optimizaciones de pipeline
    pipeline_model = pipeline_optimizer.optimize(model)
    
    # Configurar entrenamiento
    training_config = {
        "num_epochs": 3,
        "learning_rate": 5e-5,
        "micro_batch_size": 2,
        "use_mixed_precision": True
    }
    
    # Entrenar
    results = pipeline_optimizer.train(
        pipeline_model,
        train_loader,
        training_config
    )
    
    return results

# Usar pipeline parallelism
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)
train_loader = create_train_loader()

results = train_pipeline_parallel_model(optimizer, train_loader, config)
print(f"Pipeline parallel training results: {results}")
```

### Ejemplo 3: Tensor Parallelism

```python
def setup_tensor_parallelism():
    """Configurar tensor parallelism"""
    
    # Configuración de tensor parallelism
    tensor_config = {
        "tensor_parallel_size": 4,
        "use_tensor_parallelism": True,
        "use_pipeline_parallelism": True,
        "use_data_parallelism": True,
        "use_hybrid_parallelism": True
    }
    
    # Crear optimizador de tensor
    tensor_optimizer = create_distributed_optimizer(tensor_config)
    
    return tensor_optimizer

def train_tensor_parallel_model(model, train_loader, config):
    """Entrenar modelo con tensor parallelism"""
    
    # Configurar tensor parallelism
    tensor_optimizer = setup_tensor_parallelism()
    
    # Aplicar optimizaciones de tensor
    tensor_model = tensor_optimizer.optimize(model)
    
    # Configurar entrenamiento
    training_config = {
        "num_epochs": 3,
        "learning_rate": 5e-5,
        "batch_size": 8,
        "use_mixed_precision": True
    }
    
    # Entrenar
    results = tensor_optimizer.train(
        tensor_model,
        train_loader,
        training_config
    )
    
    return results

# Usar tensor parallelism
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)
train_loader = create_train_loader()

results = train_tensor_parallel_model(optimizer, train_loader, config)
print(f"Tensor parallel training results: {results}")
```

## 🎯 Casos de Uso Especializados

### Ejemplo 1: Asistente de Código Avanzado

```python
class AdvancedCodeAssistant:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.code_context = []
        self.error_patterns = {}
    
    def analyze_code(self, code):
        """Analizar código y detectar problemas"""
        
        prompt = f"""
        Analiza este código Python y detecta problemas:
        
        {code}
        
        Análisis:
        """
        
        analysis = self.optimizer.generate(
            input_text=prompt,
            max_length=400,
            temperature=0.3
        )
        
        return analysis
    
    def suggest_improvements(self, code):
        """Sugerir mejoras para el código"""
        
        prompt = f"""
        Sugiere mejoras para este código Python:
        
        {code}
        
        Mejoras sugeridas:
        """
        
        improvements = self.optimizer.generate(
            input_text=prompt,
            max_length=400,
            temperature=0.4
        )
        
        return improvements
    
    def generate_tests(self, code):
        """Generar tests para el código"""
        
        prompt = f"""
        Genera tests unitarios para este código Python:
        
        {code}
        
        Tests:
        """
        
        tests = self.optimizer.generate(
            input_text=prompt,
            max_length=500,
            temperature=0.3
        )
        
        return tests
    
    def refactor_code(self, code, refactor_type):
        """Refactorizar código"""
        
        prompt = f"""
        Refactoriza este código Python para {refactor_type}:
        
        {code}
        
        Código refactorizado:
        """
        
        refactored_code = self.optimizer.generate(
            input_text=prompt,
            max_length=600,
            temperature=0.3
        )
        
        return refactored_code

# Usar asistente de código avanzado
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)
assistant = AdvancedCodeAssistant(optimizer)

# Analizar código
code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""

analysis = assistant.analyze_code(code)
print(f"Análisis: {analysis}")

improvements = assistant.suggest_improvements(code)
print(f"Mejoras: {improvements}")

tests = assistant.generate_tests(code)
print(f"Tests: {tests}")

refactored = assistant.refactor_code(code, "optimización de rendimiento")
print(f"Refactorizado: {refactored}")
```

### Ejemplo 2: Generador de Contenido Multimodal

```python
class MultimodalContentGenerator:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.content_templates = {}
        self.style_guides = {}
    
    def generate_blog_post(self, topic, style="technical"):
        """Generar blog post con estilo específico"""
        
        style_guide = self.style_guides.get(style, "técnico")
        
        prompt = f"""
        Escribe un blog post sobre {topic} con estilo {style_guide}:
        
        Título:
        """
        
        blog_post = self.optimizer.generate(
            input_text=prompt,
            max_length=800,
            temperature=0.7
        )
        
        return blog_post
    
    def generate_social_media_campaign(self, topic, platforms):
        """Generar campaña para redes sociales"""
        
        campaign = {}
        
        for platform in platforms:
            if platform == "twitter":
                prompt = f"""
                Escribe tweets sobre {topic}:
                
                Tweets:
                """
                max_length = 200
            elif platform == "linkedin":
                prompt = f"""
                Escribe posts de LinkedIn sobre {topic}:
                
                Posts:
                """
                max_length = 400
            elif platform == "instagram":
                prompt = f"""
                Escribe captions de Instagram sobre {topic}:
                
                Captions:
                """
                max_length = 300
            
            content = self.optimizer.generate(
                input_text=prompt,
                max_length=max_length,
                temperature=0.8
            )
            
            campaign[platform] = content
        
        return campaign
    
    def generate_email_sequence(self, purpose, recipient_type):
        """Generar secuencia de emails"""
        
        sequence = []
        
        # Email inicial
        initial_prompt = f"""
        Escribe un email inicial {purpose} para {recipient_type}:
        
        Email inicial:
        """
        
        initial_email = self.optimizer.generate(
            input_text=initial_prompt,
            max_length=300,
            temperature=0.6
        )
        sequence.append(initial_email)
        
        # Email de seguimiento
        followup_prompt = f"""
        Escribe un email de seguimiento {purpose} para {recipient_type}:
        
        Email de seguimiento:
        """
        
        followup_email = self.optimizer.generate(
            input_text=followup_prompt,
            max_length=300,
            temperature=0.6
        )
        sequence.append(followup_email)
        
        return sequence

# Usar generador de contenido multimodal
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)
generator = MultimodalContentGenerator(optimizer)

# Generar blog post
blog_post = generator.generate_blog_post("inteligencia artificial", "técnico")
print(f"Blog post: {blog_post}")

# Generar campaña de redes sociales
campaign = generator.generate_social_media_campaign(
    "machine learning", 
    ["twitter", "linkedin", "instagram"]
)
print(f"Campaña: {campaign}")

# Generar secuencia de emails
email_sequence = generator.generate_email_sequence(
    "de presentación", 
    "cliente potencial"
)
print(f"Secuencia de emails: {email_sequence}")
```

### Ejemplo 3: Sistema de Análisis de Sentimientos

```python
class SentimentAnalysisSystem:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.sentiment_labels = ["positivo", "negativo", "neutral"]
        self.emotion_labels = ["alegría", "tristeza", "ira", "miedo", "sorpresa"]
    
    def analyze_sentiment(self, text):
        """Analizar sentimiento del texto"""
        
        prompt = f"""
        Analiza el sentimiento de este texto:
        
        {text}
        
        Sentimiento:
        """
        
        sentiment = self.optimizer.generate(
            input_text=prompt,
            max_length=100,
            temperature=0.3
        )
        
        return sentiment
    
    def analyze_emotion(self, text):
        """Analizar emoción del texto"""
        
        prompt = f"""
        Analiza la emoción de este texto:
        
        {text}
        
        Emoción:
        """
        
        emotion = self.optimizer.generate(
            input_text=prompt,
            max_length=100,
            temperature=0.3
        )
        
        return emotion
    
    def generate_response(self, text, sentiment, emotion):
        """Generar respuesta basada en sentimiento y emoción"""
        
        prompt = f"""
        Genera una respuesta apropiada para este texto con sentimiento {sentiment} y emoción {emotion}:
        
        {text}
        
        Respuesta:
        """
        
        response = self.optimizer.generate(
            input_text=prompt,
            max_length=200,
            temperature=0.7
        )
        
        return response
    
    def analyze_conversation(self, conversation):
        """Analizar conversación completa"""
        
        analysis = {
            "overall_sentiment": None,
            "emotion_trends": [],
            "key_insights": []
        }
        
        for message in conversation:
            sentiment = self.analyze_sentiment(message)
            emotion = self.analyze_emotion(message)
            
            analysis["emotion_trends"].append({
                "message": message,
                "sentiment": sentiment,
                "emotion": emotion
            })
        
        # Generar insights
        insights_prompt = f"""
        Genera insights sobre esta conversación:
        
        {conversation}
        
        Insights:
        """
        
        insights = self.optimizer.generate(
            input_text=insights_prompt,
            max_length=300,
            temperature=0.5
        )
        
        analysis["key_insights"] = insights
        
        return analysis

# Usar sistema de análisis de sentimientos
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)
sentiment_system = SentimentAnalysisSystem(optimizer)

# Analizar texto
text = "Estoy muy contento con los resultados del proyecto"
sentiment = sentiment_system.analyze_sentiment(text)
emotion = sentiment_system.analyze_emotion(text)
response = sentiment_system.generate_response(text, sentiment, emotion)

print(f"Texto: {text}")
print(f"Sentimiento: {sentiment}")
print(f"Emoción: {emotion}")
print(f"Respuesta: {response}")

# Analizar conversación
conversation = [
    "Hola, ¿cómo estás?",
    "Muy bien, gracias. ¿Y tú?",
    "Genial, trabajando en un proyecto interesante"
]

conversation_analysis = sentiment_system.analyze_conversation(conversation)
print(f"Análisis de conversación: {conversation_analysis}")
```

## 🔗 Integración con Sistemas Externos

### Ejemplo 1: API REST

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

app = FastAPI()

# Configuración global
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)

class GenerationRequest(BaseModel):
    text: str
    max_length: int = 100
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    generated_text: str
    input_text: str
    parameters: dict

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generar texto usando TruthGPT"""
    
    try:
        generated_text = optimizer.generate(
            input_text=request.text,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return GenerationResponse(
            generated_text=generated_text,
            input_text=request.text,
            parameters={
                "max_length": request.max_length,
                "temperature": request.temperature
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Verificar salud del servicio"""
    return {"status": "healthy", "model": config.model_name}

# Ejecutar servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Ejemplo 2: Integración con Base de Datos

```python
import sqlite3
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class TruthGPTDatabase:
    def __init__(self, db_path, config):
        self.db_path = db_path
        self.config = config
        self.optimizer = ModernTruthGPTOptimizer(config)
        self.init_database()
    
    def init_database(self):
        """Inicializar base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                generated_text TEXT NOT NULL,
                parameters TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                config TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def generate_and_store(self, input_text, max_length=100, temperature=0.7):
        """Generar texto y almacenar en base de datos"""
        
        # Generar texto
        generated_text = self.optimizer.generate(
            input_text=input_text,
            max_length=max_length,
            temperature=temperature
        )
        
        # Almacenar en base de datos
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO generations (input_text, generated_text, parameters)
            VALUES (?, ?, ?)
        """, (
            input_text,
            generated_text,
            str({"max_length": max_length, "temperature": temperature})
        ))
        
        conn.commit()
        conn.close()
        
        return generated_text
    
    def get_generation_history(self, limit=10):
        """Obtener historial de generaciones"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT input_text, generated_text, parameters, timestamp
            FROM generations
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def search_generations(self, query):
        """Buscar generaciones por texto"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT input_text, generated_text, parameters, timestamp
            FROM generations
            WHERE input_text LIKE ? OR generated_text LIKE ?
            ORDER BY timestamp DESC
        """, (f"%{query}%", f"%{query}%"))
        
        results = cursor.fetchall()
        conn.close()
        
        return results

# Usar base de datos
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

db = TruthGPTDatabase("truthgpt.db", config)

# Generar y almacenar
generated = db.generate_and_store(
    "Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)

print(f"Generated: {generated}")

# Obtener historial
history = db.get_generation_history(5)
print(f"History: {history}")

# Buscar generaciones
search_results = db.search_generations("Hola")
print(f"Search results: {search_results}")
```

### Ejemplo 3: Integración con Sistemas de Monitoreo

```python
import time
import psutil
import logging
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class TruthGPTMonitor:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.metrics = []
        self.logger = self.setup_logger()
    
    def setup_logger(self):
        """Configurar logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def monitor_generation(self, input_text, max_length=100, temperature=0.7):
        """Monitorear generación de texto"""
        
        # Métricas antes de la generación
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.Process().cpu_percent()
        
        # Generar texto
        generated_text = self.optimizer.generate(
            input_text=input_text,
            max_length=max_length,
            temperature=temperature
        )
        
        # Métricas después de la generación
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        end_cpu = psutil.Process().cpu_percent()
        
        # Calcular métricas
        generation_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = end_cpu - start_cpu
        
        # Almacenar métricas
        metrics = {
            "timestamp": time.time(),
            "input_length": len(input_text),
            "output_length": len(generated_text),
            "generation_time": generation_time,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage,
            "tokens_per_second": len(generated_text) / generation_time
        }
        
        self.metrics.append(metrics)
        
        # Log métricas
        self.logger.info(f"Generation completed in {generation_time:.2f}s")
        self.logger.info(f"Memory usage: {memory_usage / 1024 / 1024:.2f} MB")
        self.logger.info(f"Tokens per second: {metrics['tokens_per_second']:.2f}")
        
        return generated_text, metrics
    
    def get_performance_report(self):
        """Obtener reporte de rendimiento"""
        if not self.metrics:
            return {"error": "No metrics available"}
        
        # Calcular estadísticas
        generation_times = [m["generation_time"] for m in self.metrics]
        memory_usage = [m["memory_usage"] for m in self.metrics]
        tokens_per_second = [m["tokens_per_second"] for m in self.metrics]
        
        report = {
            "total_generations": len(self.metrics),
            "avg_generation_time": sum(generation_times) / len(generation_times),
            "min_generation_time": min(generation_times),
            "max_generation_time": max(generation_times),
            "avg_memory_usage": sum(memory_usage) / len(memory_usage),
            "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "metrics": self.metrics
        }
        
        return report
    
    def check_performance_alerts(self):
        """Verificar alertas de rendimiento"""
        alerts = []
        
        if not self.metrics:
            return alerts
        
        # Verificar tiempo de generación
        recent_metrics = self.metrics[-5:]  # Últimas 5 generaciones
        avg_time = sum(m["generation_time"] for m in recent_metrics) / len(recent_metrics)
        
        if avg_time > 10:  # Más de 10 segundos
            alerts.append("High generation time detected")
        
        # Verificar uso de memoria
        recent_memory = [m["memory_usage"] for m in recent_metrics]
        avg_memory = sum(recent_memory) / len(recent_memory)
        
        if avg_memory > 1024 * 1024 * 1024:  # Más de 1GB
            alerts.append("High memory usage detected")
        
        # Verificar tokens por segundo
        recent_tps = [m["tokens_per_second"] for m in recent_metrics]
        avg_tps = sum(recent_tps) / len(recent_tps)
        
        if avg_tps < 10:  # Menos de 10 tokens por segundo
            alerts.append("Low generation speed detected")
        
        return alerts

# Usar monitoreo
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True
)

optimizer = ModernTruthGPTOptimizer(config)
monitor = TruthGPTMonitor(optimizer, config)

# Generar con monitoreo
generated, metrics = monitor.monitor_generation(
    "Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)

print(f"Generated: {generated}")
print(f"Metrics: {metrics}")

# Obtener reporte de rendimiento
report = monitor.get_performance_report()
print(f"Performance report: {report}")

# Verificar alertas
alerts = monitor.check_performance_alerts()
if alerts:
    print(f"Alerts: {alerts}")
```

## 🎯 Próximos Pasos

1. **Experimenta** con diferentes arquitecturas de modelos
2. **Optimiza** según tu hardware específico
3. **Integra** con tus sistemas existentes
4. **Monitorea** el rendimiento en producción
5. **Escala** según tus necesidades

---

*¡Estos ejemplos avanzados te dan las herramientas para crear sistemas complejos con TruthGPT!*


