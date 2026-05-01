# Guía de Adaptación de Modelos a TruthGPT

Esta guía te enseñará cómo adaptar modelos existentes para trabajar con TruthGPT, incluyendo modelos pre-entrenados y modelos personalizados.

## 📋 Tabla de Contenidos

1. [Tipos de Adaptación](#tipos-de-adaptación)
2. [Adaptación de Modelos Pre-entrenados](#adaptación-de-modelos-pre-entrenados)
3. [Adaptación de Modelos Personalizados](#adaptación-de-modelos-personalizados)
4. [Migración de Frameworks](#migración-de-frameworks)
5. [Optimización Post-Adaptación](#optimización-post-adaptación)
6. [Casos de Uso Específicos](#casos-de-uso-específicos)

## 🔄 Tipos de Adaptación

### 1. Adaptación Directa
- Modelos que ya son compatibles con PyTorch
- Requieren mínimos cambios
- Optimización automática

### 2. Adaptación con Wrapper
- Modelos de otros frameworks
- Requieren wrapper personalizado
- Mantiene funcionalidad original

### 3. Adaptación Completa
- Reescritura de arquitectura
- Máxima optimización
- Integración nativa

## 🤖 Adaptación de Modelos Pre-entrenados

### 1. Modelos de Hugging Face

```python
from transformers import AutoModel, AutoTokenizer
from optimization_core import TruthGPTConfig, ModernTruthGPTOptimizer

def adapt_huggingface_model(model_name: str, config: TruthGPTConfig):
    """Adapta un modelo de Hugging Face a TruthGPT"""
    
    # Cargar modelo y tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configurar tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Crear wrapper de TruthGPT
    class HuggingFaceTruthGPTWrapper:
        def __init__(self, model, tokenizer, config):
            self.model = model
            self.tokenizer = tokenizer
            self.config = config
        
        def generate(self, input_text: str, max_length: int = 100, **kwargs):
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    **kwargs
                )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        def forward(self, input_ids, **kwargs):
            return self.model(input_ids, **kwargs)
    
    # Crear wrapper
    wrapper = HuggingFaceTruthGPTWrapper(model, tokenizer, config)
    
    # Crear optimizador TruthGPT
    optimizer = ModernTruthGPTOptimizer(config)
    optimizer.model = wrapper
    
    return optimizer
```

### 2. Modelos GPT Personalizados

```python
def adapt_custom_gpt_model(model_path: str, config: TruthGPTConfig):
    """Adapta un modelo GPT personalizado"""
    
    # Cargar modelo personalizado
    custom_model = torch.load(model_path, map_location='cpu')
    
    # Crear wrapper de adaptación
    class CustomGPTTruthGPTWrapper:
        def __init__(self, model, config):
            self.model = model
            self.config = config
            self.tokenizer = self._load_tokenizer()
        
        def _load_tokenizer(self):
            # Cargar tokenizer apropiado
            from transformers import GPT2Tokenizer
            return GPT2Tokenizer.from_pretrained("gpt2")
        
        def generate(self, input_text: str, max_length: int = 100, **kwargs):
            # Implementar generación
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    **kwargs
                )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        def forward(self, input_ids, **kwargs):
            return self.model(input_ids, **kwargs)
    
    # Crear wrapper
    wrapper = CustomGPTTruthGPTWrapper(custom_model, config)
    
    # Aplicar optimizaciones TruthGPT
    from optimization_core import create_ultra_optimization_core
    
    ultra_config = {
        "use_quantization": True,
        "use_kernel_fusion": True,
        "use_memory_pooling": True
    }
    
    ultra_optimizer = create_ultra_optimization_core(ultra_config)
    optimized_wrapper = ultra_optimizer.optimize(wrapper)
    
    return optimized_wrapper
```

### 3. Modelos BERT y Encoder-Only

```python
def adapt_bert_model(model_name: str, config: TruthGPTConfig):
    """Adapta un modelo BERT para generación de texto"""
    
    from transformers import BertModel, BertTokenizer
    
    # Cargar modelo BERT
    bert_model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Agregar cabeza de generación
    class BERTForGeneration(nn.Module):
        def __init__(self, bert_model, vocab_size):
            super().__init__()
            self.bert = bert_model
            self.lm_head = nn.Linear(bert_model.config.hidden_size, vocab_size)
        
        def forward(self, input_ids, attention_mask=None):
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
            return logits
        
        def generate(self, input_ids, max_length=100, **kwargs):
            # Implementar generación autoregresiva
            generated = input_ids.clone()
            
            for _ in range(max_length - input_ids.size(1)):
                with torch.no_grad():
                    logits = self.forward(generated)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            
            return generated
    
    # Crear modelo de generación
    generation_model = BERTForGeneration(bert_model, tokenizer.vocab_size)
    
    # Crear wrapper TruthGPT
    class BERTTruthGPTWrapper:
        def __init__(self, model, tokenizer, config):
            self.model = model
            self.tokenizer = tokenizer
            self.config = config
        
        def generate(self, input_text: str, max_length: int = 100, **kwargs):
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            outputs = self.model.generate(inputs, max_length=max_length, **kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        def forward(self, input_ids, **kwargs):
            return self.model(input_ids, **kwargs)
    
    wrapper = BERTTruthGPTWrapper(generation_model, tokenizer, config)
    return wrapper
```

## 🔧 Adaptación de Modelos Personalizados

### 1. Modelos PyTorch Personalizados

```python
def adapt_pytorch_model(model: nn.Module, config: TruthGPTConfig):
    """Adapta un modelo PyTorch personalizado"""
    
    class PyTorchTruthGPTWrapper:
        def __init__(self, model, config):
            self.model = model
            self.config = config
            self.tokenizer = self._setup_tokenizer()
        
        def _setup_tokenizer(self):
            # Configurar tokenizer apropiado
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        
        def generate(self, input_text: str, max_length: int = 100, **kwargs):
            # Implementar generación
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            with torch.no_grad():
                generated = inputs.clone()
                
                for _ in range(max_length - inputs.size(1)):
                    logits = self.model(generated)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        def forward(self, input_ids, **kwargs):
            return self.model(input_ids, **kwargs)
    
    # Crear wrapper
    wrapper = PyTorchTruthGPTWrapper(model, config)
    
    # Aplicar optimizaciones
    from optimization_core import create_ultra_optimization_core
    
    ultra_config = {
        "use_quantization": True,
        "use_kernel_fusion": True,
        "use_memory_pooling": True,
        "use_adaptive_precision": True
    }
    
    ultra_optimizer = create_ultra_optimization_core(ultra_config)
    optimized_wrapper = ultra_optimizer.optimize(wrapper)
    
    return optimized_wrapper
```

### 2. Modelos con Arquitectura Personalizada

```python
def adapt_custom_architecture(model: nn.Module, config: TruthGPTConfig):
    """Adapta un modelo con arquitectura personalizada"""
    
    class CustomArchitectureWrapper:
        def __init__(self, model, config):
            self.model = model
            self.config = config
            self.tokenizer = self._setup_tokenizer()
        
        def _setup_tokenizer(self):
            # Configurar tokenizer según el modelo
            from transformers import GPT2Tokenizer
            return GPT2Tokenizer.from_pretrained("gpt2")
        
        def generate(self, input_text: str, max_length: int = 100, **kwargs):
            # Implementar generación específica para la arquitectura
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            with torch.no_grad():
                generated = inputs.clone()
                
                for _ in range(max_length - inputs.size(1)):
                    # Usar método específico del modelo
                    if hasattr(self.model, 'generate_step'):
                        next_token = self.model.generate_step(generated)
                    else:
                        logits = self.model(generated)
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    
                    generated = torch.cat([generated, next_token], dim=1)
            
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        def forward(self, input_ids, **kwargs):
            return self.model(input_ids, **kwargs)
    
    # Crear wrapper
    wrapper = CustomArchitectureWrapper(model, config)
    
    # Aplicar optimizaciones específicas
    from optimization_core import (
        create_ultra_optimization_core,
        create_memory_optimizer,
        create_gpu_accelerator
    )
    
    # Configuración de optimizaciones
    ultra_config = {
        "use_quantization": True,
        "use_kernel_fusion": True,
        "use_memory_pooling": True
    }
    
    memory_config = {
        "use_gradient_checkpointing": True,
        "use_activation_checkpointing": True
    }
    
    gpu_config = {
        "cuda_device": 0,
        "use_mixed_precision": True
    }
    
    # Aplicar optimizaciones
    ultra_optimizer = create_ultra_optimization_core(ultra_config)
    memory_optimizer = create_memory_optimizer(memory_config)
    gpu_accelerator = create_gpu_accelerator(gpu_config)
    
    # Optimizar wrapper
    optimized_wrapper = ultra_optimizer.optimize(wrapper)
    optimized_wrapper = memory_optimizer.optimize(optimized_wrapper)
    optimized_wrapper = gpu_accelerator.optimize(optimized_wrapper)
    
    return optimized_wrapper
```

## 🔄 Migración de Frameworks

### 1. De TensorFlow a TruthGPT

```python
def migrate_tensorflow_model(tf_model_path: str, config: TruthGPTConfig):
    """Migra un modelo de TensorFlow a TruthGPT"""
    
    import tensorflow as tf
    import torch
    from transformers import TFAutoModel, AutoTokenizer
    
    # Cargar modelo TensorFlow
    tf_model = tf.keras.models.load_model(tf_model_path)
    
    # Convertir a PyTorch
    class TensorFlowToPyTorchWrapper:
        def __init__(self, tf_model, config):
            self.tf_model = tf_model
            self.config = config
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        def forward(self, input_ids, **kwargs):
            # Convertir tensores PyTorch a TensorFlow
            tf_inputs = tf.convert_to_tensor(input_ids.numpy())
            
            # Ejecutar modelo TensorFlow
            tf_outputs = self.tf_model(tf_inputs)
            
            # Convertir de vuelta a PyTorch
            torch_outputs = torch.from_numpy(tf_outputs.numpy())
            
            return torch_outputs
        
        def generate(self, input_text: str, max_length: int = 100, **kwargs):
            # Implementar generación
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            with torch.no_grad():
                generated = inputs.clone()
                
                for _ in range(max_length - inputs.size(1)):
                    logits = self.forward(generated)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    # Crear wrapper
    wrapper = TensorFlowToPyTorchWrapper(tf_model, config)
    
    # Aplicar optimizaciones TruthGPT
    from optimization_core import create_ultra_optimization_core
    
    ultra_config = {
        "use_quantization": True,
        "use_kernel_fusion": True,
        "use_memory_pooling": True
    }
    
    ultra_optimizer = create_ultra_optimization_core(ultra_config)
    optimized_wrapper = ultra_optimizer.optimize(wrapper)
    
    return optimized_wrapper
```

### 2. De Keras a TruthGPT

```python
def migrate_keras_model(keras_model_path: str, config: TruthGPTConfig):
    """Migra un modelo de Keras a TruthGPT"""
    
    import tensorflow as tf
    import torch
    
    # Cargar modelo Keras
    keras_model = tf.keras.models.load_model(keras_model_path)
    
    # Convertir a PyTorch
    class KerasToPyTorchWrapper:
        def __init__(self, keras_model, config):
            self.keras_model = keras_model
            self.config = config
            self.tokenizer = self._setup_tokenizer()
        
        def _setup_tokenizer(self):
            from transformers import GPT2Tokenizer
            return GPT2Tokenizer.from_pretrained("gpt2")
        
        def forward(self, input_ids, **kwargs):
            # Convertir tensores
            tf_inputs = tf.convert_to_tensor(input_ids.numpy())
            
            # Ejecutar modelo Keras
            tf_outputs = self.keras_model(tf_inputs)
            
            # Convertir a PyTorch
            torch_outputs = torch.from_numpy(tf_outputs.numpy())
            
            return torch_outputs
        
        def generate(self, input_text: str, max_length: int = 100, **kwargs):
            # Implementar generación
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            with torch.no_grad():
                generated = inputs.clone()
                
                for _ in range(max_length - inputs.size(1)):
                    logits = self.forward(generated)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    # Crear wrapper
    wrapper = KerasToPyTorchWrapper(keras_model, config)
    
    # Aplicar optimizaciones
    from optimization_core import create_ultra_optimization_core
    
    ultra_config = {
        "use_quantization": True,
        "use_kernel_fusion": True,
        "use_memory_pooling": True
    }
    
    ultra_optimizer = create_ultra_optimization_core(ultra_config)
    optimized_wrapper = ultra_optimizer.optimize(wrapper)
    
    return optimized_wrapper
```

## ⚡ Optimización Post-Adaptación

### 1. Optimización de Rendimiento

```python
def optimize_adapted_model(wrapper, config: TruthGPTConfig):
    """Optimiza un modelo adaptado"""
    
    from optimization_core import (
        create_ultra_optimization_core,
        create_memory_optimizer,
        create_gpu_accelerator,
        create_quantization_optimizer
    )
    
    # Configuración de optimizaciones
    ultra_config = {
        "use_quantization": True,
        "use_kernel_fusion": True,
        "use_memory_pooling": True,
        "use_adaptive_precision": True
    }
    
    memory_config = {
        "use_gradient_checkpointing": True,
        "use_activation_checkpointing": True,
        "use_memory_efficient_attention": True
    }
    
    gpu_config = {
        "cuda_device": 0,
        "use_mixed_precision": True,
        "use_tensor_cores": True
    }
    
    quantization_config = {
        "quantization_type": "int8",
        "use_dynamic_quantization": True
    }
    
    # Aplicar optimizaciones
    ultra_optimizer = create_ultra_optimization_core(ultra_config)
    memory_optimizer = create_memory_optimizer(memory_config)
    gpu_accelerator = create_gpu_accelerator(gpu_config)
    quantization_optimizer = create_quantization_optimizer(quantization_config)
    
    # Optimizar modelo
    optimized_wrapper = ultra_optimizer.optimize(wrapper)
    optimized_wrapper = memory_optimizer.optimize(optimized_wrapper)
    optimized_wrapper = gpu_accelerator.optimize(optimized_wrapper)
    optimized_wrapper = quantization_optimizer.optimize(optimized_wrapper)
    
    return optimized_wrapper
```

### 2. Fine-tuning Post-Adaptación

```python
def fine_tune_adapted_model(wrapper, train_data, config: TruthGPTConfig):
    """Fine-tune un modelo adaptado"""
    
    from optimization_core import create_training_pipeline, create_lora_optimizer
    
    # Configuración de LoRA
    lora_config = {
        "rank": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "use_peft": True
    }
    
    # Crear optimizador LoRA
    lora_optimizer = create_lora_optimizer(lora_config)
    
    # Aplicar LoRA al modelo
    lora_wrapper = lora_optimizer.apply_lora(wrapper)
    
    # Crear pipeline de entrenamiento
    pipeline = create_training_pipeline(
        model=lora_wrapper,
        config=config,
        use_lora=True
    )
    
    # Entrenar
    results = pipeline.train(train_data, None)
    
    return results
```

## 🎯 Casos de Uso Específicos

### 1. Adaptación de Modelos de Dominio Específico

```python
def adapt_domain_specific_model(model, domain_config: dict, config: TruthGPTConfig):
    """Adapta un modelo de dominio específico"""
    
    class DomainSpecificWrapper:
        def __init__(self, model, domain_config, config):
            self.model = model
            self.domain_config = domain_config
            self.config = config
            self.tokenizer = self._setup_domain_tokenizer()
        
        def _setup_domain_tokenizer(self):
            # Configurar tokenizer específico del dominio
            if domain_config.get("tokenizer_type") == "medical":
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            else:
                from transformers import GPT2Tokenizer
                return GPT2Tokenizer.from_pretrained("gpt2")
        
        def generate(self, input_text: str, max_length: int = 100, **kwargs):
            # Implementar generación específica del dominio
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            with torch.no_grad():
                generated = inputs.clone()
                
                for _ in range(max_length - inputs.size(1)):
                    logits = self.model(generated)
                    
                    # Aplicar restricciones del dominio
                    if self.domain_config.get("use_domain_constraints"):
                        logits = self._apply_domain_constraints(logits)
                    
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        def _apply_domain_constraints(self, logits):
            # Implementar restricciones específicas del dominio
            return logits
        
        def forward(self, input_ids, **kwargs):
            return self.model(input_ids, **kwargs)
    
    # Crear wrapper
    wrapper = DomainSpecificWrapper(model, domain_config, config)
    
    # Aplicar optimizaciones específicas del dominio
    from optimization_core import create_ultra_optimization_core
    
    ultra_config = {
        "use_quantization": True,
        "use_kernel_fusion": True,
        "use_memory_pooling": True,
        "domain_specific": True
    }
    
    ultra_optimizer = create_ultra_optimization_core(ultra_config)
    optimized_wrapper = ultra_optimizer.optimize(wrapper)
    
    return optimized_wrapper
```

### 2. Adaptación de Modelos Multimodales

```python
def adapt_multimodal_model(model, vision_model, config: TruthGPTConfig):
    """Adapta un modelo multimodal"""
    
    class MultimodalWrapper:
        def __init__(self, text_model, vision_model, config):
            self.text_model = text_model
            self.vision_model = vision_model
            self.config = config
            self.tokenizer = self._setup_tokenizer()
        
        def _setup_tokenizer(self):
            from transformers import GPT2Tokenizer
            return GPT2Tokenizer.from_pretrained("gpt2")
        
        def generate(self, input_text: str, images=None, max_length: int = 100, **kwargs):
            # Procesar texto
            text_inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # Procesar imágenes si están disponibles
            if images is not None:
                vision_features = self.vision_model(images)
                # Combinar características de texto e imagen
                combined_inputs = self._combine_text_vision(text_inputs, vision_features)
            else:
                combined_inputs = text_inputs
            
            with torch.no_grad():
                generated = combined_inputs.clone()
                
                for _ in range(max_length - combined_inputs.size(1)):
                    logits = self.text_model(generated)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        def _combine_text_vision(self, text_inputs, vision_features):
            # Implementar combinación de características
            return text_inputs
        
        def forward(self, input_ids, images=None, **kwargs):
            if images is not None:
                vision_features = self.vision_model(images)
                return self.text_model(input_ids, vision_features, **kwargs)
            else:
                return self.text_model(input_ids, **kwargs)
    
    # Crear wrapper
    wrapper = MultimodalWrapper(model, vision_model, config)
    
    # Aplicar optimizaciones multimodales
    from optimization_core import create_ultra_optimization_core
    
    ultra_config = {
        "use_quantization": True,
        "use_kernel_fusion": True,
        "use_memory_pooling": True,
        "multimodal": True
    }
    
    ultra_optimizer = create_ultra_optimization_core(ultra_config)
    optimized_wrapper = ultra_optimizer.optimize(wrapper)
    
    return optimized_wrapper
```

## 📊 Evaluación de Modelos Adaptados

### 1. Métricas de Adaptación

```python
def evaluate_adaptation(wrapper, test_data, original_model=None):
    """Evalúa la calidad de la adaptación"""
    
    metrics = {}
    
    # Métricas de rendimiento
    performance_metrics = benchmark_model(wrapper, test_data)
    metrics.update(performance_metrics)
    
    # Métricas de calidad (si hay modelo original)
    if original_model is not None:
        quality_metrics = compare_with_original(wrapper, original_model, test_data)
        metrics.update(quality_metrics)
    
    # Métricas de compatibilidad
    compatibility_metrics = test_truthgpt_compatibility(wrapper)
    metrics.update(compatibility_metrics)
    
    return metrics

def compare_with_original(adapted_model, original_model, test_data):
    """Compara el modelo adaptado con el original"""
    
    adapted_outputs = []
    original_outputs = []
    
    for batch in test_data:
        # Generar con modelo adaptado
        adapted_output = adapted_model.generate(batch["input_text"])
        adapted_outputs.append(adapted_output)
        
        # Generar con modelo original
        original_output = original_model.generate(batch["input_text"])
        original_outputs.append(original_output)
    
    # Calcular similitud
    similarity = calculate_similarity(adapted_outputs, original_outputs)
    
    return {"similarity": similarity}

def test_truthgpt_compatibility(wrapper):
    """Prueba la compatibilidad con TruthGPT"""
    
    compatibility_tests = {
        "generation": test_generation(wrapper),
        "optimization": test_optimization(wrapper),
        "training": test_training(wrapper)
    }
    
    return compatibility_tests
```

## 🎯 Mejores Prácticas

### 1. Selección de Modelo Base

- **Compatibilidad**: Elige modelos compatibles con PyTorch
- **Tamaño**: Considera el tamaño del modelo vs. recursos disponibles
- **Dominio**: Selecciona modelos apropiados para tu dominio

### 2. Estrategia de Adaptación

- **Wrapper vs. Reescritura**: Decide según la complejidad
- **Optimizaciones**: Aplica optimizaciones apropiadas
- **Testing**: Prueba exhaustivamente la adaptación

### 3. Optimización Post-Adaptación

- **Quantización**: Reduce el tamaño del modelo
- **Fine-tuning**: Ajusta el modelo a tu dominio
- **Monitoreo**: Supervisa el rendimiento

## 🚀 Próximos Pasos

1. **Identifica** el tipo de modelo a adaptar
2. **Selecciona** la estrategia de adaptación
3. **Implementa** el wrapper o reescritura
4. **Optimiza** con TruthGPT
5. **Evalúa** y ajusta según sea necesario

---

*¡Ahora tienes las herramientas para adaptar cualquier modelo a TruthGPT de manera efectiva!*


