# Tutorial Avanzado - TruthGPT

Este tutorial avanzado te llevará a través de técnicas sofisticadas y casos de uso complejos de TruthGPT.

## 📋 Tabla de Contenidos

1. [Arquitectura Avanzada](#arquitectura-avanzada)
2. [Optimización de Rendimiento](#optimización-de-rendimiento)
3. [Modelos Personalizados](#modelos-personalizados)
4. [Integración con Sistemas Empresariales](#integración-con-sistemas-empresariales)
5. [Monitoreo y Observabilidad](#monitoreo-y-observabilidad)
6. [Seguridad y Compliance](#seguridad-y-compliance)
7. [Escalabilidad](#escalabilidad)
8. [Casos de Uso Avanzados](#casos-de-uso-avanzados)

## 🏗️ Arquitectura Avanzada

### Patrón de Microservicios

```python
# truthgpt_microservices.py
from fastapi import FastAPI, HTTPException
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig
import asyncio
import redis
import json
from typing import Dict, List, Optional

class TruthGPTMicroservices:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.services = {}
        self.load_balancer = LoadBalancer()
        self.circuit_breaker = CircuitBreaker()
    
    def register_service(self, service_name: str, service_config: Dict):
        """Registrar servicio"""
        self.services[service_name] = {
            'config': service_config,
            'status': 'healthy',
            'load': 0,
            'last_health_check': None
        }
    
    async def health_check_all_services(self):
        """Verificar salud de todos los servicios"""
        for service_name, service_info in self.services.items():
            try:
                # Verificar salud del servicio
                health_status = await self.check_service_health(service_name)
                service_info['status'] = health_status
                service_info['last_health_check'] = asyncio.get_event_loop().time()
            except Exception as e:
                service_info['status'] = 'unhealthy'
                print(f"Error en health check de {service_name}: {e}")
    
    async def check_service_health(self, service_name: str) -> str:
        """Verificar salud de un servicio específico"""
        # Implementar verificación de salud
        return "healthy"
    
    def get_optimal_service(self, service_type: str) -> str:
        """Obtener servicio óptimo basado en carga"""
        available_services = [
            name for name, info in self.services.items()
            if info['status'] == 'healthy' and service_type in name
        ]
        
        if not available_services:
            raise HTTPException(status_code=503, detail="No hay servicios disponibles")
        
        # Seleccionar servicio con menor carga
        optimal_service = min(available_services, 
                            key=lambda x: self.services[x]['load'])
        
        return optimal_service
    
    async def process_request(self, request_data: Dict, service_type: str):
        """Procesar solicitud a través de microservicios"""
        try:
            # Obtener servicio óptimo
            service_name = self.get_optimal_service(service_type)
            
            # Aplicar circuit breaker
            if self.circuit_breaker.is_open(service_name):
                raise HTTPException(status_code=503, detail="Service circuit breaker open")
            
            # Procesar solicitud
            result = await self.route_request(service_name, request_data)
            
            # Actualizar métricas
            self.services[service_name]['load'] += 1
            
            return result
            
        except Exception as e:
            # Registrar fallo en circuit breaker
            self.circuit_breaker.record_failure(service_name)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def route_request(self, service_name: str, request_data: Dict):
        """Enrutar solicitud a servicio específico"""
        # Implementar enrutamiento
        pass

class LoadBalancer:
    def __init__(self):
        self.strategies = {
            'round_robin': self.round_robin,
            'least_connections': self.least_connections,
            'weighted': self.weighted
        }
    
    def round_robin(self, services: List[str]) -> str:
        """Balanceador round-robin"""
        # Implementar round-robin
        pass
    
    def least_connections(self, services: List[str]) -> str:
        """Balanceador de menor conexiones"""
        # Implementar least connections
        pass
    
    def weighted(self, services: List[str], weights: Dict[str, int]) -> str:
        """Balanceador ponderado"""
        # Implementar weighted
        pass

class CircuitBreaker:
    def __init__(self):
        self.states = {}
        self.failure_threshold = 5
        self.timeout = 60
    
    def is_open(self, service_name: str) -> bool:
        """Verificar si circuit breaker está abierto"""
        if service_name not in self.states:
            return False
        
        state = self.states[service_name]
        if state['state'] == 'open':
            # Verificar si ha pasado el timeout
            if asyncio.get_event_loop().time() - state['last_failure'] > self.timeout:
                state['state'] = 'half_open'
                return False
            return True
        
        return False
    
    def record_failure(self, service_name: str):
        """Registrar fallo"""
        if service_name not in self.states:
            self.states[service_name] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': 0
            }
        
        state = self.states[service_name]
        state['failure_count'] += 1
        state['last_failure'] = asyncio.get_event_loop().time()
        
        if state['failure_count'] >= self.failure_threshold:
            state['state'] = 'open'
    
    def record_success(self, service_name: str):
        """Registrar éxito"""
        if service_name in self.states:
            self.states[service_name]['state'] = 'closed'
            self.states[service_name]['failure_count'] = 0

# Usar microservicios
microservices = TruthGPTMicroservices()

# Registrar servicios
microservices.register_service('text_generation', {
    'model': 'microsoft/DialoGPT-medium',
    'max_length': 200,
    'temperature': 0.7
})

microservices.register_service('sentiment_analysis', {
    'model': 'microsoft/DialoGPT-medium',
    'task': 'sentiment',
    'temperature': 0.3
})
```

### Patrón de Event Sourcing

```python
# event_sourcing.py
from datetime import datetime
from typing import List, Dict, Any
import json

class Event:
    def __init__(self, event_type: str, data: Dict[str, Any], timestamp: datetime = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()
        self.id = f"{event_type}_{self.timestamp.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

class EventStore:
    def __init__(self):
        self.events: List[Event] = []
        self.projections: Dict[str, Any] = {}
    
    def append_event(self, event: Event):
        """Agregar evento al store"""
        self.events.append(event)
        self.update_projections(event)
    
    def get_events(self, event_type: str = None, from_timestamp: datetime = None) -> List[Event]:
        """Obtener eventos"""
        filtered_events = self.events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if from_timestamp:
            filtered_events = [e for e in filtered_events if e.timestamp >= from_timestamp]
        
        return filtered_events
    
    def update_projections(self, event: Event):
        """Actualizar proyecciones"""
        if event.event_type == 'text_generated':
            self.update_text_generation_projection(event)
        elif event.event_type == 'model_optimized':
            self.update_optimization_projection(event)
    
    def update_text_generation_projection(self, event: Event):
        """Actualizar proyección de generación de texto"""
        if 'text_generation' not in self.projections:
            self.projections['text_generation'] = {
                'total_generations': 0,
                'total_tokens': 0,
                'average_length': 0
            }
        
        projection = self.projections['text_generation']
        projection['total_generations'] += 1
        projection['total_tokens'] += len(event.data.get('generated_text', ''))
        projection['average_length'] = projection['total_tokens'] / projection['total_generations']
    
    def update_optimization_projection(self, event: Event):
        """Actualizar proyección de optimización"""
        if 'optimization' not in self.projections:
            self.projections['optimization'] = {
                'total_optimizations': 0,
                'optimization_types': {}
            }
        
        projection = self.projections['optimization']
        projection['total_optimizations'] += 1
        
        opt_type = event.data.get('optimization_type', 'unknown')
        if opt_type not in projection['optimization_types']:
            projection['optimization_types'][opt_type] = 0
        projection['optimization_types'][opt_type] += 1

class TruthGPTEventSourcing:
    def __init__(self):
        self.event_store = EventStore()
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
    
    def generate_text_with_events(self, input_text: str, max_length: int = 100) -> str:
        """Generar texto con eventos"""
        # Evento: inicio de generación
        start_event = Event('generation_started', {
            'input_text': input_text,
            'max_length': max_length
        })
        self.event_store.append_event(start_event)
        
        try:
            # Generar texto
            generated_text = self.optimizer.generate(
                input_text=input_text,
                max_length=max_length,
                temperature=0.7
            )
            
            # Evento: texto generado exitosamente
            success_event = Event('text_generated', {
                'input_text': input_text,
                'generated_text': generated_text,
                'generation_time': 0.5,  # Simulado
                'tokens_generated': len(generated_text)
            })
            self.event_store.append_event(success_event)
            
            return generated_text
            
        except Exception as e:
            # Evento: error en generación
            error_event = Event('generation_failed', {
                'input_text': input_text,
                'error': str(e)
            })
            self.event_store.append_event(error_event)
            raise
    
    def optimize_model_with_events(self, optimization_type: str, config: Dict):
        """Optimizar modelo con eventos"""
        # Evento: inicio de optimización
        start_event = Event('optimization_started', {
            'optimization_type': optimization_type,
            'config': config
        })
        self.event_store.append_event(start_event)
        
        try:
            # Aplicar optimización
            # (Implementar optimización real)
            
            # Evento: optimización completada
            success_event = Event('model_optimized', {
                'optimization_type': optimization_type,
                'config': config,
                'performance_improvement': 0.15  # Simulado
            })
            self.event_store.append_event(success_event)
            
        except Exception as e:
            # Evento: error en optimización
            error_event = Event('optimization_failed', {
                'optimization_type': optimization_type,
                'error': str(e)
            })
            self.event_store.append_event(error_event)
            raise
    
    def get_analytics(self) -> Dict[str, Any]:
        """Obtener analytics basados en eventos"""
        return self.event_store.projections
    
    def replay_events(self, from_timestamp: datetime = None) -> List[Event]:
        """Replay de eventos"""
        return self.event_store.get_events(from_timestamp=from_timestamp)

# Usar event sourcing
event_sourcing = TruthGPTEventSourcing()

# Generar texto con eventos
text = event_sourcing.generate_text_with_events("Hola, ¿cómo estás?", 100)
print(f"Texto generado: {text}")

# Obtener analytics
analytics = event_sourcing.get_analytics()
print(f"Analytics: {analytics}")
```

## ⚡ Optimización de Rendimiento

### Caché Inteligente

```python
# intelligent_cache.py
import hashlib
import time
from typing import Dict, Any, Optional
import redis
import json

class IntelligentCache:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.ttl_strategies = {
            'text_generation': 3600,  # 1 hora
            'sentiment_analysis': 1800,  # 30 minutos
            'translation': 7200,  # 2 horas
            'summarization': 3600  # 1 hora
        }
    
    def generate_cache_key(self, input_text: str, task_type: str, config: Dict) -> str:
        """Generar clave de caché"""
        # Crear hash del input y configuración
        content = f"{input_text}_{task_type}_{json.dumps(config, sort_keys=True)}"
        hash_key = hashlib.md5(content.encode()).hexdigest()
        return f"truthgpt:{task_type}:{hash_key}"
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Obtener resultado del caché"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                self.cache_stats['hits'] += 1
                return json.loads(cached_data)
            else:
                self.cache_stats['misses'] += 1
                return None
        except Exception as e:
            print(f"Error al obtener del caché: {e}")
            return None
    
    def cache_result(self, cache_key: str, result: Dict[str, Any], ttl: int = None):
        """Guardar resultado en caché"""
        try:
            if ttl is None:
                ttl = self.ttl_strategies.get(result.get('task_type', 'default'), 3600)
            
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result, ensure_ascii=False)
            )
        except Exception as e:
            print(f"Error al guardar en caché: {e}")
    
    def invalidate_cache(self, pattern: str):
        """Invalidar caché por patrón"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                self.cache_stats['evictions'] += len(keys)
        except Exception as e:
            print(f"Error al invalidar caché: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self.cache_stats['evictions']
        }
    
    def optimize_cache_strategy(self, usage_patterns: Dict[str, Any]):
        """Optimizar estrategia de caché"""
        # Ajustar TTL basado en patrones de uso
        for task_type, patterns in usage_patterns.items():
            if patterns['frequency'] > 100:  # Alta frecuencia
                self.ttl_strategies[task_type] = min(self.ttl_strategies[task_type] * 2, 7200)
            elif patterns['frequency'] < 10:  # Baja frecuencia
                self.ttl_strategies[task_type] = max(self.ttl_strategies[task_type] // 2, 300)

class CachedTruthGPT:
    def __init__(self):
        self.cache = IntelligentCache()
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
    
    def generate_text_cached(self, input_text: str, max_length: int = 100, 
                           temperature: float = 0.7, task_type: str = "text_generation") -> str:
        """Generar texto con caché"""
        # Generar clave de caché
        config = {
            'max_length': max_length,
            'temperature': temperature,
            'task_type': task_type
        }
        cache_key = self.cache.generate_cache_key(input_text, task_type, config)
        
        # Intentar obtener del caché
        cached_result = self.cache.get_cached_result(cache_key)
        if cached_result:
            return cached_result['generated_text']
        
        # Generar texto si no está en caché
        generated_text = self.optimizer.generate(
            input_text=input_text,
            max_length=max_length,
            temperature=temperature
        )
        
        # Guardar en caché
        result = {
            'generated_text': generated_text,
            'task_type': task_type,
            'timestamp': time.time()
        }
        self.cache.cache_result(cache_key, result)
        
        return generated_text
    
    def analyze_sentiment_cached(self, text: str) -> str:
        """Analizar sentimiento con caché"""
        return self.generate_text_cached(
            f"Analiza el sentimiento de: {text}",
            max_length=100,
            temperature=0.3,
            task_type="sentiment_analysis"
        )
    
    def translate_text_cached(self, text: str, target_language: str) -> str:
        """Traducir texto con caché"""
        return self.generate_text_cached(
            f"Traduce al {target_language}: {text}",
            max_length=len(text) * 2,
            temperature=0.3,
            task_type="translation"
        )

# Usar caché inteligente
cached_truthgpt = CachedTruthGPT()

# Generar texto con caché
text1 = cached_truthgpt.generate_text_cached("Hola, ¿cómo estás?", 100)
text2 = cached_truthgpt.generate_text_cached("Hola, ¿cómo estás?", 100)  # Debería usar caché

# Obtener estadísticas del caché
stats = cached_truthgpt.cache.get_cache_stats()
print(f"Estadísticas del caché: {stats}")
```

### Optimización de Batch

```python
# batch_optimization.py
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import time

class BatchOptimizer:
    def __init__(self, max_batch_size: int = 8, max_workers: int = 4):
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
    
    async def process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Procesar lote de solicitudes"""
        # Dividir en lotes más pequeños
        batches = self.split_into_batches(requests, self.max_batch_size)
        
        # Procesar lotes en paralelo
        tasks = [self.process_single_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        # Combinar resultados
        return [result for batch_results in results for result in batch_results]
    
    def split_into_batches(self, requests: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """Dividir solicitudes en lotes"""
        batches = []
        for i in range(0, len(requests), batch_size):
            batches.append(requests[i:i + batch_size])
        return batches
    
    async def process_single_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Procesar un lote individual"""
        results = []
        
        for request in batch:
            try:
                # Procesar solicitud individual
                result = await self.process_single_request(request)
                results.append(result)
            except Exception as e:
                # Manejar error
                results.append({
                    'id': request.get('id'),
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    async def process_single_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar solicitud individual"""
        # Ejecutar en thread pool para no bloquear
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._generate_text_sync,
            request
        )
        
        return {
            'id': request.get('id'),
            'result': result,
            'success': True,
            'timestamp': time.time()
        }
    
    def _generate_text_sync(self, request: Dict[str, Any]) -> str:
        """Generar texto de forma síncrona"""
        input_text = request.get('input_text', '')
        max_length = request.get('max_length', 100)
        temperature = request.get('temperature', 0.7)
        
        return self.optimizer.generate(
            input_text=input_text,
            max_length=max_length,
            temperature=temperature
        )
    
    def optimize_batch_size(self, historical_data: List[Dict[str, Any]]) -> int:
        """Optimizar tamaño de lote basado en datos históricos"""
        # Analizar rendimiento por tamaño de lote
        performance_by_batch_size = {}
        
        for data in historical_data:
            batch_size = data.get('batch_size', 1)
            processing_time = data.get('processing_time', 0)
            throughput = data.get('throughput', 0)
            
            if batch_size not in performance_by_batch_size:
                performance_by_batch_size[batch_size] = {
                    'total_time': 0,
                    'total_throughput': 0,
                    'count': 0
                }
            
            performance_by_batch_size[batch_size]['total_time'] += processing_time
            performance_by_batch_size[batch_size]['total_throughput'] += throughput
            performance_by_batch_size[batch_size]['count'] += 1
        
        # Calcular promedio y encontrar óptimo
        best_batch_size = 1
        best_throughput = 0
        
        for batch_size, stats in performance_by_batch_size.items():
            avg_throughput = stats['total_throughput'] / stats['count']
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_batch_size = batch_size
        
        return best_batch_size
    
    def adaptive_batching(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Agrupación adaptativa de solicitudes"""
        # Agrupar por similitud de input
        similarity_groups = self.group_by_similarity(requests)
        
        # Crear lotes optimizados
        optimized_batches = []
        for group in similarity_groups:
            batches = self.split_into_batches(group, self.max_batch_size)
            optimized_batches.extend(batches)
        
        return optimized_batches
    
    def group_by_similarity(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Agrupar solicitudes por similitud"""
        # Implementar agrupación por similitud
        # Por ahora, agrupar por longitud de input
        groups = {}
        for request in requests:
            input_length = len(request.get('input_text', ''))
            length_group = (input_length // 50) * 50  # Agrupar por rangos de 50 caracteres
            
            if length_group not in groups:
                groups[length_group] = []
            groups[length_group].append(request)
        
        return list(groups.values())

# Usar optimizador de lotes
batch_optimizer = BatchOptimizer(max_batch_size=4, max_workers=2)

# Crear solicitudes de prueba
requests = [
    {'id': 1, 'input_text': 'Hola, ¿cómo estás?', 'max_length': 100},
    {'id': 2, 'input_text': '¿Qué tal el clima?', 'max_length': 100},
    {'id': 3, 'input_text': 'Buenos días', 'max_length': 100},
    {'id': 4, 'input_text': '¿Cómo te encuentras?', 'max_length': 100}
]

# Procesar lote
async def process_requests():
    results = await batch_optimizer.process_batch(requests)
    return results

# Ejecutar
results = asyncio.run(process_requests())
print(f"Resultados del lote: {results}")
```

## 🧠 Modelos Personalizados

### Fine-tuning Avanzado

```python
# advanced_finetuning.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig
from typing import List, Dict, Any
import json

class AdvancedFineTuner:
    def __init__(self, base_model_name: str = "microsoft/DialoGPT-medium"):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Configurar tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_training_data(self, dataset: List[Dict[str, str]], 
                            max_length: int = 512) -> List[Dict[str, torch.Tensor]]:
        """Preparar datos de entrenamiento"""
        processed_data = []
        
        for item in dataset:
            # Tokenizar input y output
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            
            # Crear prompt completo
            full_text = f"{input_text} {output_text}"
            
            # Tokenizar
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Crear labels (mismo que input_ids para causal LM)
            labels = encoding['input_ids'].clone()
            
            # Enmascarar input para que solo se entrene en output
            input_length = len(self.tokenizer.encode(input_text))
            labels[0, :input_length] = -100
            
            processed_data.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': labels.squeeze()
            })
        
        return processed_data
    
    def create_lora_adapters(self, target_modules: List[str], 
                           rank: int = 16, alpha: float = 32.0):
        """Crear adaptadores LoRA"""
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        return self.model
    
    def train_model(self, training_data: List[Dict[str, torch.Tensor]], 
                   epochs: int = 3, learning_rate: float = 5e-5,
                   batch_size: int = 4):
        """Entrenar modelo"""
        from torch.utils.data import DataLoader
        from transformers import TrainingArguments, Trainer
        
        # Crear dataset
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = CustomDataset(training_data)
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            save_total_limit=2,
            load_best_model_at_end=False,
            report_to=None
        )
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer
        )
        
        # Entrenar
        trainer.train()
        
        return trainer
    
    def evaluate_model(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluar modelo"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for item in test_data:
                input_text = item.get('input', '')
                expected_output = item.get('output', '')
                
                # Generar respuesta
                generated = self.generate_response(input_text)
                
                # Calcular métricas
                # (Implementar métricas específicas)
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': 0.85,  # Simulado
            'bleu_score': 0.78  # Simulado
        }
    
    def generate_response(self, input_text: str, max_length: int = 100) -> str:
        """Generar respuesta"""
        self.model.eval()
        
        # Tokenizar input
        inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        # Generar
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remover input del output
        if input_text in generated_text:
            generated_text = generated_text.replace(input_text, '').strip()
        
        return generated_text
    
    def save_model(self, path: str):
        """Guardar modelo"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: str):
        """Cargar modelo"""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)

# Usar fine-tuning avanzado
fine_tuner = AdvancedFineTuner()

# Preparar datos de entrenamiento
training_data = [
    {'input': '¿Cómo estás?', 'output': 'Estoy bien, gracias por preguntar.'},
    {'input': '¿Qué tal el día?', 'output': 'Ha sido un día excelente, ¿y el tuyo?'},
    {'input': 'Hola', 'output': '¡Hola! ¿En qué puedo ayudarte?'}
]

# Procesar datos
processed_data = fine_tuner.prepare_training_data(training_data)

# Crear adaptadores LoRA
lora_model = fine_tuner.create_lora_adapters(
    target_modules=["c_attn", "c_proj"],
    rank=16,
    alpha=32.0
)

# Entrenar modelo
trainer = fine_tuner.train_model(processed_data, epochs=3)

# Generar respuesta
response = fine_tuner.generate_response("¿Cómo te encuentras?")
print(f"Respuesta: {response}")
```

### Modelos Especializados

```python
# specialized_models.py
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig
from typing import Dict, List, Any
import torch

class SpecializedModelManager:
    def __init__(self):
        self.models = {}
        self.configs = {}
        self.load_balancer = ModelLoadBalancer()
    
    def create_specialized_model(self, domain: str, config: Dict[str, Any]):
        """Crear modelo especializado"""
        # Configuración específica del dominio
        domain_config = TruthGPTConfig(
            model_name=config.get('base_model', 'microsoft/DialoGPT-medium'),
            use_mixed_precision=config.get('use_mixed_precision', True),
            temperature=config.get('temperature', 0.7),
            max_length=config.get('max_length', 200)
        )
        
        # Crear optimizador
        optimizer = ModernTruthGPTOptimizer(domain_config)
        
        # Aplicar optimizaciones específicas del dominio
        if domain == 'medical':
            optimizer = self.optimize_for_medical(optimizer)
        elif domain == 'legal':
            optimizer = self.optimize_for_legal(optimizer)
        elif domain == 'technical':
            optimizer = self.optimize_for_technical(optimizer)
        
        # Guardar modelo
        self.models[domain] = optimizer
        self.configs[domain] = config
        
        return optimizer
    
    def optimize_for_medical(self, optimizer):
        """Optimizar para dominio médico"""
        # Configuraciones específicas para medicina
        medical_config = {
            'temperature': 0.3,  # Más conservador
            'max_length': 300,
            'use_medical_terminology': True,
            'safety_checks': True
        }
        
        # Aplicar configuraciones
        # (Implementar optimizaciones específicas)
        
        return optimizer
    
    def optimize_for_legal(self, optimizer):
        """Optimizar para dominio legal"""
        legal_config = {
            'temperature': 0.2,  # Muy conservador
            'max_length': 500,
            'use_legal_terminology': True,
            'citation_required': True
        }
        
        # Aplicar configuraciones
        # (Implementar optimizaciones específicas)
        
        return optimizer
    
    def optimize_for_technical(self, optimizer):
        """Optimizar para dominio técnico"""
        technical_config = {
            'temperature': 0.4,
            'max_length': 400,
            'use_technical_terminology': True,
            'code_examples': True
        }
        
        # Aplicar configuraciones
        # (Implementar optimizaciones específicas)
        
        return optimizer
    
    def get_optimal_model(self, query: str, domain: str = None) -> Any:
        """Obtener modelo óptimo para consulta"""
        if domain:
            return self.models.get(domain)
        
        # Detectar dominio automáticamente
        detected_domain = self.detect_domain(query)
        return self.models.get(detected_domain)
    
    def detect_domain(self, query: str) -> str:
        """Detectar dominio de la consulta"""
        # Implementar detección de dominio
        medical_keywords = ['síntoma', 'diagnóstico', 'tratamiento', 'medicina']
        legal_keywords = ['ley', 'contrato', 'jurídico', 'legal']
        technical_keywords = ['código', 'programación', 'software', 'técnico']
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in medical_keywords):
            return 'medical'
        elif any(keyword in query_lower for keyword in legal_keywords):
            return 'legal'
        elif any(keyword in query_lower for keyword in technical_keywords):
            return 'technical'
        else:
            return 'general'
    
    def generate_with_specialized_model(self, query: str, domain: str = None) -> str:
        """Generar con modelo especializado"""
        model = self.get_optimal_model(query, domain)
        
        if not model:
            raise ValueError(f"No hay modelo disponible para el dominio: {domain}")
        
        return model.generate(
            input_text=query,
            max_length=200,
            temperature=0.7
        )

class ModelLoadBalancer:
    def __init__(self):
        self.model_usage = {}
        self.model_performance = {}
    
    def select_model(self, available_models: List[str], query: str) -> str:
        """Seleccionar modelo basado en carga y rendimiento"""
        # Implementar lógica de selección
        # Por ahora, seleccionar el primero disponible
        return available_models[0] if available_models else None
    
    def update_usage(self, model_name: str, processing_time: float):
        """Actualizar uso del modelo"""
        if model_name not in self.model_usage:
            self.model_usage[model_name] = {
                'total_requests': 0,
                'total_time': 0,
                'average_time': 0
            }
        
        usage = self.model_usage[model_name]
        usage['total_requests'] += 1
        usage['total_time'] += processing_time
        usage['average_time'] = usage['total_time'] / usage['total_requests']

# Usar modelos especializados
model_manager = SpecializedModelManager()

# Crear modelos especializados
medical_model = model_manager.create_specialized_model('medical', {
    'base_model': 'microsoft/DialoGPT-medium',
    'temperature': 0.3,
    'max_length': 300
})

legal_model = model_manager.create_specialized_model('legal', {
    'base_model': 'microsoft/DialoGPT-medium',
    'temperature': 0.2,
    'max_length': 500
})

# Generar con modelo especializado
medical_query = "¿Cuáles son los síntomas de la gripe?"
medical_response = model_manager.generate_with_specialized_model(medical_query, 'medical')
print(f"Respuesta médica: {medical_response}")

legal_query = "¿Qué es un contrato de trabajo?"
legal_response = model_manager.generate_with_specialized_model(legal_query, 'legal')
print(f"Respuesta legal: {legal_response}")
```

## 🔒 Seguridad y Compliance

### Autenticación y Autorización

```python
# security.py
import jwt
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class SecurityManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security_scheme = HTTPBearer()
        self.user_permissions = {}
        self.rate_limits = {}
    
    def create_access_token(self, user_id: str, permissions: List[str], 
                          expires_delta: timedelta = None) -> str:
        """Crear token de acceso"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
        
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': expire,
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verificar token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expirado")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Token inválido")
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Verificar permiso"""
        return required_permission in user_permissions
    
    def rate_limit_check(self, user_id: str, endpoint: str, limit: int = 100) -> bool:
        """Verificar límite de velocidad"""
        key = f"{user_id}:{endpoint}"
        current_time = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Limpiar requests antiguos (última hora)
        self.rate_limits[key] = [
            req_time for req_time in self.rate_limits[key]
            if current_time - req_time < 3600
        ]
        
        # Verificar límite
        if len(self.rate_limits[key]) >= limit:
            return False
        
        # Agregar request actual
        self.rate_limits[key].append(current_time)
        return True
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hashear datos sensibles"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def sanitize_input(self, input_text: str) -> str:
        """Sanitizar input"""
        # Remover caracteres peligrosos
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
        for char in dangerous_chars:
            input_text = input_text.replace(char, '')
        
        # Limitar longitud
        if len(input_text) > 10000:
            input_text = input_text[:10000]
        
        return input_text.strip()

class SecureTruthGPT:
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
        self.audit_log = []
    
    def generate_secure(self, input_text: str, user_id: str, 
                       max_length: int = 100, temperature: float = 0.7) -> str:
        """Generar texto de forma segura"""
        # Verificar límite de velocidad
        if not self.security_manager.rate_limit_check(user_id, 'generate', 50):
            raise HTTPException(status_code=429, detail="Límite de velocidad excedido")
        
        # Sanitizar input
        sanitized_input = self.security_manager.sanitize_input(input_text)
        
        # Verificar contenido sensible
        if self.contains_sensitive_data(sanitized_input):
            raise HTTPException(status_code=400, detail="Contenido sensible detectado")
        
        # Generar texto
        try:
            generated_text = self.optimizer.generate(
                input_text=sanitized_input,
                max_length=max_length,
                temperature=temperature
            )
            
            # Registrar en audit log
            self.audit_log.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'input_hash': self.security_manager.hash_sensitive_data(sanitized_input),
                'output_length': len(generated_text),
                'success': True
            })
            
            return generated_text
            
        except Exception as e:
            # Registrar error en audit log
            self.audit_log.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'input_hash': self.security_manager.hash_sensitive_data(sanitized_input),
                'error': str(e),
                'success': False
            })
            raise HTTPException(status_code=500, detail="Error en generación")
    
    def contains_sensitive_data(self, text: str) -> bool:
        """Detectar datos sensibles"""
        sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Tarjetas de crédito
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Emails
        ]
        
        import re
        for pattern in sensitive_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def get_audit_log(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Obtener log de auditoría"""
        if user_id:
            return [log for log in self.audit_log if log.get('user_id') == user_id]
        return self.audit_log

# Usar seguridad
security_manager = SecurityManager(secret_key="your-secret-key")
secure_truthgpt = SecureTruthGPT(security_manager)

# Crear token de acceso
token = security_manager.create_access_token(
    user_id="user123",
    permissions=["generate", "read"],
    expires_delta=timedelta(hours=2)
)

# Generar texto de forma segura
try:
    text = secure_truthgpt.generate_secure(
        "Hola, ¿cómo estás?",
        user_id="user123",
        max_length=100
    )
    print(f"Texto generado: {text}")
except HTTPException as e:
    print(f"Error: {e.detail}")
```

## 📊 Monitoreo y Observabilidad

### Sistema de Métricas Avanzado

```python
# advanced_monitoring.py
import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, List, Any
import json
import threading

class AdvancedMonitoring:
    def __init__(self, port: int = 9090):
        self.port = port
        self.metrics = {}
        self.alerts = []
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Métricas de Prometheus
        self.generation_counter = Counter(
            'truthgpt_generations_total',
            'Total number of text generations',
            ['model_name', 'status', 'user_id']
        )
        
        self.generation_duration = Histogram(
            'truthgpt_generation_duration_seconds',
            'Time spent on text generation',
            ['model_name', 'user_id']
        )
        
        self.memory_usage = Gauge(
            'truthgpt_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.gpu_memory_usage = Gauge(
            'truthgpt_gpu_memory_usage_bytes',
            'GPU memory usage in bytes'
        )
        
        self.active_users = Gauge(
            'truthgpt_active_users',
            'Number of active users'
        )
        
        self.error_rate = Gauge(
            'truthgpt_error_rate',
            'Error rate percentage'
        )
        
        # Iniciar servidor de métricas
        start_http_server(self.port)
        print(f"📊 Servidor de métricas iniciado en puerto {self.port}")
    
    def start_monitoring(self):
        """Iniciar monitoreo"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        print("📊 Monitoreo iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("📊 Monitoreo detenido")
    
    def _monitor_loop(self):
        """Loop de monitoreo"""
        while self.is_monitoring:
            try:
                # Actualizar métricas del sistema
                self._update_system_metrics()
                
                # Verificar alertas
                self._check_alerts()
                
                # Limpiar métricas antiguas
                self._cleanup_old_metrics()
                
                time.sleep(10)  # Monitorear cada 10 segundos
                
            except Exception as e:
                print(f"Error en monitoreo: {e}")
                time.sleep(5)
    
    def _update_system_metrics(self):
        """Actualizar métricas del sistema"""
        # Memoria RAM
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)
        
        # Memoria GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            self.gpu_memory_usage.set(gpu_memory)
        
        # Usuarios activos
        active_users = len(self.metrics.get('active_users', set()))
        self.active_users.set(active_users)
        
        # Tasa de error
        total_requests = self.metrics.get('total_requests', 0)
        total_errors = self.metrics.get('total_errors', 0)
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        self.error_rate.set(error_rate)
    
    def _check_alerts(self):
        """Verificar alertas"""
        # Alerta de memoria alta
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            self._trigger_alert('high_memory', f"Memoria alta: {memory.percent:.1f}%")
        
        # Alerta de GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_percent = (gpu_memory / gpu_total) * 100
            
            if gpu_percent > 95:
                self._trigger_alert('high_gpu_memory', f"GPU memoria alta: {gpu_percent:.1f}%")
        
        # Alerta de tasa de error
        total_requests = self.metrics.get('total_requests', 0)
        total_errors = self.metrics.get('total_errors', 0)
        if total_requests > 100:  # Solo después de 100 requests
            error_rate = (total_errors / total_requests) * 100
            if error_rate > 10:  # Más del 10% de errores
                self._trigger_alert('high_error_rate', f"Tasa de error alta: {error_rate:.1f}%")
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Disparar alerta"""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message,
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        print(f"🚨 ALERTA: {message}")
        
        # Enviar notificación (implementar según necesidad)
        self._send_notification(alert)
    
    def _send_notification(self, alert: Dict[str, Any]):
        """Enviar notificación"""
        # Implementar envío de notificaciones
        # Email, Slack, webhook, etc.
        pass
    
    def _cleanup_old_metrics(self):
        """Limpiar métricas antiguas"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hora
        
        # Limpiar alertas antiguas
        self.alerts = [
            alert for alert in self.alerts
            if alert['timestamp'] > cutoff_time
        ]
    
    def record_generation(self, model_name: str, user_id: str, 
                        duration: float, success: bool):
        """Registrar generación"""
        # Actualizar contadores
        self.generation_counter.labels(
            model_name=model_name,
            status='success' if success else 'error',
            user_id=user_id
        ).inc()
        
        if success:
            self.generation_duration.labels(
                model_name=model_name,
                user_id=user_id
            ).observe(duration)
        
        # Actualizar métricas internas
        self.metrics['total_requests'] = self.metrics.get('total_requests', 0) + 1
        if not success:
            self.metrics['total_errors'] = self.metrics.get('total_errors', 0) + 1
        
        # Actualizar usuarios activos
        if 'active_users' not in self.metrics:
            self.metrics['active_users'] = set()
        self.metrics['active_users'].add(user_id)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas"""
        return {
            'total_requests': self.metrics.get('total_requests', 0),
            'total_errors': self.metrics.get('total_errors', 0),
            'active_users': len(self.metrics.get('active_users', set())),
            'alerts_count': len(self.alerts),
            'memory_usage': psutil.virtual_memory().percent,
            'gpu_available': torch.cuda.is_available()
        }
    
    def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener alertas"""
        return self.alerts[-limit:] if self.alerts else []
    
    def clear_alerts(self):
        """Limpiar alertas"""
        self.alerts.clear()

# Usar monitoreo avanzado
monitoring = AdvancedMonitoring(port=9090)
monitoring.start_monitoring()

# Registrar generación
monitoring.record_generation(
    model_name="microsoft/DialoGPT-medium",
    user_id="user123",
    duration=1.5,
    success=True
)

# Obtener resumen
summary = monitoring.get_metrics_summary()
print(f"Resumen de métricas: {summary}")

# Obtener alertas
alerts = monitoring.get_alerts(5)
print(f"Alertas: {alerts}")
```

## 🎯 Próximos Pasos

### 1. Implementar en Producción
```python
# Configuración de producción
production_config = {
    'monitoring': True,
    'security': True,
    'caching': True,
    'load_balancing': True,
    'auto_scaling': True
}
```

### 2. Optimizar Continuamente
```python
# Sistema de optimización continua
def continuous_optimization():
    # Monitorear rendimiento
    # Ajustar parámetros
    # Optimizar modelos
    pass
```

### 3. Escalar Horizontalmente
```python
# Escalabilidad horizontal
def horizontal_scaling():
    # Distribuir carga
    # Balancear requests
    # Sincronizar estado
    pass
```

---

*¡Este tutorial avanzado te da las herramientas para implementar TruthGPT a nivel empresarial! 🚀✨*


