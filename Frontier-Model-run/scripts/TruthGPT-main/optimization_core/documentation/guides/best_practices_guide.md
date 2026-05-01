# Guía de Mejores Prácticas - TruthGPT

Esta guía te proporciona las mejores prácticas para usar TruthGPT de manera eficiente, segura y escalable.

## 📋 Tabla de Contenidos

1. [Mejores Prácticas de Desarrollo](#mejores-prácticas-de-desarrollo)
2. [Optimización de Rendimiento](#optimización-de-rendimiento)
3. [Seguridad y Privacidad](#seguridad-y-privacidad)
4. [Escalabilidad](#escalabilidad)
5. [Monitoreo y Observabilidad](#monitoreo-y-observabilidad)
6. [Mantenimiento](#mantenimiento)
7. [Troubleshooting Proactivo](#troubleshooting-proactivo)

## 🛠️ Mejores Prácticas de Desarrollo

### 1. Configuración de Entorno

```python
# config/environment.py
import os
from optimization_core import TruthGPTConfig

class EnvironmentConfig:
    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.load_config()
    
    def load_config(self):
        """Cargar configuración según el entorno"""
        if self.environment == 'development':
            self.config = self.get_development_config()
        elif self.environment == 'staging':
            self.config = self.get_staging_config()
        elif self.environment == 'production':
            self.config = self.get_production_config()
        else:
            raise ValueError(f"Entorno no válido: {self.environment}")
    
    def get_development_config(self):
        """Configuración para desarrollo"""
        return TruthGPTConfig(
            model_name="microsoft/DialoGPT-small",  # Modelo más pequeño para desarrollo
            use_mixed_precision=False,  # Desactivar para debugging
            device="cpu",  # Usar CPU para desarrollo
            batch_size=1,
            max_length=50,
            temperature=0.7,
            debug=True
        )
    
    def get_staging_config(self):
        """Configuración para staging"""
        return TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=2,
            max_length=100,
            temperature=0.7,
            debug=False
        )
    
    def get_production_config(self):
        """Configuración para producción"""
        return TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=4,
            max_length=200,
            temperature=0.7,
            debug=False,
            use_gradient_checkpointing=True,
            use_flash_attention=True
        )

# Usar configuración de entorno
env_config = EnvironmentConfig()
config = env_config.config
```

### 2. Manejo de Errores Robusto

```python
# utils/error_handling.py
import logging
from typing import Optional, Dict, Any
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class TruthGPTErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
        self.circuit_breaker_threshold = 5
    
    def handle_generation_error(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:
        """Manejar errores de generación"""
        error_type = type(error).__name__
        
        # Incrementar contador de errores
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log del error
        self.logger.error(f"Error en generación: {error}", extra=context)
        
        # Verificar circuit breaker
        if self.error_counts[error_type] >= self.circuit_breaker_threshold:
            self.logger.critical(f"Circuit breaker activado para {error_type}")
            return None
        
        # Manejar errores específicos
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return self.handle_oom_error()
        elif isinstance(error, RuntimeError):
            return self.handle_runtime_error(error)
        else:
            return self.handle_generic_error(error)
    
    def handle_oom_error(self) -> str:
        """Manejar error de memoria GPU"""
        return "Lo siento, el sistema está experimentando problemas de memoria. Por favor, intenta con un texto más corto."
    
    def handle_runtime_error(self, error: RuntimeError) -> str:
        """Manejar error de runtime"""
        if "CUDA" in str(error):
            return "Error de GPU detectado. Cambiando a CPU..."
        return "Error interno del sistema. Por favor, intenta nuevamente."
    
    def handle_generic_error(self, error: Exception) -> str:
        """Manejar error genérico"""
        return "Ha ocurrido un error inesperado. Por favor, contacta al soporte técnico."
    
    def reset_error_counts(self):
        """Resetear contadores de error"""
        self.error_counts.clear()
        self.logger.info("Contadores de error reseteados")

class RobustTruthGPT:
    def __init__(self, config: TruthGPTConfig):
        self.config = config
        self.optimizer = ModernTruthGPTOptimizer(config)
        self.error_handler = TruthGPTErrorHandler()
        self.fallback_responses = [
            "Lo siento, no puedo procesar tu solicitud en este momento.",
            "Estoy experimentando dificultades técnicas. Por favor, intenta más tarde.",
            "El sistema está temporalmente no disponible."
        ]
    
    def generate_safe(self, input_text: str, max_length: int = 100, 
                     temperature: float = 0.7) -> str:
        """Generar texto de forma segura"""
        context = {
            'input_text': input_text,
            'max_length': max_length,
            'temperature': temperature,
            'user_id': 'anonymous'
        }
        
        try:
            # Validar input
            if not self.validate_input(input_text):
                return "Input no válido. Por favor, proporciona un texto válido."
            
            # Generar texto
            result = self.optimizer.generate(
                input_text=input_text,
                max_length=max_length,
                temperature=temperature
            )
            
            # Validar output
            if not self.validate_output(result):
                return "Error en la generación. Por favor, intenta nuevamente."
            
            return result
            
        except Exception as e:
            # Manejar error
            error_response = self.error_handler.handle_generation_error(e, context)
            
            if error_response:
                return error_response
            else:
                # Usar respuesta de fallback
                import random
                return random.choice(self.fallback_responses)
    
    def validate_input(self, input_text: str) -> bool:
        """Validar input"""
        if not input_text or len(input_text.strip()) == 0:
            return False
        
        if len(input_text) > 10000:  # Límite de caracteres
            return False
        
        # Verificar caracteres peligrosos
        dangerous_patterns = ['<script>', 'javascript:', 'data:']
        for pattern in dangerous_patterns:
            if pattern in input_text.lower():
                return False
        
        return True
    
    def validate_output(self, output: str) -> bool:
        """Validar output"""
        if not output or len(output.strip()) == 0:
            return False
        
        if len(output) > 5000:  # Límite de caracteres
            return False
        
        return True

# Usar TruthGPT robusto
config = TruthGPTConfig(model_name="microsoft/DialoGPT-medium")
robust_truthgpt = RobustTruthGPT(config)

# Generar de forma segura
result = robust_truthgpt.generate_safe("Hola, ¿cómo estás?", max_length=100)
print(f"Resultado: {result}")
```

### 3. Logging y Auditoría

```python
# utils/logging.py
import logging
import json
from datetime import datetime
from typing import Dict, Any
import hashlib

class TruthGPTLogger:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('truthgpt')
        self.logger.setLevel(log_level)
        
        # Configurar formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler para archivo
        file_handler = logging.FileHandler('truthgpt.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_generation(self, input_text: str, output_text: str, 
                     user_id: str, duration: float, success: bool):
        """Log de generación"""
        # Hash del input para privacidad
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'generation',
            'user_id': user_id,
            'input_hash': input_hash,
            'output_length': len(output_text),
            'duration': duration,
            'success': success
        }
        
        if success:
            self.logger.info(f"Generación exitosa: {json.dumps(log_data)}")
        else:
            self.logger.error(f"Generación fallida: {json.dumps(log_data)}")
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log de error"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        }
        
        self.logger.error(f"Error: {json.dumps(log_data)}")
    
    def log_performance(self, metrics: Dict[str, Any]):
        """Log de rendimiento"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'performance',
            'metrics': metrics
        }
        
        self.logger.info(f"Rendimiento: {json.dumps(log_data)}")
    
    def log_security(self, event: str, details: Dict[str, Any]):
        """Log de seguridad"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'security',
            'event': event,
            'details': details
        }
        
        self.logger.warning(f"Seguridad: {json.dumps(log_data)}")

# Usar logger
logger = TruthGPTLogger()

# Log de generación
logger.log_generation(
    input_text="Hola, ¿cómo estás?",
    output_text="Hola, estoy bien, gracias por preguntar.",
    user_id="user123",
    duration=1.5,
    success=True
)
```

## ⚡ Optimización de Rendimiento

### 1. Caché Inteligente

```python
# cache/intelligent_cache.py
import time
import hashlib
import json
from typing import Dict, Any, Optional
import redis
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

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
    
    def generate_cache_key(self, input_text: str, task_type: str, 
                          config: Dict[str, Any]) -> str:
        """Generar clave de caché"""
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
    
    def cache_result(self, cache_key: str, result: Dict[str, Any], 
                    ttl: int = None):
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
        for task_type, patterns in usage_patterns.items():
            if patterns['frequency'] > 100:  # Alta frecuencia
                self.ttl_strategies[task_type] = min(self.ttl_strategies[task_type] * 2, 7200)
            elif patterns['frequency'] < 10:  # Baja frecuencia
                self.ttl_strategies[task_type] = max(self.ttl_strategies[task_type] // 2, 300)

class CachedTruthGPT:
    def __init__(self, config: TruthGPTConfig):
        self.config = config
        self.optimizer = ModernTruthGPTOptimizer(config)
        self.cache = IntelligentCache()
        self.logger = TruthGPTLogger()
    
    def generate_cached(self, input_text: str, max_length: int = 100,
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
            self.logger.logger.info(f"Cache hit para: {cache_key[:20]}...")
            return cached_result['generated_text']
        
        # Generar texto si no está en caché
        start_time = time.time()
        generated_text = self.optimizer.generate(
            input_text=input_text,
            max_length=max_length,
            temperature=temperature
        )
        duration = time.time() - start_time
        
        # Guardar en caché
        result = {
            'generated_text': generated_text,
            'task_type': task_type,
            'timestamp': time.time(),
            'duration': duration
        }
        self.cache.cache_result(cache_key, result)
        
        # Log de generación
        self.logger.log_generation(
            input_text=input_text,
            output_text=generated_text,
            user_id='cached_user',
            duration=duration,
            success=True
        )
        
        return generated_text

# Usar caché inteligente
config = TruthGPTConfig(model_name="microsoft/DialoGPT-medium")
cached_truthgpt = CachedTruthGPT(config)

# Generar con caché
text1 = cached_truthgpt.generate_cached("Hola, ¿cómo estás?", 100)
text2 = cached_truthgpt.generate_cached("Hola, ¿cómo estás?", 100)  # Debería usar caché

# Obtener estadísticas del caché
stats = cached_truthgpt.cache.get_cache_stats()
print(f"Estadísticas del caché: {stats}")
```

### 2. Optimización de Batch

```python
# optimization/batch_optimizer.py
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
    
    def split_into_batches(self, requests: List[Dict[str, Any]], 
                          batch_size: int) -> List[List[Dict[str, Any]]]:
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

## 🔒 Seguridad y Privacidad

### 1. Sanitización de Input

```python
# security/input_sanitizer.py
import re
import html
from typing import str

class InputSanitizer:
    def __init__(self):
        self.dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'onclick=',
            r'<iframe.*?>',
            r'<object.*?>',
            r'<embed.*?>'
        ]
        self.max_length = 10000
        self.allowed_chars = re.compile(r'[a-zA-Z0-9\s\.,!?;:()\-_\'"]')
    
    def sanitize_input(self, input_text: str) -> str:
        """Sanitizar input de texto"""
        if not input_text:
            return ""
        
        # Limitar longitud
        if len(input_text) > self.max_length:
            input_text = input_text[:self.max_length]
        
        # Escapar HTML
        input_text = html.escape(input_text)
        
        # Remover patrones peligrosos
        for pattern in self.dangerous_patterns:
            input_text = re.sub(pattern, '', input_text, flags=re.IGNORECASE)
        
        # Remover caracteres no permitidos
        input_text = ''.join(char for char in input_text if self.allowed_chars.match(char))
        
        # Limpiar espacios múltiples
        input_text = re.sub(r'\s+', ' ', input_text).strip()
        
        return input_text
    
    def validate_input(self, input_text: str) -> bool:
        """Validar input"""
        if not input_text or len(input_text.strip()) == 0:
            return False
        
        # Verificar patrones peligrosos
        for pattern in self.dangerous_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                return False
        
        return True

# Usar sanitizador
sanitizer = InputSanitizer()

# Sanitizar input
dirty_input = "<script>alert('xss')</script>Hola, ¿cómo estás?"
clean_input = sanitizer.sanitize_input(dirty_input)
print(f"Input limpio: {clean_input}")

# Validar input
is_valid = sanitizer.validate_input(clean_input)
print(f"Input válido: {is_valid}")
```

### 2. Rate Limiting

```python
# security/rate_limiter.py
import time
from typing import Dict, List
from collections import defaultdict, deque

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(deque)
        self.limits = {
            'per_minute': 60,
            'per_hour': 1000,
            'per_day': 10000
        }
    
    def is_allowed(self, user_id: str, endpoint: str = 'default') -> bool:
        """Verificar si la solicitud está permitida"""
        key = f"{user_id}:{endpoint}"
        current_time = time.time()
        
        # Limpiar requests antiguos
        self._cleanup_old_requests(key, current_time)
        
        # Verificar límites
        if not self._check_limits(key, current_time):
            return False
        
        # Agregar request actual
        self.requests[key].append(current_time)
        return True
    
    def _cleanup_old_requests(self, key: str, current_time: float):
        """Limpiar requests antiguos"""
        if key in self.requests:
            # Limpiar requests de más de 1 día
            cutoff_time = current_time - 86400  # 24 horas
            while self.requests[key] and self.requests[key][0] < cutoff_time:
                self.requests[key].popleft()
    
    def _check_limits(self, key: str, current_time: float) -> bool:
        """Verificar límites de velocidad"""
        if key not in self.requests:
            return True
        
        requests = self.requests[key]
        
        # Verificar límite por minuto
        minute_ago = current_time - 60
        recent_requests = [req for req in requests if req > minute_ago]
        if len(recent_requests) >= self.limits['per_minute']:
            return False
        
        # Verificar límite por hora
        hour_ago = current_time - 3600
        hour_requests = [req for req in requests if req > hour_ago]
        if len(hour_requests) >= self.limits['per_hour']:
            return False
        
        # Verificar límite por día
        day_ago = current_time - 86400
        day_requests = [req for req in requests if req > day_ago]
        if len(day_requests) >= self.limits['per_day']:
            return False
        
        return True
    
    def get_remaining_requests(self, user_id: str, endpoint: str = 'default') -> Dict[str, int]:
        """Obtener requests restantes"""
        key = f"{user_id}:{endpoint}"
        current_time = time.time()
        
        if key not in self.requests:
            return {
                'per_minute': self.limits['per_minute'],
                'per_hour': self.limits['per_hour'],
                'per_day': self.limits['per_day']
            }
        
        requests = self.requests[key]
        
        # Calcular requests restantes
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        minute_requests = len([req for req in requests if req > minute_ago])
        hour_requests = len([req for req in requests if req > hour_ago])
        day_requests = len([req for req in requests if req > day_ago])
        
        return {
            'per_minute': max(0, self.limits['per_minute'] - minute_requests),
            'per_hour': max(0, self.limits['per_hour'] - hour_requests),
            'per_day': max(0, self.limits['per_day'] - day_requests)
        }

# Usar rate limiter
rate_limiter = RateLimiter()

# Verificar si está permitido
user_id = "user123"
is_allowed = rate_limiter.is_allowed(user_id, "generate")
print(f"Request permitido: {is_allowed}")

# Obtener requests restantes
remaining = rate_limiter.get_remaining_requests(user_id, "generate")
print(f"Requests restantes: {remaining}")
```

## 📊 Monitoreo y Observabilidad

### 1. Métricas Avanzadas

```python
# monitoring/advanced_metrics.py
import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, List, Any
import json
import threading

class AdvancedMetrics:
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

# Usar métricas avanzadas
metrics = AdvancedMetrics(port=9090)
metrics.start_monitoring()

# Registrar generación
metrics.record_generation(
    model_name="microsoft/DialoGPT-medium",
    user_id="user123",
    duration=1.5,
    success=True
)

# Obtener resumen
summary = metrics.get_metrics_summary()
print(f"Resumen de métricas: {summary}")

# Obtener alertas
alerts = metrics.get_alerts(5)
print(f"Alertas: {alerts}")
```

## 🔧 Mantenimiento

### 1. Limpieza Automática

```python
# maintenance/auto_cleanup.py
import schedule
import time
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List

class AutoCleanup:
    def __init__(self):
        self.cleanup_tasks = []
        self.log_files = []
        self.cache_files = []
        self.temp_files = []
    
    def schedule_cleanup(self):
        """Programar tareas de limpieza"""
        # Limpiar logs diariamente a las 2 AM
        schedule.every().day.at("02:00").do(self.cleanup_logs)
        
        # Limpiar caché cada 6 horas
        schedule.every(6).hours.do(self.cleanup_cache)
        
        # Limpiar archivos temporales cada hora
        schedule.every().hour.do(self.cleanup_temp_files)
        
        # Limpiar métricas antiguas cada día
        schedule.every().day.at("03:00").do(self.cleanup_old_metrics)
        
        print("📅 Tareas de limpieza programadas")
    
    def cleanup_logs(self):
        """Limpiar logs antiguos"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            return
        
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    os.remove(file_path)
                    print(f"🗑️  Log eliminado: {filename}")
    
    def cleanup_cache(self):
        """Limpiar caché"""
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            return
        
        cutoff_date = datetime.now() - timedelta(hours=24)
        
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    os.remove(file_path)
                    print(f"🗑️  Cache eliminado: {filename}")
    
    def cleanup_temp_files(self):
        """Limpiar archivos temporales"""
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            return
        
        cutoff_date = datetime.now() - timedelta(hours=1)
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    os.remove(file_path)
                    print(f"🗑️  Archivo temporal eliminado: {filename}")
    
    def cleanup_old_metrics(self):
        """Limpiar métricas antiguas"""
        metrics_file = "metrics.json"
        if not os.path.exists(metrics_file):
            return
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Limpiar métricas de más de 30 días
            cutoff_date = datetime.now() - timedelta(days=30)
            
            cleaned_metrics = []
            for metric in metrics:
                metric_date = datetime.fromisoformat(metric.get('timestamp', ''))
                if metric_date > cutoff_date:
                    cleaned_metrics.append(metric)
            
            with open(metrics_file, 'w') as f:
                json.dump(cleaned_metrics, f)
            
            print(f"📊 Métricas limpiadas: {len(metrics) - len(cleaned_metrics)} entradas eliminadas")
            
        except Exception as e:
            print(f"Error al limpiar métricas: {e}")
    
    def run_cleanup(self):
        """Ejecutar limpieza"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Verificar cada minuto

# Usar limpieza automática
cleanup = AutoCleanup()
cleanup.schedule_cleanup()

# Ejecutar limpieza
cleanup.run_cleanup()
```

### 2. Backup y Recuperación

```python
# maintenance/backup_recovery.py
import os
import shutil
import json
import zipfile
from datetime import datetime
from typing import Dict, List

class BackupRecovery:
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = backup_dir
        self.ensure_backup_dir()
    
    def ensure_backup_dir(self):
        """Asegurar que existe el directorio de backup"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
    
    def create_backup(self, source_paths: List[str], backup_name: str = None):
        """Crear backup"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = os.path.join(self.backup_dir, f"{backup_name}.zip")
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for source_path in source_paths:
                if os.path.exists(source_path):
                    if os.path.isfile(source_path):
                        zipf.write(source_path, os.path.basename(source_path))
                    elif os.path.isdir(source_path):
                        for root, dirs, files in os.walk(source_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arc_path = os.path.relpath(file_path, source_path)
                                zipf.write(file_path, arc_path)
        
        print(f"💾 Backup creado: {backup_path}")
        return backup_path
    
    def restore_backup(self, backup_path: str, target_dir: str):
        """Restaurar backup"""
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup no encontrado: {backup_path}")
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        with zipfile.ZipFile(backup_path, 'r') as zipf:
            zipf.extractall(target_dir)
        
        print(f"🔄 Backup restaurado: {backup_path} -> {target_dir}")
    
    def list_backups(self) -> List[Dict[str, str]]:
        """Listar backups disponibles"""
        backups = []
        
        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.zip'):
                file_path = os.path.join(self.backup_dir, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                file_size = os.path.getsize(file_path)
                
                backups.append({
                    'name': filename,
                    'path': file_path,
                    'created': file_time.isoformat(),
                    'size': file_size
                })
        
        return sorted(backups, key=lambda x: x['created'], reverse=True)
    
    def cleanup_old_backups(self, keep_days: int = 30):
        """Limpiar backups antiguos"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.zip'):
                file_path = os.path.join(self.backup_dir, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time < cutoff_date:
                    os.remove(file_path)
                    print(f"🗑️  Backup antiguo eliminado: {filename}")

# Usar backup y recuperación
backup_recovery = BackupRecovery()

# Crear backup
source_paths = ["models", "cache", "logs"]
backup_path = backup_recovery.create_backup(source_paths)

# Listar backups
backups = backup_recovery.list_backups()
print(f"Backups disponibles: {backups}")

# Limpiar backups antiguos
backup_recovery.cleanup_old_backups(keep_days=30)
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
    'auto_scaling': True,
    'backup': True,
    'cleanup': True
}
```

### 2. Optimizar Continuamente
```python
# Sistema de optimización continua
def continuous_optimization():
    # Monitorear rendimiento
    # Ajustar parámetros
    # Optimizar modelos
    # Limpiar recursos
    pass
```

### 3. Escalar Horizontalmente
```python
# Escalabilidad horizontal
def horizontal_scaling():
    # Distribuir carga
    # Balancear requests
    # Sincronizar estado
    # Replicar datos
    pass
```

---

*¡Con estas mejores prácticas tienes todo lo necesario para usar TruthGPT de manera profesional! 🚀✨*


