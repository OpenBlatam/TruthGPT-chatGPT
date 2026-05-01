# Ejemplos Empresariales - TruthGPT

Esta sección contiene ejemplos específicos para implementar TruthGPT en entornos empresariales de gran escala.

## 📋 Tabla de Contenidos

1. [Arquitectura Empresarial](#arquitectura-empresarial)
2. [Sistemas de Alta Disponibilidad](#sistemas-de-alta-disponibilidad)
3. [Integración con ERP/CRM](#integración-con-erpcrm)
4. [Compliance y Auditoría](#compliance-y-auditoría)
5. [Multi-tenant](#multi-tenant)
6. [Disaster Recovery](#disaster-recovery)
7. [Performance a Escala](#performance-a-escala)

## 🏢 Arquitectura Empresarial

### Ejemplo 1: Sistema de Microservicios Empresarial

```python
# enterprise/microservices_architecture.py
from fastapi import FastAPI, HTTPException, Depends
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig
import asyncio
import redis
import json
from typing import Dict, List, Optional
import uuid
from datetime import datetime, timedelta

class EnterpriseTruthGPT:
    def __init__(self):
        self.app = FastAPI(
            title="TruthGPT Enterprise API",
            description="API empresarial para TruthGPT",
            version="2.0.0"
        )
        
        # Configuración de microservicios
        self.services = {
            'text_generation': TextGenerationService(),
            'sentiment_analysis': SentimentAnalysisService(),
            'translation': TranslationService(),
            'summarization': SummarizationService(),
            'classification': ClassificationService()
        }
        
        # Load balancer
        self.load_balancer = EnterpriseLoadBalancer()
        
        # Circuit breaker
        self.circuit_breaker = EnterpriseCircuitBreaker()
        
        # Rate limiter
        self.rate_limiter = EnterpriseRateLimiter()
        
        # Audit logger
        self.audit_logger = EnterpriseAuditLogger()
        
        # Health checker
        self.health_checker = EnterpriseHealthChecker()
        
        self.setup_routes()
    
    def setup_routes(self):
        """Configurar rutas de la API"""
        
        @self.app.get("/health")
        async def health_check():
            return await self.health_checker.check_all_services()
        
        @self.app.post("/generate")
        async def generate_text(request: Dict, user_id: str = Depends(self.get_user_id)):
            return await self.process_request('text_generation', request, user_id)
        
        @self.app.post("/analyze-sentiment")
        async def analyze_sentiment(request: Dict, user_id: str = Depends(self.get_user_id)):
            return await self.process_request('sentiment_analysis', request, user_id)
        
        @self.app.post("/translate")
        async def translate_text(request: Dict, user_id: str = Depends(self.get_user_id)):
            return await self.process_request('translation', request, user_id)
        
        @self.app.post("/summarize")
        async def summarize_text(request: Dict, user_id: str = Depends(self.get_user_id)):
            return await self.process_request('summarization', request, user_id)
        
        @self.app.post("/classify")
        async def classify_text(request: Dict, user_id: str = Depends(self.get_user_id)):
            return await self.process_request('classification', request, user_id)
    
    async def process_request(self, service_type: str, request: Dict, user_id: str):
        """Procesar solicitud empresarial"""
        request_id = str(uuid.uuid4())
        
        try:
            # Verificar rate limiting
            if not await self.rate_limiter.is_allowed(user_id, service_type):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Verificar circuit breaker
            if self.circuit_breaker.is_open(service_type):
                raise HTTPException(status_code=503, detail="Service temporarily unavailable")
            
            # Obtener servicio
            service = self.services[service_type]
            
            # Procesar solicitud
            start_time = datetime.utcnow()
            result = await service.process(request)
            end_time = datetime.utcnow()
            
            # Registrar en audit log
            await self.audit_logger.log_request(
                request_id=request_id,
                user_id=user_id,
                service_type=service_type,
                request=request,
                response=result,
                duration=(end_time - start_time).total_seconds(),
                success=True
            )
            
            return result
            
        except Exception as e:
            # Registrar error en audit log
            await self.audit_logger.log_error(
                request_id=request_id,
                user_id=user_id,
                service_type=service_type,
                error=str(e),
                request=request
            )
            
            # Registrar fallo en circuit breaker
            self.circuit_breaker.record_failure(service_type)
            
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_user_id(self, authorization: str = None) -> str:
        """Obtener ID de usuario desde token"""
        # Implementar autenticación JWT
        # Por ahora, retornar usuario por defecto
        return "enterprise_user"

class TextGenerationService:
    def __init__(self):
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
    
    async def process(self, request: Dict) -> Dict:
        """Procesar solicitud de generación de texto"""
        input_text = request.get('text', '')
        max_length = request.get('max_length', 100)
        temperature = request.get('temperature', 0.7)
        
        generated_text = self.optimizer.generate(
            input_text=input_text,
            max_length=max_length,
            temperature=temperature
        )
        
        return {
            'generated_text': generated_text,
            'input_text': input_text,
            'parameters': {
                'max_length': max_length,
                'temperature': temperature
            }
        }

class SentimentAnalysisService:
    def __init__(self):
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
    
    async def process(self, request: Dict) -> Dict:
        """Procesar solicitud de análisis de sentimientos"""
        text = request.get('text', '')
        
        prompt = f"Analiza el sentimiento de este texto: {text}"
        analysis = self.optimizer.generate(
            input_text=prompt,
            max_length=100,
            temperature=0.3
        )
        
        return {
            'text': text,
            'sentiment_analysis': analysis,
            'confidence': 0.85  # Simulado
        }

class TranslationService:
    def __init__(self):
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
    
    async def process(self, request: Dict) -> Dict:
        """Procesar solicitud de traducción"""
        text = request.get('text', '')
        target_language = request.get('target_language', 'español')
        
        prompt = f"Traduce al {target_language}: {text}"
        translation = self.optimizer.generate(
            input_text=prompt,
            max_length=len(text) * 2,
            temperature=0.3
        )
        
        return {
            'original_text': text,
            'translated_text': translation,
            'target_language': target_language
        }

class SummarizationService:
    def __init__(self):
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
    
    async def process(self, request: Dict) -> Dict:
        """Procesar solicitud de resumen"""
        text = request.get('text', '')
        max_length = request.get('max_length', 100)
        
        prompt = f"Resume este texto: {text}"
        summary = self.optimizer.generate(
            input_text=prompt,
            max_length=max_length,
            temperature=0.3
        )
        
        return {
            'original_text': text,
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary)
        }

class ClassificationService:
    def __init__(self):
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
    
    async def process(self, request: Dict) -> Dict:
        """Procesar solicitud de clasificación"""
        text = request.get('text', '')
        categories = request.get('categories', ['positivo', 'negativo', 'neutral'])
        
        prompt = f"Clasifica este texto en una de estas categorías {categories}: {text}"
        classification = self.optimizer.generate(
            input_text=prompt,
            max_length=50,
            temperature=0.3
        )
        
        return {
            'text': text,
            'classification': classification,
            'categories': categories,
            'confidence': 0.90  # Simulado
        }

class EnterpriseLoadBalancer:
    def __init__(self):
        self.service_instances = {}
        self.load_metrics = {}
    
    def get_optimal_instance(self, service_type: str) -> str:
        """Obtener instancia óptima del servicio"""
        if service_type not in self.service_instances:
            return f"{service_type}_primary"
        
        instances = self.service_instances[service_type]
        # Seleccionar instancia con menor carga
        optimal_instance = min(instances, key=lambda x: self.load_metrics.get(x, 0))
        return optimal_instance
    
    def update_load_metrics(self, instance: str, load: float):
        """Actualizar métricas de carga"""
        self.load_metrics[instance] = load

class EnterpriseCircuitBreaker:
    def __init__(self):
        self.states = {}
        self.failure_threshold = 5
        self.timeout = 60
    
    def is_open(self, service_type: str) -> bool:
        """Verificar si circuit breaker está abierto"""
        if service_type not in self.states:
            return False
        
        state = self.states[service_type]
        if state['state'] == 'open':
            # Verificar si ha pasado el timeout
            if datetime.utcnow().timestamp() - state['last_failure'] > self.timeout:
                state['state'] = 'half_open'
                return False
            return True
        
        return False
    
    def record_failure(self, service_type: str):
        """Registrar fallo"""
        if service_type not in self.states:
            self.states[service_type] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': 0
            }
        
        state = self.states[service_type]
        state['failure_count'] += 1
        state['last_failure'] = datetime.utcnow().timestamp()
        
        if state['failure_count'] >= self.failure_threshold:
            state['state'] = 'open'
    
    def record_success(self, service_type: str):
        """Registrar éxito"""
        if service_type in self.states:
            self.states[service_type]['state'] = 'closed'
            self.states[service_type]['failure_count'] = 0

class EnterpriseRateLimiter:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.limits = {
            'text_generation': {'per_minute': 60, 'per_hour': 1000},
            'sentiment_analysis': {'per_minute': 120, 'per_hour': 2000},
            'translation': {'per_minute': 80, 'per_hour': 1500},
            'summarization': {'per_minute': 100, 'per_hour': 1800},
            'classification': {'per_minute': 150, 'per_hour': 2500}
        }
    
    async def is_allowed(self, user_id: str, service_type: str) -> bool:
        """Verificar si la solicitud está permitida"""
        if service_type not in self.limits:
            return True
        
        limits = self.limits[service_type]
        current_time = datetime.utcnow()
        
        # Verificar límite por minuto
        minute_key = f"rate_limit:{user_id}:{service_type}:minute:{current_time.minute}"
        minute_count = self.redis_client.get(minute_key)
        if minute_count and int(minute_count) >= limits['per_minute']:
            return False
        
        # Verificar límite por hora
        hour_key = f"rate_limit:{user_id}:{service_type}:hour:{current_time.hour}"
        hour_count = self.redis_client.get(hour_key)
        if hour_count and int(hour_count) >= limits['per_hour']:
            return False
        
        # Incrementar contadores
        self.redis_client.incr(minute_key)
        self.redis_client.expire(minute_key, 60)
        self.redis_client.incr(hour_key)
        self.redis_client.expire(hour_key, 3600)
        
        return True

class EnterpriseAuditLogger:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    async def log_request(self, request_id: str, user_id: str, service_type: str,
                        request: Dict, response: Dict, duration: float, success: bool):
        """Registrar solicitud en audit log"""
        audit_entry = {
            'request_id': request_id,
            'user_id': user_id,
            'service_type': service_type,
            'timestamp': datetime.utcnow().isoformat(),
            'request': request,
            'response': response,
            'duration': duration,
            'success': success
        }
        
        # Guardar en Redis
        self.redis_client.setex(
            f"audit:{request_id}",
            86400,  # 24 horas
            json.dumps(audit_entry, ensure_ascii=False)
        )
    
    async def log_error(self, request_id: str, user_id: str, service_type: str,
                       error: str, request: Dict):
        """Registrar error en audit log"""
        error_entry = {
            'request_id': request_id,
            'user_id': user_id,
            'service_type': service_type,
            'timestamp': datetime.utcnow().isoformat(),
            'error': error,
            'request': request,
            'success': False
        }
        
        # Guardar en Redis
        self.redis_client.setex(
            f"audit:{request_id}",
            86400,  # 24 horas
            json.dumps(error_entry, ensure_ascii=False)
        )

class EnterpriseHealthChecker:
    def __init__(self):
        self.services = {}
    
    async def check_all_services(self) -> Dict:
        """Verificar salud de todos los servicios"""
        health_status = {
            'overall': 'healthy',
            'services': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for service_name, service in self.services.items():
            try:
                # Verificar salud del servicio
                service_health = await self.check_service_health(service_name)
                health_status['services'][service_name] = service_health
                
                if service_health['status'] != 'healthy':
                    health_status['overall'] = 'degraded'
                    
            except Exception as e:
                health_status['services'][service_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['overall'] = 'unhealthy'
        
        return health_status
    
    async def check_service_health(self, service_name: str) -> Dict:
        """Verificar salud de un servicio específico"""
        # Implementar verificación de salud específica
        return {
            'status': 'healthy',
            'response_time': 0.1,
            'memory_usage': 0.5,
            'cpu_usage': 0.3
        }

# Usar arquitectura empresarial
enterprise_truthgpt = EnterpriseTruthGPT()
app = enterprise_truthgpt.app
```

### Ejemplo 2: Sistema de Alta Disponibilidad

```python
# enterprise/high_availability.py
import asyncio
import time
from typing import Dict, List, Optional
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class HighAvailabilityTruthGPT:
    def __init__(self):
        self.primary_instance = None
        self.secondary_instances = []
        self.health_checker = HealthChecker()
        self.failover_manager = FailoverManager()
        self.load_balancer = LoadBalancer()
        
        self.initialize_instances()
    
    def initialize_instances(self):
        """Inicializar instancias"""
        # Instancia primaria
        self.primary_instance = TruthGPTInstance(
            name="primary",
            config=TruthGPTConfig(
                model_name="microsoft/DialoGPT-medium",
                use_mixed_precision=True
            )
        )
        
        # Instancias secundarias
        for i in range(3):
            secondary = TruthGPTInstance(
                name=f"secondary_{i+1}",
                config=TruthGPTConfig(
                    model_name="microsoft/DialoGPT-medium",
                    use_mixed_precision=True
                )
            )
            self.secondary_instances.append(secondary)
    
    async def generate_text(self, input_text: str, max_length: int = 100) -> str:
        """Generar texto con alta disponibilidad"""
        # Obtener instancia disponible
        instance = await self.get_available_instance()
        
        if not instance:
            raise Exception("No hay instancias disponibles")
        
        try:
            # Generar texto
            result = await instance.generate_text(input_text, max_length)
            
            # Registrar éxito
            await self.health_checker.record_success(instance.name)
            
            return result
            
        except Exception as e:
            # Registrar fallo
            await self.health_checker.record_failure(instance.name)
            
            # Intentar con otra instancia
            return await self.generate_text_fallback(input_text, max_length)
    
    async def get_available_instance(self) -> Optional['TruthGPTInstance']:
        """Obtener instancia disponible"""
        # Verificar instancia primaria
        if await self.health_checker.is_healthy(self.primary_instance.name):
            return self.primary_instance
        
        # Verificar instancias secundarias
        for secondary in self.secondary_instances:
            if await self.health_checker.is_healthy(secondary.name):
                return secondary
        
        return None
    
    async def generate_text_fallback(self, input_text: str, max_length: int) -> str:
        """Generar texto con fallback"""
        # Intentar con todas las instancias
        for instance in [self.primary_instance] + self.secondary_instances:
            try:
                if await self.health_checker.is_healthy(instance.name):
                    result = await instance.generate_text(input_text, max_length)
                    return result
            except Exception:
                continue
        
        # Si todas fallan, usar respuesta de fallback
        return "Lo siento, el sistema está temporalmente no disponible. Por favor, intenta más tarde."

class TruthGPTInstance:
    def __init__(self, name: str, config: TruthGPTConfig):
        self.name = name
        self.config = config
        self.optimizer = ModernTruthGPTOptimizer(config)
        self.is_initialized = False
    
    async def initialize(self):
        """Inicializar instancia"""
        try:
            # Cargar modelo
            self.optimizer = ModernTruthGPTOptimizer(self.config)
            self.is_initialized = True
            print(f"✅ Instancia {self.name} inicializada")
        except Exception as e:
            print(f"❌ Error al inicializar {self.name}: {e}")
            self.is_initialized = False
    
    async def generate_text(self, input_text: str, max_length: int) -> str:
        """Generar texto"""
        if not self.is_initialized:
            await self.initialize()
        
        if not self.is_initialized:
            raise Exception(f"Instancia {self.name} no está inicializada")
        
        return self.optimizer.generate(
            input_text=input_text,
            max_length=max_length,
            temperature=0.7
        )

class HealthChecker:
    def __init__(self):
        self.health_status = {}
        self.failure_counts = {}
        self.success_counts = {}
    
    async def is_healthy(self, instance_name: str) -> bool:
        """Verificar si la instancia está saludable"""
        if instance_name not in self.health_status:
            return True
        
        status = self.health_status[instance_name]
        return status['is_healthy'] and status['last_check'] > time.time() - 60
    
    async def record_success(self, instance_name: str):
        """Registrar éxito"""
        self.success_counts[instance_name] = self.success_counts.get(instance_name, 0) + 1
        self.health_status[instance_name] = {
            'is_healthy': True,
            'last_check': time.time()
        }
    
    async def record_failure(self, instance_name: str):
        """Registrar fallo"""
        self.failure_counts[instance_name] = self.failure_counts.get(instance_name, 0) + 1
        
        # Marcar como no saludable si hay muchos fallos
        if self.failure_counts[instance_name] > 3:
            self.health_status[instance_name] = {
                'is_healthy': False,
                'last_check': time.time()
            }

class FailoverManager:
    def __init__(self):
        self.failover_threshold = 3
        self.failover_timeout = 300  # 5 minutos
    
    async def should_failover(self, instance_name: str) -> bool:
        """Verificar si se debe hacer failover"""
        # Implementar lógica de failover
        return False
    
    async def perform_failover(self, from_instance: str, to_instance: str):
        """Realizar failover"""
        print(f"🔄 Failover de {from_instance} a {to_instance}")

class LoadBalancer:
    def __init__(self):
        self.instance_weights = {}
        self.current_weights = {}
    
    def get_weighted_instance(self, instances: List['TruthGPTInstance']) -> 'TruthGPTInstance':
        """Obtener instancia con peso"""
        # Implementar balanceador de carga con pesos
        return instances[0]  # Simplificado

# Usar sistema de alta disponibilidad
ha_truthgpt = HighAvailabilityTruthGPT()

# Generar texto con alta disponibilidad
async def generate_with_ha():
    result = await ha_truthgpt.generate_text("Hola, ¿cómo estás?", 100)
    return result

# Ejecutar
result = asyncio.run(generate_with_ha())
print(f"Resultado: {result}")
```

## 🔗 Integración con ERP/CRM

### Ejemplo 1: Integración con Salesforce

```python
# enterprise/salesforce_integration.py
import requests
import json
from typing import Dict, List, Optional
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class SalesforceTruthGPT:
    def __init__(self, salesforce_config: Dict):
        self.salesforce_config = salesforce_config
        self.access_token = None
        self.instance_url = salesforce_config['instance_url']
        
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
        
        self.authenticate()
    
    def authenticate(self):
        """Autenticar con Salesforce"""
        auth_url = f"{self.instance_url}/services/oauth2/token"
        auth_data = {
            'grant_type': 'password',
            'client_id': self.salesforce_config['client_id'],
            'client_secret': self.salesforce_config['client_secret'],
            'username': self.salesforce_config['username'],
            'password': self.salesforce_config['password']
        }
        
        response = requests.post(auth_url, data=auth_data)
        if response.status_code == 200:
            auth_response = response.json()
            self.access_token = auth_response['access_token']
            print("✅ Autenticado con Salesforce")
        else:
            raise Exception(f"Error de autenticación: {response.text}")
    
    def get_headers(self) -> Dict[str, str]:
        """Obtener headers para requests"""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def analyze_lead_sentiment(self, lead_id: str) -> Dict:
        """Analizar sentimiento de un lead"""
        # Obtener datos del lead
        lead_data = self.get_lead_data(lead_id)
        
        # Analizar sentimiento
        prompt = f"Analiza el sentimiento de este lead: {lead_data.get('description', '')}"
        sentiment = self.optimizer.generate(
            input_text=prompt,
            max_length=100,
            temperature=0.3
        )
        
        # Guardar análisis en Salesforce
        self.save_sentiment_analysis(lead_id, sentiment)
        
        return {
            'lead_id': lead_id,
            'sentiment': sentiment,
            'confidence': 0.85
        }
    
    def get_lead_data(self, lead_id: str) -> Dict:
        """Obtener datos del lead"""
        url = f"{self.instance_url}/services/data/v52.0/sobjects/Lead/{lead_id}"
        response = requests.get(url, headers=self.get_headers())
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error al obtener lead: {response.text}")
    
    def save_sentiment_analysis(self, lead_id: str, sentiment: str):
        """Guardar análisis de sentimiento"""
        # Crear registro personalizado
        custom_object = {
            'Lead__c': lead_id,
            'Sentiment__c': sentiment,
            'Analysis_Date__c': datetime.utcnow().isoformat()
        }
        
        url = f"{self.instance_url}/services/data/v52.0/sobjects/Sentiment_Analysis__c"
        response = requests.post(url, headers=self.get_headers(), json=custom_object)
        
        if response.status_code == 201:
            print(f"✅ Análisis guardado para lead {lead_id}")
        else:
            print(f"❌ Error al guardar análisis: {response.text}")
    
    def generate_email_response(self, email_id: str) -> str:
        """Generar respuesta de email"""
        # Obtener email
        email_data = self.get_email_data(email_id)
        
        # Generar respuesta
        prompt = f"Genera una respuesta profesional para este email: {email_data.get('body', '')}"
        response = self.optimizer.generate(
            input_text=prompt,
            max_length=200,
            temperature=0.7
        )
        
        return response
    
    def get_email_data(self, email_id: str) -> Dict:
        """Obtener datos del email"""
        url = f"{self.instance_url}/services/data/v52.0/sobjects/EmailMessage/{email_id}"
        response = requests.get(url, headers=self.get_headers())
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error al obtener email: {response.text}")
    
    def classify_case(self, case_id: str) -> Dict:
        """Clasificar caso de soporte"""
        # Obtener datos del caso
        case_data = self.get_case_data(case_id)
        
        # Clasificar caso
        prompt = f"Clasifica este caso de soporte: {case_data.get('description', '')}"
        classification = self.optimizer.generate(
            input_text=prompt,
            max_length=100,
            temperature=0.3
        )
        
        # Actualizar caso en Salesforce
        self.update_case_classification(case_id, classification)
        
        return {
            'case_id': case_id,
            'classification': classification,
            'confidence': 0.90
        }
    
    def get_case_data(self, case_id: str) -> Dict:
        """Obtener datos del caso"""
        url = f"{self.instance_url}/services/data/v52.0/sobjects/Case/{case_id}"
        response = requests.get(url, headers=self.get_headers())
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error al obtener caso: {response.text}")
    
    def update_case_classification(self, case_id: str, classification: str):
        """Actualizar clasificación del caso"""
        update_data = {
            'Classification__c': classification,
            'Classification_Date__c': datetime.utcnow().isoformat()
        }
        
        url = f"{self.instance_url}/services/data/v52.0/sobjects/Case/{case_id}"
        response = requests.patch(url, headers=self.get_headers(), json=update_data)
        
        if response.status_code == 204:
            print(f"✅ Clasificación actualizada para caso {case_id}")
        else:
            print(f"❌ Error al actualizar clasificación: {response.text}")

# Usar integración con Salesforce
salesforce_config = {
    'instance_url': 'https://your-instance.salesforce.com',
    'client_id': 'your_client_id',
    'client_secret': 'your_client_secret',
    'username': 'your_username',
    'password': 'your_password'
}

salesforce_truthgpt = SalesforceTruthGPT(salesforce_config)

# Analizar sentimiento de lead
lead_analysis = salesforce_truthgpt.analyze_lead_sentiment('lead123')
print(f"Análisis de lead: {lead_analysis}")

# Generar respuesta de email
email_response = salesforce_truthgpt.generate_email_response('email123')
print(f"Respuesta de email: {email_response}")

# Clasificar caso
case_classification = salesforce_truthgpt.classify_case('case123')
print(f"Clasificación de caso: {case_classification}")
```

### Ejemplo 2: Integración con SAP

```python
# enterprise/sap_integration.py
import requests
import json
from typing import Dict, List, Optional
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class SAPTruthGPT:
    def __init__(self, sap_config: Dict):
        self.sap_config = sap_config
        self.base_url = sap_config['base_url']
        self.username = sap_config['username']
        self.password = sap_config['password']
        
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
        
        self.authenticate()
    
    def authenticate(self):
        """Autenticar con SAP"""
        auth_url = f"{self.base_url}/sap/bc/rest/authentication"
        auth_data = {
            'username': self.username,
            'password': self.password
        }
        
        response = requests.post(auth_url, json=auth_data)
        if response.status_code == 200:
            auth_response = response.json()
            self.session_id = auth_response['sessionId']
            print("✅ Autenticado con SAP")
        else:
            raise Exception(f"Error de autenticación: {response.text}")
    
    def get_headers(self) -> Dict[str, str]:
        """Obtener headers para requests"""
        return {
            'X-SAP-SessionId': self.session_id,
            'Content-Type': 'application/json'
        }
    
    def analyze_purchase_order(self, po_number: str) -> Dict:
        """Analizar orden de compra"""
        # Obtener datos de la orden
        po_data = self.get_purchase_order_data(po_number)
        
        # Analizar orden
        prompt = f"Analiza esta orden de compra: {po_data.get('description', '')}"
        analysis = self.optimizer.generate(
            input_text=prompt,
            max_length=200,
            temperature=0.3
        )
        
        return {
            'po_number': po_number,
            'analysis': analysis,
            'recommendations': self.generate_recommendations(analysis)
        }
    
    def get_purchase_order_data(self, po_number: str) -> Dict:
        """Obtener datos de la orden de compra"""
        url = f"{self.base_url}/sap/bc/rest/purchase_orders/{po_number}"
        response = requests.get(url, headers=self.get_headers())
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error al obtener orden: {response.text}")
    
    def generate_recommendations(self, analysis: str) -> List[str]:
        """Generar recomendaciones"""
        prompt = f"Genera recomendaciones basadas en este análisis: {analysis}"
        recommendations = self.optimizer.generate(
            input_text=prompt,
            max_length=300,
            temperature=0.5
        )
        
        return recommendations.split('\n')
    
    def optimize_inventory(self, material_number: str) -> Dict:
        """Optimizar inventario"""
        # Obtener datos del material
        material_data = self.get_material_data(material_number)
        
        # Optimizar inventario
        prompt = f"Optimiza el inventario para este material: {material_data.get('description', '')}"
        optimization = self.optimizer.generate(
            input_text=prompt,
            max_length=200,
            temperature=0.3
        )
        
        return {
            'material_number': material_number,
            'optimization': optimization,
            'suggested_actions': self.generate_inventory_actions(optimization)
        }
    
    def get_material_data(self, material_number: str) -> Dict:
        """Obtener datos del material"""
        url = f"{self.base_url}/sap/bc/rest/materials/{material_number}"
        response = requests.get(url, headers=self.get_headers())
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error al obtener material: {response.text}")
    
    def generate_inventory_actions(self, optimization: str) -> List[str]:
        """Generar acciones de inventario"""
        prompt = f"Genera acciones específicas basadas en esta optimización: {optimization}"
        actions = self.optimizer.generate(
            input_text=prompt,
            max_length=300,
            temperature=0.5
        )
        
        return actions.split('\n')

# Usar integración con SAP
sap_config = {
    'base_url': 'https://your-sap-system.com',
    'username': 'your_username',
    'password': 'your_password'
}

sap_truthgpt = SAPTruthGPT(sap_config)

# Analizar orden de compra
po_analysis = sap_truthgpt.analyze_purchase_order('PO123456')
print(f"Análisis de orden: {po_analysis}")

# Optimizar inventario
inventory_optimization = sap_truthgpt.optimize_inventory('MAT123456')
print(f"Optimización de inventario: {inventory_optimization}")
```

## 📊 Compliance y Auditoría

### Ejemplo 1: Sistema de Auditoría Completo

```python
# enterprise/compliance_audit.py
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class ComplianceAuditSystem:
    def __init__(self):
        self.audit_logs = []
        self.compliance_rules = self.load_compliance_rules()
        self.data_retention_policy = self.load_data_retention_policy()
        
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
    
    def load_compliance_rules(self) -> Dict:
        """Cargar reglas de compliance"""
        return {
            'gdpr': {
                'data_retention_days': 365,
                'anonymization_required': True,
                'consent_required': True
            },
            'sox': {
                'audit_trail_required': True,
                'data_integrity_required': True,
                'access_controls_required': True
            },
            'hipaa': {
                'phi_protection_required': True,
                'encryption_required': True,
                'access_logging_required': True
            }
        }
    
    def load_data_retention_policy(self) -> Dict:
        """Cargar política de retención de datos"""
        return {
            'audit_logs': 2555,  # 7 años en días
            'user_data': 365,    # 1 año
            'system_logs': 90,   # 3 meses
            'performance_metrics': 30  # 1 mes
        }
    
    def log_audit_event(self, event_type: str, user_id: str, action: str, 
                       data: Dict, compliance_framework: str = 'general'):
        """Registrar evento de auditoría"""
        audit_entry = {
            'event_id': self.generate_event_id(),
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'action': action,
            'data_hash': self.hash_sensitive_data(data),
            'compliance_framework': compliance_framework,
            'ip_address': self.get_client_ip(),
            'user_agent': self.get_user_agent(),
            'session_id': self.get_session_id()
        }
        
        # Aplicar reglas de compliance
        audit_entry = self.apply_compliance_rules(audit_entry, compliance_framework)
        
        # Guardar en audit log
        self.audit_logs.append(audit_entry)
        
        # Verificar compliance
        self.check_compliance_violations(audit_entry)
        
        return audit_entry['event_id']
    
    def generate_event_id(self) -> str:
        """Generar ID único del evento"""
        timestamp = datetime.utcnow().timestamp()
        random_component = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        return f"audit_{int(timestamp)}_{random_component}"
    
    def hash_sensitive_data(self, data: Dict) -> str:
        """Hashear datos sensibles"""
        data_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def apply_compliance_rules(self, audit_entry: Dict, framework: str) -> Dict:
        """Aplicar reglas de compliance"""
        if framework in self.compliance_rules:
            rules = self.compliance_rules[framework]
            
            if rules.get('anonymization_required', False):
                audit_entry['user_id'] = self.anonymize_user_id(audit_entry['user_id'])
            
            if rules.get('encryption_required', False):
                audit_entry['data_hash'] = self.encrypt_data(audit_entry['data_hash'])
        
        return audit_entry
    
    def anonymize_user_id(self, user_id: str) -> str:
        """Anonimizar ID de usuario"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def encrypt_data(self, data: str) -> str:
        """Encriptar datos"""
        # Implementar encriptación real
        return hashlib.sha256(data.encode()).hexdigest()
    
    def check_compliance_violations(self, audit_entry: Dict):
        """Verificar violaciones de compliance"""
        violations = []
        
        # Verificar retención de datos
        if self.is_data_retention_violation(audit_entry):
            violations.append('data_retention_violation')
        
        # Verificar acceso no autorizado
        if self.is_unauthorized_access(audit_entry):
            violations.append('unauthorized_access')
        
        # Verificar integridad de datos
        if self.is_data_integrity_violation(audit_entry):
            violations.append('data_integrity_violation')
        
        if violations:
            self.handle_compliance_violations(audit_entry, violations)
    
    def is_data_retention_violation(self, audit_entry: Dict) -> bool:
        """Verificar violación de retención de datos"""
        # Implementar verificación de retención
        return False
    
    def is_unauthorized_access(self, audit_entry: Dict) -> bool:
        """Verificar acceso no autorizado"""
        # Implementar verificación de acceso
        return False
    
    def is_data_integrity_violation(self, audit_entry: Dict) -> bool:
        """Verificar violación de integridad de datos"""
        # Implementar verificación de integridad
        return False
    
    def handle_compliance_violations(self, audit_entry: Dict, violations: List[str]):
        """Manejar violaciones de compliance"""
        violation_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_id': audit_entry['event_id'],
            'violations': violations,
            'severity': self.calculate_severity(violations),
            'action_required': self.determine_action_required(violations)
        }
        
        # Enviar alerta
        self.send_compliance_alert(violation_report)
        
        # Registrar violación
        self.log_compliance_violation(violation_report)
    
    def calculate_severity(self, violations: List[str]) -> str:
        """Calcular severidad de violaciones"""
        if 'unauthorized_access' in violations:
            return 'critical'
        elif 'data_integrity_violation' in violations:
            return 'high'
        elif 'data_retention_violation' in violations:
            return 'medium'
        else:
            return 'low'
    
    def determine_action_required(self, violations: List[str]) -> List[str]:
        """Determinar acciones requeridas"""
        actions = []
        
        if 'unauthorized_access' in violations:
            actions.extend(['block_user', 'notify_security', 'investigate_incident'])
        
        if 'data_integrity_violation' in violations:
            actions.extend(['verify_data', 'restore_backup', 'notify_admin'])
        
        if 'data_retention_violation' in violations:
            actions.extend(['cleanup_data', 'update_policy', 'notify_compliance'])
        
        return actions
    
    def send_compliance_alert(self, violation_report: Dict):
        """Enviar alerta de compliance"""
        print(f"🚨 ALERTA DE COMPLIANCE: {violation_report}")
        # Implementar envío de alertas (email, Slack, etc.)
    
    def log_compliance_violation(self, violation_report: Dict):
        """Registrar violación de compliance"""
        # Guardar en base de datos de violaciones
        pass
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generar reporte de compliance"""
        # Filtrar logs por fecha
        filtered_logs = [
            log for log in self.audit_logs
            if start_date <= datetime.fromisoformat(log['timestamp']) <= end_date
        ]
        
        # Generar estadísticas
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_events': len(filtered_logs),
            'compliance_frameworks': self.analyze_compliance_frameworks(filtered_logs),
            'violations': self.analyze_violations(filtered_logs),
            'recommendations': self.generate_recommendations(filtered_logs)
        }
        
        return report
    
    def analyze_compliance_frameworks(self, logs: List[Dict]) -> Dict:
        """Analizar frameworks de compliance"""
        frameworks = {}
        for log in logs:
            framework = log.get('compliance_framework', 'general')
            frameworks[framework] = frameworks.get(framework, 0) + 1
        
        return frameworks
    
    def analyze_violations(self, logs: List[Dict]) -> Dict:
        """Analizar violaciones"""
        # Implementar análisis de violaciones
        return {
            'total_violations': 0,
            'by_severity': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
    
    def generate_recommendations(self, logs: List[Dict]) -> List[str]:
        """Generar recomendaciones"""
        # Usar TruthGPT para generar recomendaciones
        prompt = f"Genera recomendaciones de compliance basadas en estos logs: {len(logs)} eventos"
        recommendations = self.optimizer.generate(
            input_text=prompt,
            max_length=300,
            temperature=0.5
        )
        
        return recommendations.split('\n')
    
    def cleanup_old_data(self):
        """Limpiar datos antiguos según política de retención"""
        current_date = datetime.utcnow()
        
        for data_type, retention_days in self.data_retention_policy.items():
            cutoff_date = current_date - timedelta(days=retention_days)
            
            # Limpiar datos antiguos
            if data_type == 'audit_logs':
                self.audit_logs = [
                    log for log in self.audit_logs
                    if datetime.fromisoformat(log['timestamp']) > cutoff_date
                ]
    
    def get_client_ip(self) -> str:
        """Obtener IP del cliente"""
        # Implementar obtención de IP
        return "127.0.0.1"
    
    def get_user_agent(self) -> str:
        """Obtener User Agent"""
        # Implementar obtención de User Agent
        return "TruthGPT-Enterprise/2.0"
    
    def get_session_id(self) -> str:
        """Obtener ID de sesión"""
        # Implementar obtención de ID de sesión
        return "session_123"

# Usar sistema de compliance
compliance_system = ComplianceAuditSystem()

# Registrar evento de auditoría
event_id = compliance_system.log_audit_event(
    event_type='text_generation',
    user_id='user123',
    action='generate_text',
    data={'input_text': 'Hola', 'output_text': 'Hola, ¿cómo estás?'},
    compliance_framework='gdpr'
)

print(f"Evento registrado: {event_id}")

# Generar reporte de compliance
start_date = datetime.utcnow() - timedelta(days=30)
end_date = datetime.utcnow()
report = compliance_system.generate_compliance_report(start_date, end_date)
print(f"Reporte de compliance: {report}")
```

## 🏢 Multi-tenant

### Ejemplo 1: Sistema Multi-tenant

```python
# enterprise/multi_tenant.py
from typing import Dict, List, Optional
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig
import json
import hashlib

class MultiTenantTruthGPT:
    def __init__(self):
        self.tenants = {}
        self.tenant_configs = {}
        self.tenant_models = {}
        self.tenant_metrics = {}
        
        self.initialize_tenants()
    
    def initialize_tenants(self):
        """Inicializar tenants"""
        # Tenant 1: Empresa A
        self.create_tenant('tenant_a', {
            'name': 'Empresa A',
            'model': 'microsoft/DialoGPT-medium',
            'max_requests_per_hour': 1000,
            'features': ['text_generation', 'sentiment_analysis']
        })
        
        # Tenant 2: Empresa B
        self.create_tenant('tenant_b', {
            'name': 'Empresa B',
            'model': 'microsoft/DialoGPT-large',
            'max_requests_per_hour': 5000,
            'features': ['text_generation', 'translation', 'summarization']
        })
        
        # Tenant 3: Empresa C
        self.create_tenant('tenant_c', {
            'name': 'Empresa C',
            'model': 'microsoft/DialoGPT-small',
            'max_requests_per_hour': 100,
            'features': ['text_generation']
        })
    
    def create_tenant(self, tenant_id: str, config: Dict):
        """Crear tenant"""
        self.tenants[tenant_id] = {
            'id': tenant_id,
            'name': config['name'],
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        self.tenant_configs[tenant_id] = config
        
        # Crear modelo específico para el tenant
        self.tenant_models[tenant_id] = self.create_tenant_model(config)
        
        # Inicializar métricas
        self.tenant_metrics[tenant_id] = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'last_request': None
        }
    
    def create_tenant_model(self, config: Dict) -> ModernTruthGPTOptimizer:
        """Crear modelo específico para el tenant"""
        truthgpt_config = TruthGPTConfig(
            model_name=config['model'],
            use_mixed_precision=True
        )
        
        return ModernTruthGPTOptimizer(truthgpt_config)
    
    def process_tenant_request(self, tenant_id: str, request: Dict) -> Dict:
        """Procesar solicitud del tenant"""
        # Verificar que el tenant existe
        if tenant_id not in self.tenants:
            raise Exception(f"Tenant {tenant_id} no encontrado")
        
        # Verificar límites del tenant
        if not self.check_tenant_limits(tenant_id):
            raise Exception(f"Límites excedidos para tenant {tenant_id}")
        
        # Obtener modelo del tenant
        model = self.tenant_models[tenant_id]
        
        # Procesar solicitud
        start_time = datetime.utcnow()
        
        try:
            result = model.generate(
                input_text=request.get('text', ''),
                max_length=request.get('max_length', 100),
                temperature=request.get('temperature', 0.7)
            )
            
            # Registrar éxito
            self.record_tenant_success(tenant_id, start_time)
            
            return {
                'tenant_id': tenant_id,
                'result': result,
                'success': True
            }
            
        except Exception as e:
            # Registrar fallo
            self.record_tenant_failure(tenant_id, start_time)
            raise e
    
    def check_tenant_limits(self, tenant_id: str) -> bool:
        """Verificar límites del tenant"""
        config = self.tenant_configs[tenant_id]
        metrics = self.tenant_metrics[tenant_id]
        
        # Verificar límite por hora
        max_requests = config.get('max_requests_per_hour', 1000)
        if metrics['total_requests'] >= max_requests:
            return False
        
        return True
    
    def record_tenant_success(self, tenant_id: str, start_time: datetime):
        """Registrar éxito del tenant"""
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        metrics = self.tenant_metrics[tenant_id]
        metrics['total_requests'] += 1
        metrics['successful_requests'] += 1
        metrics['last_request'] = end_time.isoformat()
        
        # Actualizar tiempo promedio de respuesta
        if metrics['average_response_time'] == 0:
            metrics['average_response_time'] = duration
        else:
            metrics['average_response_time'] = (
                metrics['average_response_time'] + duration
            ) / 2
    
    def record_tenant_failure(self, tenant_id: str, start_time: datetime):
        """Registrar fallo del tenant"""
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        metrics = self.tenant_metrics[tenant_id]
        metrics['total_requests'] += 1
        metrics['failed_requests'] += 1
        metrics['last_request'] = end_time.isoformat()
    
    def get_tenant_metrics(self, tenant_id: str) -> Dict:
        """Obtener métricas del tenant"""
        if tenant_id not in self.tenant_metrics:
            return {}
        
        return self.tenant_metrics[tenant_id]
    
    def get_all_tenants_metrics(self) -> Dict:
        """Obtener métricas de todos los tenants"""
        all_metrics = {}
        
        for tenant_id in self.tenants:
            all_metrics[tenant_id] = self.get_tenant_metrics(tenant_id)
        
        return all_metrics
    
    def update_tenant_config(self, tenant_id: str, new_config: Dict):
        """Actualizar configuración del tenant"""
        if tenant_id not in self.tenants:
            raise Exception(f"Tenant {tenant_id} no encontrado")
        
        # Actualizar configuración
        self.tenant_configs[tenant_id].update(new_config)
        
        # Recrear modelo si es necesario
        if 'model' in new_config:
            self.tenant_models[tenant_id] = self.create_tenant_model(
                self.tenant_configs[tenant_id]
            )
    
    def suspend_tenant(self, tenant_id: str):
        """Suspender tenant"""
        if tenant_id in self.tenants:
            self.tenants[tenant_id]['status'] = 'suspended'
    
    def activate_tenant(self, tenant_id: str):
        """Activar tenant"""
        if tenant_id in self.tenants:
            self.tenants[tenant_id]['status'] = 'active'
    
    def delete_tenant(self, tenant_id: str):
        """Eliminar tenant"""
        if tenant_id in self.tenants:
            del self.tenants[tenant_id]
            del self.tenant_configs[tenant_id]
            del self.tenant_models[tenant_id]
            del self.tenant_metrics[tenant_id]

# Usar sistema multi-tenant
multi_tenant_truthgpt = MultiTenantTruthGPT()

# Procesar solicitud del tenant A
request_a = {
    'text': 'Hola, ¿cómo estás?',
    'max_length': 100,
    'temperature': 0.7
}

result_a = multi_tenant_truthgpt.process_tenant_request('tenant_a', request_a)
print(f"Resultado tenant A: {result_a}")

# Procesar solicitud del tenant B
request_b = {
    'text': '¿Qué tal el clima?',
    'max_length': 150,
    'temperature': 0.8
}

result_b = multi_tenant_truthgpt.process_tenant_request('tenant_b', request_b)
print(f"Resultado tenant B: {result_b}")

# Obtener métricas de todos los tenants
all_metrics = multi_tenant_truthgpt.get_all_tenants_metrics()
print(f"Métricas de todos los tenants: {all_metrics}")
```

## 🎯 Próximos Pasos

### 1. Implementar en Producción
```python
# Configuración de producción empresarial
enterprise_config = {
    'high_availability': True,
    'load_balancing': True,
    'multi_tenant': True,
    'compliance': True,
    'audit': True,
    'monitoring': True,
    'security': True
}
```

### 2. Escalar Horizontalmente
```python
# Escalabilidad horizontal empresarial
def enterprise_scaling():
    # Distribuir carga
    # Balancear requests
    # Sincronizar estado
    # Replicar datos
    # Manejar fallos
    pass
```

### 3. Mantener Compliance
```python
# Compliance continuo
def continuous_compliance():
    # Monitorear compliance
    # Generar reportes
    # Manejar violaciones
    # Actualizar políticas
    pass
```

---

*¡Con estos ejemplos empresariales tienes todo lo necesario para implementar TruthGPT a escala empresarial! 🚀✨*


