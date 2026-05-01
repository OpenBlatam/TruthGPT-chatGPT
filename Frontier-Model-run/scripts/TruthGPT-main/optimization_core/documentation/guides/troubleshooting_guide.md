# Guía de Solución de Problemas - TruthGPT

Esta guía te ayudará a resolver los problemas más comunes que pueden surgir al usar TruthGPT.

## 📋 Tabla de Contenidos

1. [Problemas de Instalación](#problemas-de-instalación)
2. [Problemas de Memoria](#problemas-de-memoria)
3. [Problemas de GPU](#problemas-de-gpu)
4. [Problemas de Rendimiento](#problemas-de-rendimiento)
5. [Problemas de Modelos](#problemas-de-modelos)
6. [Problemas de API](#problemas-de-api)
7. [Problemas de Despliegue](#problemas-de-despliegue)
8. [Diagnóstico Automático](#diagnóstico-automático)

## 🔧 Problemas de Instalación

### Error: "ModuleNotFoundError: No module named 'optimization_core'"

**Causa**: TruthGPT no está instalado correctamente.

**Solución**:
```bash
# Verificar instalación
pip list | grep truthgpt

# Reinstalar si es necesario
pip uninstall truthgpt
pip install -r requirements_modern.txt

# Verificar instalación
python -c "from optimization_core import *; print('✅ TruthGPT instalado')"
```

### Error: "CUDA out of memory"

**Causa**: Memoria GPU insuficiente.

**Solución**:
```python
# Reducir batch size
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    batch_size=1,  # Reducir de 4 a 1
    use_mixed_precision=True
)

# Limpiar memoria GPU
import torch
torch.cuda.empty_cache()

# Usar CPU si es necesario
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    device="cpu"
)
```

### Error: "No module named 'torch'"

**Causa**: PyTorch no está instalado.

**Solución**:
```bash
# Instalar PyTorch
pip install torch torchvision torchaudio

# Para CUDA (si tienes GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verificar instalación
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## 💾 Problemas de Memoria

### Error: "Out of memory"

**Causa**: Memoria RAM insuficiente.

**Solución**:
```python
# Optimización de memoria
from optimization_core import create_memory_optimizer

memory_config = {
    "use_gradient_checkpointing": True,
    "use_activation_checkpointing": True,
    "use_memory_efficient_attention": True,
    "use_offload": True
}

memory_optimizer = create_memory_optimizer(memory_config)
optimized_optimizer = memory_optimizer.optimize(optimizer)
```

### Error: "Memory allocation failed"

**Causa**: Fragmentación de memoria.

**Solución**:
```python
# Limpiar memoria
import gc
import torch

# Limpiar Python
gc.collect()

# Limpiar PyTorch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Reiniciar proceso si es necesario
```

### Error: "Memory usage too high"

**Causa**: Uso excesivo de memoria.

**Solución**:
```python
# Monitorear memoria
import psutil
import torch

def check_memory_usage():
    # Memoria RAM
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.percent:.1f}% usado")
    
    # Memoria GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_memory:.1f}GB / {gpu_total:.1f}GB")

# Verificar antes de usar
check_memory_usage()
```

## 🎮 Problemas de GPU

### Error: "CUDA device not found"

**Causa**: GPU no disponible o no configurada.

**Solución**:
```python
import torch

# Verificar GPU
if torch.cuda.is_available():
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("GPU no disponible, usando CPU")
    device = "cpu"

# Configurar dispositivo
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    device=device
)
```

### Error: "CUDA out of memory"

**Causa**: Memoria GPU insuficiente.

**Solución**:
```python
# Reducir memoria GPU
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True,
    batch_size=1,
    max_length=50  # Reducir longitud
)

# Limpiar memoria GPU
torch.cuda.empty_cache()

# Usar CPU si es necesario
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    device="cpu"
)
```

### Error: "CUDA kernel launch failed"

**Causa**: Problema con kernels CUDA.

**Solución**:
```python
# Verificar CUDA
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"CUDA versión: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Reiniciar CUDA
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Usar CPU como fallback
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    device="cpu"
)
```

## ⚡ Problemas de Rendimiento

### Error: "Generation too slow"

**Causa**: Rendimiento lento.

**Solución**:
```python
# Optimización de velocidad
from optimization_core import create_ultra_fast_optimizer

speed_config = {
    "use_parallel_processing": True,
    "use_kernel_fusion": True,
    "use_batch_optimization": True
}

speed_optimizer = create_ultra_fast_optimizer(speed_config)
optimized_optimizer = speed_optimizer.optimize(optimizer)
```

### Error: "High CPU usage"

**Causa**: Uso excesivo de CPU.

**Solución**:
```python
# Optimizar CPU
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True,
    use_gradient_checkpointing=True,
    num_workers=1  # Reducir workers
)

# Usar GPU si está disponible
if torch.cuda.is_available():
    config.device = "cuda"
```

### Error: "Low throughput"

**Causa**: Rendimiento bajo.

**Solución**:
```python
# Benchmark de rendimiento
import time

def benchmark_optimizer(optimizer, test_text="Hola"):
    start_time = time.time()
    result = optimizer.generate(test_text, max_length=100)
    end_time = time.time()
    
    generation_time = end_time - start_time
    tokens_per_second = len(result) / generation_time
    
    print(f"Tiempo: {generation_time:.2f}s")
    print(f"Velocidad: {tokens_per_second:.2f} tokens/s")
    
    return generation_time, tokens_per_second

# Probar rendimiento
benchmark_optimizer(optimizer)
```

## 🧠 Problemas de Modelos

### Error: "Model not found"

**Causa**: Modelo no disponible.

**Solución**:
```python
# Verificar modelos disponibles
from transformers import AutoTokenizer, AutoModelForCausalLM

# Lista de modelos disponibles
models = [
    "microsoft/DialoGPT-small",
    "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-large",
    "gpt2",
    "gpt2-medium"
]

# Probar cada modelo
for model_name in models:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ {model_name} disponible")
    except Exception as e:
        print(f"❌ {model_name} no disponible: {e}")
```

### Error: "Tokenizer not found"

**Causa**: Tokenizador no disponible.

**Solución**:
```python
# Usar tokenizador por defecto
from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
except:
    # Fallback a GPT-2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
```

### Error: "Model loading failed"

**Causa**: Error al cargar modelo.

**Solución**:
```python
# Cargar modelo con configuración segura
from transformers import AutoModelForCausalLM

try:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-medium",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
except Exception as e:
    print(f"Error al cargar modelo: {e}")
    # Usar modelo más pequeño
    model = AutoModelForCausalLM.from_pretrained("gpt2")
```

## 🌐 Problemas de API

### Error: "Connection refused"

**Causa**: API no disponible.

**Solución**:
```python
# Verificar API
import requests

def check_api_health(base_url="http://localhost:8000"):
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API funcionando")
            return True
        else:
            print(f"❌ API error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API no disponible")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Verificar API
check_api_health()
```

### Error: "Timeout"

**Causa**: Tiempo de espera agotado.

**Solución**:
```python
# Configurar timeout
import requests

# Aumentar timeout
response = requests.post(
    "http://localhost:8000/generate",
    json={"text": "Hola", "max_length": 100},
    timeout=30  # 30 segundos
)

# Configurar retry
import time

def generate_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/generate",
                json={"text": text, "max_length": 100},
                timeout=10
            )
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Intento {attempt + 1} falló, reintentando...")
            time.sleep(2)
    
    raise Exception("Todos los intentos fallaron")
```

### Error: "Rate limit exceeded"

**Causa**: Límite de velocidad excedido.

**Solución**:
```python
# Implementar rate limiting
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests=10, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    def can_make_request(self):
        now = time.time()
        # Limpiar requests antiguos
        while self.requests and self.requests[0] <= now - self.time_window:
            self.requests.popleft()
        
        # Verificar límite
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
    
    def wait_if_needed(self):
        if not self.can_make_request():
            sleep_time = self.time_window - (time.time() - self.requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

# Usar rate limiter
rate_limiter = RateLimiter(max_requests=5, time_window=60)

def generate_with_rate_limit(text):
    rate_limiter.wait_if_needed()
    response = requests.post(
        "http://localhost:8000/generate",
        json={"text": text, "max_length": 100}
    )
    return response.json()
```

## 🚀 Problemas de Despliegue

### Error: "Docker build failed"

**Causa**: Error en construcción de Docker.

**Solución**:
```dockerfile
# Dockerfile optimizado
FROM python:3.8-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero (para cache)
COPY requirements_modern.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements_modern.txt

# Copiar código
COPY . .

# Crear usuario no-root
RUN useradd -m -u 1000 truthgpt && chown -R truthgpt:truthgpt /app
USER truthgpt

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["python", "main.py"]
```

### Error: "Kubernetes deployment failed"

**Causa**: Error en despliegue de Kubernetes.

**Solución**:
```yaml
# deployment.yaml optimizado
apiVersion: apps/v1
kind: Deployment
metadata:
  name: truthgpt
spec:
  replicas: 2
  selector:
    matchLabels:
      app: truthgpt
  template:
    metadata:
      labels:
        app: truthgpt
    spec:
      containers:
      - name: truthgpt
        image: truthgpt:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Error: "Service not accessible"

**Causa**: Servicio no accesible.

**Solución**:
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: truthgpt-service
spec:
  selector:
    app: truthgpt
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🔍 Diagnóstico Automático

### Herramienta de Diagnóstico

```python
# diagnostic_tool.py
import torch
import psutil
import requests
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class TruthGPTDiagnostic:
    def __init__(self):
        self.issues = []
        self.solutions = []
    
    def run_full_diagnostic(self):
        """Ejecutar diagnóstico completo"""
        print("🔍 Iniciando diagnóstico de TruthGPT...")
        
        # Verificar sistema
        self.check_system()
        
        # Verificar GPU
        self.check_gpu()
        
        # Verificar memoria
        self.check_memory()
        
        # Verificar modelo
        self.check_model()
        
        # Verificar API
        self.check_api()
        
        # Mostrar resultados
        self.show_results()
    
    def check_system(self):
        """Verificar sistema"""
        print("🖥️  Verificando sistema...")
        
        # Python version
        import sys
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            self.issues.append("Python version < 3.8")
            self.solutions.append("Actualizar Python a 3.8+")
        else:
            print(f"✅ Python {python_version.major}.{python_version.minor}")
        
        # PyTorch
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__}")
        except ImportError:
            self.issues.append("PyTorch no instalado")
            self.solutions.append("pip install torch")
    
    def check_gpu(self):
        """Verificar GPU"""
        print("🎮 Verificando GPU...")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  GPU no disponible, usando CPU")
            self.issues.append("GPU no disponible")
            self.solutions.append("Instalar CUDA o usar CPU")
    
    def check_memory(self):
        """Verificar memoria"""
        print("💾 Verificando memoria...")
        
        # RAM
        memory = psutil.virtual_memory()
        if memory.available < 2 * 1024**3:  # 2GB
            self.issues.append("Memoria RAM insuficiente")
            self.solutions.append("Aumentar RAM o reducir batch size")
        else:
            print(f"✅ RAM: {memory.available / 1024**3:.1f}GB disponible")
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_reserved() / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory > gpu_total * 0.9:
                self.issues.append("Memoria GPU alta")
                self.solutions.append("Limpiar memoria GPU o reducir modelo")
            else:
                print(f"✅ GPU Memory: {gpu_memory:.1f}GB / {gpu_total:.1f}GB")
    
    def check_model(self):
        """Verificar modelo"""
        print("🧠 Verificando modelo...")
        
        try:
            config = TruthGPTConfig(
                model_name="microsoft/DialoGPT-medium",
                use_mixed_precision=True
            )
            optimizer = ModernTruthGPTOptimizer(config)
            
            # Probar generación
            result = optimizer.generate("Test", max_length=10)
            print("✅ Modelo funcionando")
            
        except Exception as e:
            self.issues.append(f"Error en modelo: {e}")
            self.solutions.append("Verificar configuración del modelo")
    
    def check_api(self):
        """Verificar API"""
        print("🌐 Verificando API...")
        
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API funcionando")
            else:
                self.issues.append(f"API error: {response.status_code}")
                self.solutions.append("Verificar configuración de API")
        except requests.exceptions.ConnectionError:
            self.issues.append("API no disponible")
            self.solutions.append("Iniciar servidor API")
        except Exception as e:
            self.issues.append(f"Error de API: {e}")
            self.solutions.append("Verificar configuración de red")
    
    def show_results(self):
        """Mostrar resultados"""
        print("\n📊 RESULTADOS DEL DIAGNÓSTICO")
        print("=" * 40)
        
        if not self.issues:
            print("🎉 ¡Todo funcionando correctamente!")
        else:
            print(f"❌ {len(self.issues)} problemas encontrados:")
            
            for i, (issue, solution) in enumerate(zip(self.issues, self.solutions)):
                print(f"\n{i+1}. {issue}")
                print(f"   💡 Solución: {solution}")

# Ejecutar diagnóstico
diagnostic = TruthGPTDiagnostic()
diagnostic.run_full_diagnostic()
```

### Herramienta de Optimización Automática

```python
# auto_optimizer.py
class TruthGPTAutoOptimizer:
    def __init__(self):
        self.optimizations = []
    
    def auto_optimize(self, config):
        """Optimizar automáticamente según el sistema"""
        print("⚡ Iniciando optimización automática...")
        
        # Detectar hardware
        if torch.cuda.is_available():
            self.optimize_for_gpu(config)
        else:
            self.optimize_for_cpu(config)
        
        # Detectar memoria
        memory = psutil.virtual_memory()
        if memory.available < 4 * 1024**3:  # 4GB
            self.optimize_for_low_memory(config)
        
        # Aplicar optimizaciones
        self.apply_optimizations(config)
    
    def optimize_for_gpu(self, config):
        """Optimizar para GPU"""
        print("🎮 Optimizando para GPU...")
        
        config.use_mixed_precision = True
        config.device = "cuda"
        config.use_tensor_cores = True
        
        self.optimizations.append("GPU optimization")
    
    def optimize_for_cpu(self, config):
        """Optimizar para CPU"""
        print("🖥️  Optimizando para CPU...")
        
        config.device = "cpu"
        config.use_mixed_precision = False
        config.batch_size = 1
        
        self.optimizations.append("CPU optimization")
    
    def optimize_for_low_memory(self, config):
        """Optimizar para poca memoria"""
        print("💾 Optimizando para poca memoria...")
        
        config.use_gradient_checkpointing = True
        config.use_activation_checkpointing = True
        config.batch_size = 1
        config.max_length = 50
        
        self.optimizations.append("Memory optimization")
    
    def apply_optimizations(self, config):
        """Aplicar optimizaciones"""
        print(f"✅ Aplicadas {len(self.optimizations)} optimizaciones:")
        for opt in self.optimizations:
            print(f"   - {opt}")

# Usar optimizador automático
auto_optimizer = TruthGPTAutoOptimizer()
config = TruthGPTConfig(model_name="microsoft/DialoGPT-medium")
auto_optimizer.auto_optimize(config)
```

## 🆘 Soporte Adicional

### Recursos de Ayuda

1. **GitHub Issues** - Reportar problemas
2. **Documentación** - Guías detalladas
3. **Comunidad** - Discord, Reddit
4. **Email** - Soporte directo

### Comandos de Diagnóstico

```bash
# Verificar instalación
python -c "from optimization_core import *; print('TruthGPT OK')"

# Verificar GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verificar memoria
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"

# Verificar API
curl http://localhost:8000/health
```

### Logs de Diagnóstico

```python
# Habilitar logs detallados
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Logs de TruthGPT
logger = logging.getLogger('truthgpt')
logger.setLevel(logging.DEBUG)
```

---

*¡Con esta guía puedes resolver cualquier problema de TruthGPT! 🚀✨*


