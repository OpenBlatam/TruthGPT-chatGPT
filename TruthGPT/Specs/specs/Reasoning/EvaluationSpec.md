# Evaluation Specification

## Overview
This specification defines the evaluation system for TruthGPT using an advanced modular architecture. The system implements a layered approach with pluggable components, kernel management, and flexible execution pipelines.

## Core Architecture

### Key Layers of Abstraction
| Layer | Responsibility | Modular Feature |
|-------|----------------|-----------------|
| `BaseComponent` | Interface all evaluation components must implement | Unified contract |
| `LayerFactory` | Dynamically register and instantiate evaluation layers | Pluggable layers |
| `KernelRegistry` | Register evaluation kernels (Triton/PyTorch) | Kernel swapping |
| `WrapperComponent` | Adapts kernels into evaluation functions | Hardware portability |
| `Config` | Structured configs for evaluation components | Full parameterization |
| `ExecutionPipeline` | Controls evaluation logic | Clean orchestration |

### Modular Components
| Component | Responsibility | Modular Feature |
|-----------|----------------|-----------------|
| `BaseEvaluator` | Abstract base for all evaluators | Unified interface |
| `KernelInterface` | Interface for evaluation kernels | Kernel abstraction |
| `KernelRegistry` | Manages evaluation kernels | Kernel decoupling |
| `LayerFactory` | Instantiates evaluation layers | Swappable layers |
| `BackendFactory` | Selects evaluation backends | Backend plug-in |
| `ConfigManager` | Handles evaluation configs | Declarative config |
| `ExecutionPipeline` | Coordinates evaluation | Pipeline orchestration |
| `Evaluator` | Encapsulates evaluation logic | Pluggable evaluation |
| `Logger` | Tracks evaluation metrics | Observability |

## Implementation Details

### Base Components
```python
class BaseComponent(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.kernel_registry = KernelRegistry()
        self.layer_factory = LayerFactory()
    
    def forward(self, x):
        raise NotImplementedError

class BaseEvaluator(BaseComponent):
    def __init__(self, config: Config):
        super().__init__(config)
        self.metrics = config.metrics
        self.state = EvaluationState()
        self.cache = LRUCache(maxsize=CACHE_SIZE)
        self.worker_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    
    def evaluate(self, model: nn.Module, dataset: Dataset) -> Dict[str, float]:
        """Evaluate model on dataset with modular processing."""
        # Get evaluation kernel
        kernel = self.kernel_registry.get_kernel(
            self.config.evaluation_kernel
        )
        
        # Create evaluation pipeline
        pipeline = ExecutionPipeline(
            kernel=kernel,
            config=self.config
        )
        
        # Execute evaluation
        return pipeline.execute(model, dataset)
```

### Kernel Management
```python
class KernelInterface(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
    
    def forward(self, x):
        raise NotImplementedError

class EvaluationKernel(KernelInterface):
    def __init__(self, config: Config):
        super().__init__(config)
        self.backend = BackendFactory.create_backend(
            config.backend_type
        )
    
    def forward(self, model: nn.Module, batch: Batch) -> Dict[str, float]:
        """Execute evaluation kernel."""
        return self.backend.evaluate(model, batch)
```

### Layer Factory
```python
class LayerFactory:
    def __init__(self, config: Config):
        self.config = config
        self.registry = {}
    
    def register_layer(self, name: str, layer_class: Type[BaseComponent]):
        """Register a new evaluation layer."""
        self.registry[name] = layer_class
    
    def create_layer(self, name: str, **kwargs) -> BaseComponent:
        """Create an evaluation layer instance."""
        if name not in self.registry:
            raise ValueError(f"Layer {name} not registered")
        return self.registry[name](self.config, **kwargs)
```

### Execution Pipeline
```python
class ExecutionPipeline:
    def __init__(self, kernel: KernelInterface, config: Config):
        self.kernel = kernel
        self.config = config
        self.logger = Logger(config)
    
    def execute(self, model: nn.Module, dataset: Dataset) -> Dict[str, float]:
        """Execute evaluation pipeline."""
        # Create batches
        batches = self._create_batches(dataset)
        
        # Process batches
        results = []
        for batch in batches:
            # Execute kernel
            batch_result = self.kernel(model, batch)
            results.append(batch_result)
            
            # Log metrics
            self.logger.log_metrics(batch_result)
        
        return self._aggregate_results(results)
```

## Configuration

### Config Structure
```python
@dataclass
class Config:
    # Evaluation parameters
    metrics: List[str]
    batch_size: int
    max_length: int
    
    # Kernel parameters
    evaluation_kernel: str
    backend_type: str
    
    # Pipeline parameters
    num_workers: int
    cache_size: int
    timeout: int
    
    # Logging parameters
    log_level: str
    log_interval: int
```

## Integration Points

### Policy Gradients
```python
class PolicyGradientEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super().__init__(config)
        self.pipeline = ExecutionPipeline(
            kernel=EvaluationKernel(config),
            config=config
        )
    
    def evaluate(self, model: nn.Module, dataset: Dataset) -> Dict[str, float]:
        """Evaluate policy gradient training."""
        return self.pipeline.execute(model, dataset)
```

## Monitoring and Validation

### Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `accuracy` | `float` | Overall accuracy |
| `f1_score` | `float` | F1 score |
| `reasoning_score` | `float` | Reasoning score |
| `eval_time` | `float` | Evaluation time |
| `kernel_time` | `float` | Kernel execution time |
| `cache_hit_rate` | `float` | Cache efficiency |
| `batch_throughput` | `float` | Processing speed |

### Validation Process
1. Kernel validation
2. Pipeline validation
3. Metric validation
4. Performance validation

## Constants

### Time Parameters
| Name | Value | Unit | Description |
|------|-------|------|-------------|
| `EVALUATION_SLOT_TIME` | 12 | seconds | Time between evaluation slots |
| `MAX_EVALUATION_EPOCHS` | 32 | epochs | Maximum number of epochs to evaluate |
| `MIN_VALIDATION_SAMPLES` | 1000 | samples | Minimum samples for validation |

### Performance Parameters
| Name | Value | Description |
|------|-------|-------------|
| `MAX_BATCH_SIZE` | 128 | Maximum batch size for parallel processing |
| `CACHE_SIZE` | 10000 | Maximum number of cached results |
| `NUM_WORKERS` | 4 | Number of parallel evaluation workers |
| `EVAL_TIMEOUT` | 30 | Timeout for evaluation in seconds |

### Evaluation Parameters
| Name | Value | Description |
|------|-------|-------------|
| `MAX_SEQUENCE_LENGTH` | 512 | Maximum sequence length for evaluation |
| `DEFAULT_BATCH_SIZE` | 32 | Default batch size for evaluation |
| `MIN_EXAMPLES` | 3 | Minimum number of examples for few-shot |

## Data Structures

### EvaluationState
```python
class EvaluationState:
    epoch: uint64
    metrics: Dict[str, float]
    validation_results: List[Dict]
    current_slot: uint64
    last_processed_slot: uint64
    cache: LRUCache[Tuple[str, str], Dict[str, float]]  # (prompt, model_id) -> metrics
    batch_queue: Queue[Batch]  # Queue for batched evaluation
    worker_pool: ThreadPoolExecutor  # Pool for parallel evaluation
```

### Batch
```python
class Batch:
    prompts: List[str]
    model_id: str
    batch_size: uint64
    priority: uint64  # Higher priority batches processed first
    timestamp: uint64
```

### PromptConfig
```python
class PromptConfig:
    format: str  # "few_shot", "zero_shot", "cot"
    num_examples: uint64
    style: str  # "chat", "qa", "reasoning"
    max_length: uint64
    cache_enabled: bool  # Whether to cache results
    parallel_processing: bool  # Whether to use parallel processing
```

## Component Definitions

### Base Components

#### BaseEvaluator
```python
class BaseEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = config.metrics
        self.state = EvaluationState()
        self.cache = LRUCache(maxsize=CACHE_SIZE)
        self.worker_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    
    def evaluate(self, model: nn.Module, dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate model on dataset with optimized batch processing.
        
        Parameters:
            model: Model to evaluate
            dataset: Evaluation dataset
            
        Returns:
            Dictionary of metrics
        """
        # Create batches
        batches = self._create_batches(dataset)
        
        # Process batches in parallel
        futures = []
        for batch in batches:
            future = self.worker_pool.submit(
                self._evaluate_batch,
                model,
                batch
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
        
        return self._aggregate_results(results)
    
    def _create_batches(self, dataset: Dataset) -> List[Batch]:
        """Create optimized batches for parallel processing."""
        batches = []
        current_batch = []
        
        for example in dataset:
            # Check cache first
            cache_key = (example["input"], model.id)
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
                continue
            
            current_batch.append(example)
            
            if len(current_batch) >= MAX_BATCH_SIZE:
                batches.append(Batch(
                    prompts=current_batch,
                    model_id=model.id,
                    batch_size=len(current_batch),
                    priority=1,
                    timestamp=time.time()
                ))
                current_batch = []
        
        if current_batch:
            batches.append(Batch(
                prompts=current_batch,
                model_id=model.id,
                batch_size=len(current_batch),
                priority=1,
                timestamp=time.time()
            ))
        
        return batches
    
    def _evaluate_batch(self, model: nn.Module, batch: Batch) -> List[Dict]:
        """Evaluate a single batch with timeout protection."""
        try:
            with timeout(EVAL_TIMEOUT):
                results = []
                for prompt in batch.prompts:
                    response = model.generate(prompt)
                    metrics = self.compute_metrics(response, prompt["reference"])
                    
                    # Cache results
                    cache_key = (prompt["input"], model.id)
                    self.cache[cache_key] = metrics
                    
                    results.append(metrics)
                return results
        except TimeoutError:
            logger.warning(f"Batch evaluation timed out after {EVAL_TIMEOUT}s")
            return []
```

#### BasePrompt
```python
class BasePrompt:
    def __init__(self, config: PromptConfig):
        self.config = config
    
    def format_prompt(self, input_text: str, examples: List[Dict] = None) -> str:
        """
        Format prompt with optional examples.
        
        Parameters:
            input_text: Input text
            examples: Optional examples for few-shot
            
        Returns:
            Formatted prompt
        """
        raise NotImplementedError
```

### Evaluation Components

#### ChatEvaluator
```python
class ChatEvaluator(BaseEvaluator):
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.prompt_format = config.prompt_format
    
    def evaluate(self, model: nn.Module, dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate chat performance with optimized batch processing.
        
        Parameters:
            model: Model to evaluate
            dataset: Evaluation dataset
            
        Returns:
            Dictionary of metrics
        """
        # Pre-process prompts in parallel
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            prompts = list(executor.map(
                self.prompt_format.format_prompt,
                [example["input"] for example in dataset]
            ))
        
        # Create batches with pre-processed prompts
        batches = self._create_batches_with_prompts(prompts, dataset)
        
        # Process batches
        results = []
        for batch in batches:
            batch_results = self._evaluate_batch(model, batch)
            results.extend(batch_results)
        
        return self._aggregate_results(results)
```

### Prompt Components

#### ChainOfThoughtPrompt
```python
class ChainOfThoughtPrompt(BasePrompt):
    def __init__(self, config: PromptConfig):
        super().__init__(config)
        self.num_examples = config.num_examples
    
    def format_prompt(self, input_text: str, examples: List[Dict] = None) -> str:
        """
        Format prompt with CoT examples.
        
        Parameters:
            input_text: Input text
            examples: Optional examples for few-shot
            
        Returns:
            Formatted prompt
        """
        prompt = []
        
        if examples:
            for example in examples[:self.num_examples]:
                prompt.append(
                    f"Input: {example['input']}\n"
                    f"Reasoning: {example['reasoning']}\n"
                    f"Output: {example['output']}\n"
                )
        
        prompt.append(f"Input: {input_text}\nReasoning:")
        return "\n".join(prompt)
```

## Helper Functions

### get_evaluation_metrics
```python
def get_evaluation_metrics(state: EvaluationState) -> Dict[str, float]:
    """
    Get current evaluation metrics with caching.
    
    Parameters:
        state: Current evaluation state
        
    Returns:
        Dictionary of metrics
    """
    # Check cache first
    cache_key = f"metrics_{state.epoch}"
    if cache_key in state.cache:
        return state.cache[cache_key]
    
    metrics = state.metrics
    state.cache[cache_key] = metrics
    return metrics
```

### is_valid_evaluation
```python
def is_valid_evaluation(state: EvaluationState) -> bool:
    """
    Check if evaluation is valid with performance checks.
    
    Parameters:
        state: Current evaluation state
        
    Returns:
        True if evaluation is valid
    """
    return (
        len(state.validation_results) >= MIN_VALIDATION_SAMPLES and
        state.current_slot > state.last_processed_slot and
        not state.worker_pool._shutdown and
        len(state.batch_queue) < MAX_BATCH_SIZE * 2
    )
```

## Configuration Parameters

### Evaluation Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | `List[str]` | List of metrics to compute |
| `batch_size` | `uint64` | Batch size for evaluation |
| `max_length` | `uint64` | Maximum sequence length |
| `cache_enabled` | `bool` | Whether to enable caching |
| `parallel_processing` | `bool` | Whether to use parallel processing |

### Prompt Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `num_examples` | `uint64` | Number of examples |
| `format` | `str` | Prompt format |
| `style` | `str` | Prompt style |
| `cache_enabled` | `bool` | Whether to cache prompts |

## Integration Points

### Policy Gradients
```python
def evaluate_policy_gradients(
    model: nn.Module,
    evaluator: BaseEvaluator,
    dataset: Dataset
) -> Dict[str, float]:
    """
    Evaluate policy gradient training with optimized batch processing.
    
    Parameters:
        model: Trained model
        evaluator: Evaluation component
        dataset: Evaluation dataset
        
    Returns:
        Dictionary of metrics
    """
    # Enable parallel processing for policy gradients
    evaluator.config.parallel_processing = True
    return evaluator.evaluate(model, dataset)
```

### Direct Alignment
```python
def evaluate_direct_alignment(
    model: nn.Module,
    evaluator: BaseEvaluator,
    dataset: Dataset
) -> Dict[str, float]:
    """
    Evaluate direct alignment training.
    
    Parameters:
        model: Trained model
        evaluator: Evaluation component
        dataset: Evaluation dataset
        
    Returns:
        Dictionary of metrics
    """
    return evaluator.evaluate(model, dataset)
```

## Monitoring and Validation

### Key Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `accuracy` | `float` | Overall accuracy |
| `f1_score` | `float` | F1 score for classification |
| `reasoning_score` | `float` | Score for reasoning tasks |
| `chat_score` | `float` | Score for chat performance |
| `eval_time` | `float` | Average evaluation time per batch |
| `cache_hit_rate` | `float` | Cache hit rate |
| `batch_throughput` | `float` | Batches processed per second |

### Validation Process
1. Metric computation with caching
2. Result aggregation with parallel processing
3. Performance analysis with timing metrics
4. Error analysis with timeout protection 