"""
Mathematical Optimization for TruthGPT

High-performance optimization utilities with support for:
- Hyperparameter optimization (random, grid, bayesian)
- Loss functions (cross-entropy, focal loss)
- Learning rate scheduling (cosine, warmup)
- Gradient operations (clipping, accumulation)
"""

using LinearAlgebra
using Random
using Statistics

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default hyperparameter ranges
const DEFAULT_LR_MIN = 1e-6
const DEFAULT_LR_MAX = 1e-2
const DEFAULT_BATCH_MIN = 8
const DEFAULT_BATCH_MAX = 128
const DEFAULT_DROPOUT_MIN = 0.0
const DEFAULT_DROPOUT_MAX = 0.5
const DEFAULT_WARMUP_MIN = 100
const DEFAULT_WARMUP_MAX = 2000

# Bayesian optimization parameters
const DEFAULT_BAYESIAN_EXPLORATION_DECAY = 0.01
const DEFAULT_BAYESIAN_MIN_EXPLORATION = 0.1

# Focal loss defaults
const DEFAULT_FOCAL_GAMMA = 2.0
const DEFAULT_FOCAL_ALPHA = 0.25

# Numerical stability constants
const EPS_FLOAT32 = 1e-5f0
const EPS_FLOAT64 = 1e-8

# Optimization method constants
const METHOD_BAYESIAN = :bayesian
const METHOD_RANDOM = :random
const METHOD_GRID = :grid

# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS FOR HyperparamBounds
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_lr_bounds(lr_min, lr_max)

Validate learning rate bounds.

# Arguments
- `lr_min`: Minimum learning rate
- `lr_max`: Maximum learning rate

# Throws
- `ArgumentError` if bounds are invalid
"""
function validate_lr_bounds(lr_min::Float64, lr_max::Float64)
    if lr_min <= 0.0 || lr_max <= 0.0
        throw(ArgumentError("Learning rate bounds must be positive"))
    end
    if lr_min >= lr_max
        throw(ArgumentError("lr_min ($lr_min) must be < lr_max ($lr_max)"))
    end
end

"""
    validate_batch_bounds(batch_min, batch_max)

Validate batch size bounds.

# Arguments
- `batch_min`: Minimum batch size
- `batch_max`: Maximum batch size

# Throws
- `ArgumentError` if bounds are invalid
"""
function validate_batch_bounds(batch_min::Int, batch_max::Int)
    if batch_min <= 0 || batch_max <= 0
        throw(ArgumentError("Batch size bounds must be positive"))
    end
    if batch_min > batch_max
        throw(ArgumentError("batch_min ($batch_min) must be <= batch_max ($batch_max)"))
    end
end

"""
    validate_dropout_bounds(dropout_min, dropout_max)

Validate dropout probability bounds.

# Arguments
- `dropout_min`: Minimum dropout probability
- `dropout_max`: Maximum dropout probability

# Throws
- `ArgumentError` if bounds are invalid
"""
function validate_dropout_bounds(dropout_min::Float64, dropout_max::Float64)
    if dropout_min < 0.0 || dropout_max > 1.0
        throw(ArgumentError("Dropout bounds must be in [0, 1]"))
    end
    if dropout_min > dropout_max
        throw(ArgumentError("dropout_min ($dropout_min) must be <= dropout_max ($dropout_max)"))
    end
end

"""
    validate_warmup_bounds(warmup_min, warmup_max)

Validate warmup steps bounds.

# Arguments
- `warmup_min`: Minimum warmup steps
- `warmup_max`: Maximum warmup steps

# Throws
- `ArgumentError` if bounds are invalid
"""
function validate_warmup_bounds(warmup_min::Int, warmup_max::Int)
    if warmup_min <= 0 || warmup_max <= 0
        throw(ArgumentError("Warmup bounds must be positive"))
    end
    if warmup_min > warmup_max
        throw(ArgumentError("warmup_min ($warmup_min) must be <= warmup_max ($warmup_max)"))
    end
end

"""
    HyperparamBounds

Bounds for hyperparameter optimization with validation.

# Fields
- `lr_min`, `lr_max`: Learning rate bounds (log scale recommended)
- `batch_min`, `batch_max`: Batch size bounds (integers)
- `dropout_min`, `dropout_max`: Dropout probability bounds [0, 1]
- `warmup_min`, `warmup_max`: Warmup steps bounds (integers)

# Examples
```julia
bounds = HyperparamBounds(
    lr_range=(1e-5, 1e-3),
    batch_range=(16, 64)
)
```
"""
struct HyperparamBounds
    lr_min::Float64
    lr_max::Float64
    batch_min::Int
    batch_max::Int
    dropout_min::Float64
    dropout_max::Float64
    warmup_min::Int
    warmup_max::Int
    
    function HyperparamBounds(
        lr_min::Float64,
        lr_max::Float64,
        batch_min::Int,
        batch_max::Int,
        dropout_min::Float64,
        dropout_max::Float64,
        warmup_min::Int,
        warmup_max::Int
    )
        # Validate all bounds using helper functions
        validate_lr_bounds(lr_min, lr_max)
        validate_batch_bounds(batch_min, batch_max)
        validate_dropout_bounds(dropout_min, dropout_max)
        validate_warmup_bounds(warmup_min, warmup_max)
        
        new(lr_min, lr_max, batch_min, batch_max, dropout_min, dropout_max, warmup_min, warmup_max)
    end
end

"""
    HyperparamBounds(; kwargs...)

Create HyperparamBounds with keyword arguments and default ranges.

# Arguments
- `lr_range`: Tuple of (min, max) learning rates (default: (1e-6, 1e-2))
- `batch_range`: Tuple of (min, max) batch sizes (default: (8, 128))
- `dropout_range`: Tuple of (min, max) dropout probabilities (default: (0.0, 0.5))
- `warmup_range`: Tuple of (min, max) warmup steps (default: (100, 2000))

# Examples
```julia
bounds = HyperparamBounds(
    lr_range=(1e-5, 1e-3),
    batch_range=(16, 64)
)
```
"""
function HyperparamBounds(;
    lr_range::Tuple{Float64, Float64} = (DEFAULT_LR_MIN, DEFAULT_LR_MAX),
    batch_range::Tuple{Int, Int} = (DEFAULT_BATCH_MIN, DEFAULT_BATCH_MAX),
    dropout_range::Tuple{Float64, Float64} = (DEFAULT_DROPOUT_MIN, DEFAULT_DROPOUT_MAX),
    warmup_range::Tuple{Int, Int} = (DEFAULT_WARMUP_MIN, DEFAULT_WARMUP_MAX)
)
    HyperparamBounds(
        lr_range[1], lr_range[2],
        batch_range[1], batch_range[2],
        dropout_range[1], dropout_range[2],
        warmup_range[1], warmup_range[2]
    )
end

"""
    OptimizationResult

Result from hyperparameter optimization.

# Fields
- `best_params`: Dictionary of best hyperparameters found
- `best_loss`: Best loss value achieved
- `iterations`: Total number of iterations performed
- `history`: Vector of loss values at each iteration

# Examples
```julia
result = optimize_hyperparams(loss_fn, bounds)
println("Best loss: ", result.best_loss)
println("Best params: ", result.best_params)
```
"""
struct OptimizationResult
    best_params::Dict{Symbol, Any}
    best_loss::Float64
    iterations::Int
    history::Vector{Float64}
end

"""
    validate_optimization_inputs(loss_fn, bounds, method, max_iters)

Validate inputs for hyperparameter optimization.

# Arguments
- `loss_fn`: Loss function to validate
- `bounds`: HyperparamBounds to validate
- `method`: Optimization method
- `max_iters`: Maximum iterations

# Throws
- `ArgumentError` if any input is invalid
"""
function validate_optimization_inputs(
    loss_fn::Function,
    bounds::HyperparamBounds,
    method::Symbol,
    max_iters::Int
)
    if max_iters <= 0
        throw(ArgumentError("max_iters must be positive, got $max_iters"))
    end
    
    valid_methods = [METHOD_BAYESIAN, METHOD_RANDOM, METHOD_GRID]
    if method ∉ valid_methods
        throw(ArgumentError(
            "method must be one of $valid_methods, got :$method"
        ))
    end
end

"""
    optimize_hyperparams(loss_fn, bounds; method=:bayesian, max_iters=100, seed=42)

Optimize hyperparameters using specified method.

# Arguments
- `loss_fn`: Function that takes params dict and returns loss (must return Float64)
- `bounds`: HyperparamBounds struct defining search space
- `method`: Optimization method - `:bayesian`, `:random`, or `:grid` (default: :bayesian)
- `max_iters`: Maximum optimization iterations (default: 100)
- `seed`: Random seed for reproducibility (default: 42)

# Returns
- `OptimizationResult` with best parameters and optimization history

# Examples
```julia
# Define loss function
loss_fn(p) = train_and_eval(lr=p[:lr], dropout=p[:dropout])

# Create bounds
bounds = HyperparamBounds(lr_range=(1e-5, 1e-3))

# Optimize
result = optimize_hyperparams(loss_fn, bounds, method=:random, max_iters=50)
println("Best loss: ", result.best_loss)
println("Best params: ", result.best_params)
```

# Performance
- Random: Fast, good for initial exploration
- Grid: Systematic, good for small search spaces
- Bayesian: Efficient, good for expensive evaluations
"""
function optimize_hyperparams(
    loss_fn::Function,
    bounds::HyperparamBounds;
    method::Symbol = METHOD_BAYESIAN,
    max_iters::Int = 100,
    seed::Int = 42
)
    # Validate inputs
    validate_optimization_inputs(loss_fn, bounds, method, max_iters)
    
    # Initialize random state for reproducibility
    rng = MersenneTwister(seed)
    
    # Track best results
    best_loss = Inf
    best_params = Dict{Symbol, Any}()
    history = Vector{Float64}(undef, max_iters)
    
    # Optimization loop
    for iteration in 1:max_iters
        # Sample hyperparameters based on method
        params = sample_params(bounds, method, iteration, max_iters, rng)
        
        try
            # Evaluate loss function
            loss = loss_fn(params)
            
            # Validate loss is finite
            if !isfinite(loss)
                @warn "Iteration $iteration produced non-finite loss: $loss"
                history[iteration] = Inf
                continue
            end
            
            history[iteration] = loss
            
            # Update best if improved
            if loss < best_loss
                best_loss = loss
                best_params = copy(params)  # Deep copy to avoid mutation
                @info "New best at iteration $iteration: loss=$(round(loss, digits=6))"
            end
        catch e
            @warn "Iteration $iteration failed: $e"
            history[iteration] = Inf
        end
    end
    
    # Check if we found any valid results
    if best_loss == Inf
        throw(ErrorException("Optimization failed: no valid loss values found"))
    end
    
    return OptimizationResult(best_params, best_loss, max_iters, history)
end

"""
    sample_params(bounds, method, iter, max_iters, rng)

Sample hyperparameters based on optimization method.

# Arguments
- `bounds`: HyperparamBounds struct
- `method`: Sampling method (`:random`, `:grid`, or `:bayesian`)
- `iter`: Current iteration number (1-based)
- `max_iters`: Maximum iterations
- `rng`: Random number generator

# Returns
- Dictionary of sampled hyperparameters with keys:
  - `:lr`: Learning rate
  - `:batch_size`: Batch size
  - `:dropout`: Dropout probability
  - `:warmup_steps`: Warmup steps
"""
function sample_params(
    bounds::HyperparamBounds,
    method::Symbol,
    iter::Int,
    max_iters::Int,
    rng::AbstractRNG
)
    if method == METHOD_RANDOM
        return random_sample(bounds, rng)
    elseif method == METHOD_GRID
        return grid_sample(bounds, iter, max_iters)
    else  # METHOD_BAYESIAN
        return bayesian_sample(bounds, iter, rng)
    end
end

"""
    random_sample(bounds, rng)

Sample hyperparameters uniformly at random from bounds.

Uses log-uniform sampling for learning rate (common practice for wide ranges).

# Arguments
- `bounds`: HyperparamBounds struct
- `rng`: Random number generator

# Returns
- Dictionary of sampled hyperparameters

# Algorithm
- Learning rate: Log-uniform sampling (better for wide ranges)
- Batch size: Uniform integer sampling
- Dropout: Uniform continuous sampling
- Warmup: Uniform integer sampling
"""
function random_sample(bounds::HyperparamBounds, rng::AbstractRNG)
    # Sample each hyperparameter using appropriate method
    lr = sample_log_uniform_lr(bounds.lr_min, bounds.lr_max, rng)
    batch_size = sample_uniform_int(bounds.batch_min, bounds.batch_max, rng)
    dropout = sample_uniform_float(bounds.dropout_min, bounds.dropout_max, rng)
    warmup_steps = sample_uniform_int(bounds.warmup_min, bounds.warmup_max, rng)
    
    return Dict{Symbol, Any}(
        :lr => lr,
        :batch_size => batch_size,
        :dropout => dropout,
        :warmup_steps => warmup_steps
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    sample_log_uniform_lr(lr_min, lr_max, rng)

Sample learning rate using log-uniform distribution.

# Arguments
- `lr_min`: Minimum learning rate
- `lr_max`: Maximum learning rate
- `rng`: Random number generator

# Returns
- Sampled learning rate

# Notes
- Log-uniform sampling gives equal probability to each order of magnitude
- Better for wide ranges (e.g., 1e-6 to 1e-2)
"""
function sample_log_uniform_lr(lr_min::Float64, lr_max::Float64, rng::AbstractRNG)
    log_lr_min = log(lr_min)
    log_lr_max = log(lr_max)
    return exp(log_lr_min + rand(rng) * (log_lr_max - log_lr_min))
end

"""
    sample_uniform_int(min_val, max_val, rng)

Sample integer uniformly from [min_val, max_val].

# Arguments
- `min_val`: Minimum value (inclusive)
- `max_val`: Maximum value (inclusive)
- `rng`: Random number generator

# Returns
- Sampled integer
"""
function sample_uniform_int(min_val::Int, max_val::Int, rng::AbstractRNG)
    return rand(rng, min_val:max_val)
end

"""
    sample_uniform_float(min_val, max_val, rng)

Sample float uniformly from [min_val, max_val).

# Arguments
- `min_val`: Minimum value (inclusive)
- `max_val`: Maximum value (exclusive)
- `rng`: Random number generator

# Returns
- Sampled float
"""
function sample_uniform_float(min_val::Float64, max_val::Float64, rng::AbstractRNG)
    return min_val + rand(rng) * (max_val - min_val)
end

"""
    grid_sample(bounds, iter, max_iters)

Sample hyperparameters using grid search (linear progression).

# Arguments
- `bounds`: HyperparamBounds struct
- `iter`: Current iteration (1-based)
- `max_iters`: Total iterations

# Returns
- Dictionary of sampled hyperparameters

# Algorithm
- Progresses linearly through parameter space
- Uses log-linear interpolation for learning rate
- Linear interpolation for other parameters
"""
function grid_sample(bounds::HyperparamBounds, iter::Int, max_iters::Int)
    # Normalized progress [0, 1]
    progress = compute_grid_progress(iter, max_iters)
    
    # Sample each hyperparameter using appropriate interpolation
    lr = interpolate_log_linear(bounds.lr_min, bounds.lr_max, progress)
    batch_size = round(Int, interpolate_linear(bounds.batch_min, bounds.batch_max, progress))
    dropout = interpolate_linear(bounds.dropout_min, bounds.dropout_max, progress)
    warmup_steps = round(Int, interpolate_linear(bounds.warmup_min, bounds.warmup_max, progress))
    
    return Dict{Symbol, Any}(
        :lr => lr,
        :batch_size => batch_size,
        :dropout => dropout,
        :warmup_steps => warmup_steps
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# INTERPOLATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_grid_progress(iter, max_iters)

Compute normalized progress for grid search.

# Arguments
- `iter`: Current iteration (1-based)
- `max_iters`: Total iterations

# Returns
- Normalized progress in [0, 1]
"""
function compute_grid_progress(iter::Int, max_iters::Int)
    return (iter - 1) / max(max_iters - 1, 1)
end

"""
    interpolate_log_linear(min_val, max_val, progress)

Interpolate value using log-linear interpolation.

# Arguments
- `min_val`: Minimum value
- `max_val`: Maximum value
- `progress`: Progress in [0, 1]

# Returns
- Interpolated value

# Notes
- Better for wide ranges (e.g., learning rates)
- Uses exponential interpolation in log space
"""
function interpolate_log_linear(min_val::Float64, max_val::Float64, progress::Float64)
    log_min = log(min_val)
    log_max = log(max_val)
    return exp(log_min + progress * (log_max - log_min))
end

"""
    interpolate_linear(min_val, max_val, progress)

Interpolate value using linear interpolation.

# Arguments
- `min_val`: Minimum value
- `max_val`: Maximum value
- `progress`: Progress in [0, 1]

# Returns
- Interpolated value
"""
function interpolate_linear(min_val::Real, max_val::Real, progress::Float64)
    return min_val + progress * (max_val - min_val)
end

"""
    bayesian_sample(bounds, iter, rng)

Sample hyperparameters using simplified Bayesian optimization.

Uses exploration-exploitation tradeoff: starts with more exploration,
gradually shifts to exploitation. Currently uses random sampling with
decaying exploration probability (simplified implementation).

# Arguments
- `bounds`: HyperparamBounds struct
- `iter`: Current iteration number
- `rng`: Random number generator

# Returns
- Dictionary of sampled hyperparameters

# Notes
A full Bayesian optimization would use Gaussian Processes or Tree-structured
Parzen Estimators. This is a simplified version for demonstration.

# Algorithm
- Exploration probability decays over iterations
- With high probability: explore (random sample)
- With low probability: exploit (use best known region - simplified to random for now)
"""
function bayesian_sample(bounds::HyperparamBounds, iter::Int, rng::AbstractRNG)
    # Exploration probability decays over iterations
    # Starts at 1.0, decays to DEFAULT_BAYESIAN_MIN_EXPLORATION
    exploration_prob = max(
        DEFAULT_BAYESIAN_MIN_EXPLORATION,
        1.0 - iter * DEFAULT_BAYESIAN_EXPLORATION_DECAY
    )
    
    # With probability `exploration_prob`, explore (random sample)
    # Otherwise, exploit (use best known region - simplified to random for now)
    # In full implementation, would sample from promising regions based on history
    if rand(rng) < exploration_prob
        return random_sample(bounds, rng)
    else
        # In full implementation, would sample from promising regions
        # For now, use random sampling as placeholder
        return random_sample(bounds, rng)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    cross_entropy(logits, targets)

Compute numerically stable cross-entropy loss.

Uses log-sum-exp trick for numerical stability to prevent overflow/underflow.

# Arguments
- `logits`: Logit scores [n_classes, batch_size]
- `targets`: Target class indices [batch_size] (1-based, Julia convention)

# Returns
- Average cross-entropy loss (scalar)

# Examples
```julia
logits = randn(Float32, 10, 32)  # 10 classes, batch size 32
targets = rand(1:10, 32)
loss = cross_entropy(logits, targets)
```

# Algorithm
1. Compute log-sum-exp: log(sum(exp(logits))) using max trick
2. Compute log probabilities: logits - log_sum_exp
3. Extract log probability of target class
4. Average negative log probabilities
"""
function cross_entropy(logits::AbstractMatrix{T}, targets::AbstractVector{<:Integer}) where T
    n_classes, batch_size = size(logits)
    
    # Validate inputs
    if length(targets) != batch_size
        throw(DimensionMismatch(
            "targets length $(length(targets)) must match batch_size $batch_size"
        ))
    end
    if any(t -> t < 1 || t > n_classes, targets)
        throw(ArgumentError("Target indices must be in [1, $n_classes]"))
    end
    
    # Numerically stable computation using log-sum-exp trick
    # log(softmax) = logits - log(sum(exp(logits)))
    # We compute: logits - max_logits - log(sum(exp(logits - max_logits)))
    max_logits = maximum(logits, dims=1)
    shifted = logits .- max_logits
    log_sum_exp = log.(sum(exp.(shifted), dims=1) .+ eps(T))
    log_probs = shifted .- log_sum_exp
    
    # Compute negative log-likelihood for target classes
    loss = zero(T)
    @inbounds @simd for i in 1:batch_size
        loss -= log_probs[targets[i], i]
    end
    
    return loss / batch_size
end

"""
    focal_loss(logits, targets; gamma=2.0, alpha=0.25)

Compute focal loss for imbalanced classification.

Focal loss downweights easy examples, focusing learning on hard examples.
Useful when classes are imbalanced.

# Arguments
- `logits`: Logit scores [n_classes, batch_size]
- `targets`: Target class indices [batch_size] (1-based)
- `gamma`: Focusing parameter (higher = more focus on hard examples, default: 2.0)
- `alpha`: Class weighting factor (default: 0.25)

# Returns
- Average focal loss (scalar)

# Examples
```julia
logits = randn(Float32, 10, 32)
targets = rand(1:10, 32)
loss = focal_loss(logits, targets, gamma=2.0, alpha=0.25)
```

# References
Lin et al. "Focal Loss for Dense Object Detection" (2017)

# Algorithm
Focal loss = -alpha * (1 - p)^gamma * log(p)
where p is the predicted probability of the target class.
"""
function focal_loss(
    logits::AbstractMatrix{T},
    targets::AbstractVector{<:Integer};
    gamma::T = T(DEFAULT_FOCAL_GAMMA),
    alpha::T = T(DEFAULT_FOCAL_ALPHA)
) where T
    n_classes, batch_size = size(logits)
    
    # Validate inputs
    if length(targets) != batch_size
        throw(DimensionMismatch(
            "targets length $(length(targets)) must match batch_size $batch_size"
        ))
    end
    if any(t -> t < 1 || t > n_classes, targets)
        throw(ArgumentError("Target indices must be in [1, $n_classes]"))
    end
    if gamma < 0.0
        throw(ArgumentError("gamma must be non-negative, got $gamma"))
    end
    if alpha < 0.0 || alpha > 1.0
        throw(ArgumentError("alpha must be in [0, 1], got $alpha"))
    end
    
    # Compute probabilities using numerically stable softmax
    probs = softmax(logits)
    
    # Compute focal loss: -alpha * (1 - p)^gamma * log(p)
    loss = zero(T)
    eps_val = eps(T)
    
    @inbounds @simd for i in 1:batch_size
        p = probs[targets[i], i]
        # Clamp probability to avoid numerical issues (log(0) or log(1))
        p = max(eps_val, min(1.0 - eps_val, p))
        
        # Focal weight: (1 - p)^gamma (higher for hard examples)
        focal_weight = (1.0 - p)^gamma
        
        # Focal loss component
        loss -= alpha * focal_weight * log(p)
    end
    
    return loss / batch_size
end

"""
    softmax(x; dims=1)

Compute numerically stable softmax along specified dimension.

# Arguments
- `x`: Input array
- `dims`: Dimension along which to compute softmax (default: 1)

# Returns
- Softmax probabilities (same shape as input, sums to 1 along dims)

# Examples
```julia
logits = randn(Float32, 10, 32)
probs = softmax(logits, dims=1)  # Softmax over classes
```

# Algorithm
1. Subtract max for numerical stability: x - max(x)
2. Compute exp: exp(x - max)
3. Normalize: exp / sum(exp)
"""
function softmax(x::AbstractArray{T}; dims=1) where T
    # Numerically stable: subtract max before exp to prevent overflow
    max_x = maximum(x, dims=dims)
    exp_x = exp.(x .- max_x)
    
    # Normalize by sum (add eps to prevent division by zero)
    exp_x ./ (sum(exp_x, dims=dims) .+ eps(T))
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    cosine_schedule(step, total_steps, lr_max, lr_min=0)

Cosine annealing learning rate schedule.

# Arguments
- `step`: Current training step
- `total_steps`: Total number of training steps
- `lr_max`: Maximum learning rate
- `lr_min`: Minimum learning rate (default: 0)

# Returns
- Learning rate at current step

# Examples
```julia
lr = cosine_schedule(100, 1000, 0.01, 0.001)
```

# Algorithm
lr = lr_min + (lr_max - lr_min) * (1 + cos(π * step / total_steps)) / 2
"""
function cosine_schedule(step::Int, total_steps::Int, lr_max::T, lr_min::T=zero(T)) where T
    if step >= total_steps
        return lr_min
    end
    if total_steps <= 0
        throw(ArgumentError("total_steps must be positive, got $total_steps"))
    end
    
    # Cosine annealing: smoothly decay from lr_max to lr_min
    cosine_factor = (1.0 + cos(π * step / total_steps)) / 2.0
    return lr_min + (lr_max - lr_min) * cosine_factor
end

"""
    warmup_cosine_schedule(step, warmup_steps, total_steps, lr_max)

Warmup + cosine decay schedule.

Combines linear warmup with cosine annealing for better training stability.

# Arguments
- `step`: Current training step
- `warmup_steps`: Number of warmup steps
- `total_steps`: Total number of training steps
- `lr_max`: Maximum learning rate (reached after warmup)

# Returns
- Learning rate at current step

# Examples
```julia
lr = warmup_cosine_schedule(50, 100, 1000, 0.01)
```

# Algorithm
- If step < warmup_steps: linear warmup from 0 to lr_max
- Otherwise: cosine decay from lr_max to 0
"""
function warmup_cosine_schedule(
    step::Int,
    warmup_steps::Int,
    total_steps::Int,
    lr_max::T
) where T
    if warmup_steps <= 0
        throw(ArgumentError("warmup_steps must be positive, got $warmup_steps"))
    end
    if total_steps <= warmup_steps
        throw(ArgumentError("total_steps ($total_steps) must be > warmup_steps ($warmup_steps)"))
    end
    
    if step < warmup_steps
        # Linear warmup: gradually increase from 0 to lr_max
        return lr_max * step / warmup_steps
    else
        # Cosine decay: smoothly decrease from lr_max to 0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_factor = (1.0 + cos(π * progress)) / 2.0
        return lr_max * cosine_factor
    end
end

"""
    linear_warmup(step, warmup_steps, lr_max)

Linear warmup schedule.

# Arguments
- `step`: Current training step
- `warmup_steps`: Number of warmup steps
- `lr_max`: Maximum learning rate (reached after warmup)

# Returns
- Learning rate at current step (clamped to lr_max)

# Examples
```julia
lr = linear_warmup(50, 100, 0.01)  # Returns 0.005
```
"""
function linear_warmup(step::Int, warmup_steps::Int, lr_max::T) where T
    if warmup_steps <= 0
        throw(ArgumentError("warmup_steps must be positive, got $warmup_steps"))
    end
    
    # Linear warmup: lr = lr_max * step / warmup_steps
    # Clamp to lr_max to handle steps beyond warmup
    return min(lr_max, lr_max * step / warmup_steps)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GRADIENT OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    clip_grad_norm!(grads, max_norm)

Clip gradient norm in-place to prevent exploding gradients.

# Arguments
- `grads`: Array of gradient arrays (e.g., from Flux.params)
- `max_norm`: Maximum allowed gradient norm

# Returns
- Total gradient norm before clipping

# Examples
```julia
grads = [randn(10, 10), randn(5, 5)]
norm = clip_grad_norm!(grads, 1.0)
println("Gradient norm: $norm")
```

# Algorithm
1. Compute L2 norm across all gradients
2. If norm > max_norm: scale all gradients by max_norm / norm
3. Otherwise: leave gradients unchanged
"""
function clip_grad_norm!(grads::AbstractArray, max_norm::Real)
    if max_norm <= 0.0
        throw(ArgumentError("max_norm must be positive, got $max_norm"))
    end
    
    # Compute total L2 norm across all gradients
    total_norm_sq = zero(typeof(max_norm))
    
    for g in grads
        if g isa AbstractArray
            total_norm_sq += sum(abs2, g)
        end
    end
    
    total_norm = sqrt(total_norm_sq)
    
    # Clip if norm exceeds threshold
    if total_norm > max_norm
        scale = max_norm / (total_norm + eps(typeof(max_norm)))
        for g in grads
            if g isa AbstractArray
                g .*= scale
            end
        end
    end
    
    return total_norm
end

"""
    gradient_accumulate!(accum, grads, scale=1)

Accumulate scaled gradients into accumulator.

# Arguments
- `accum`: Accumulator array (modified in-place)
- `grads`: Gradient arrays to accumulate
- `scale`: Scaling factor for gradients (default: 1)

# Returns
- Nothing (modifies accum in-place)

# Examples
```julia
accum = [zeros(10, 10), zeros(5, 5)]
grads = [randn(10, 10), randn(5, 5)]
gradient_accumulate!(accum, grads, scale=0.5)
```

# Notes
- Useful for gradient accumulation across multiple batches
- Accumulates: accum += scale * grads
"""
function gradient_accumulate!(accum::AbstractArray, grads::AbstractArray, scale::Real=1)
    if length(accum) != length(grads)
        throw(DimensionMismatch(
            "accum and grads must have same length, got $(length(accum)) and $(length(grads))"
        ))
    end
    
    @inbounds for i in 1:length(accum)
        if accum[i] isa AbstractArray && grads[i] isa AbstractArray
            accum[i] .+= scale .* grads[i]
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

export HyperparamBounds, OptimizationResult
export optimize_hyperparams
export cross_entropy, focal_loss, softmax
export cosine_schedule, warmup_cosine_schedule, linear_warmup
export clip_grad_norm!, gradient_accumulate!
