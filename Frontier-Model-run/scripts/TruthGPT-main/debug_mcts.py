"""Debug script to isolate the MCTS type error."""

import random
import traceback
from optimization_core.enhanced_mcts_optimizer import create_enhanced_mcts_with_benchmarks

def mock_objective(config):
    return random.uniform(0.1, 1.0)

def debug_mcts():
    print("Creating enhanced MCTS with benchmarks...")
    
    try:
        optimizer = create_enhanced_mcts_with_benchmarks(mock_objective, 'deepseek_v3')
        print(f"Optimizer created successfully")
        print(f"Args type: {type(optimizer.args)}")
        print(f"Performance weight: {optimizer.args.performance_weight} (type: {type(optimizer.args.performance_weight)})")
        print(f"Reasoning weight: {optimizer.args.reasoning_weight} (type: {type(optimizer.args.reasoning_weight)})")
        
        optimizer.args.mcts_args.fe_max = 3
        optimizer.args.mcts_args.init_size = 1
        optimizer.args.benchmark_config.problems_per_category = 1
        
        print("Starting optimization...")
        best_config, best_score, stats = optimizer.optimize_with_benchmarks()
        print(f"SUCCESS: Optimization completed with score {best_score}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_mcts()
