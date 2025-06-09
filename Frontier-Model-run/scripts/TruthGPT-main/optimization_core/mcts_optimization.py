"""
MCTS-based optimization for hyperparameter and architecture search.
Integrated from mcts.py optimization file.
"""

import copy
import json
import numpy as np
import random
import time
import math
from collections import deque
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass, field

@dataclass
class MCTSOptimizationArgs:
    """Arguments for MCTS-based optimization."""
    exploration_constant_0: float = field(default=0.1, metadata={"help": "Initial exploration constant"})
    alpha: float = field(default=0.5, metadata={"help": "Alpha parameter for UCT"})
    max_depth: int = field(default=10, metadata={"help": "Maximum search depth"})
    max_children: int = field(default=5, metadata={"help": "Maximum children per node"})
    discount_factor: float = field(default=1.0, metadata={"help": "Discount factor for rewards"})
    
    init_size: int = field(default=5, metadata={"help": "Initial population size"})
    pop_size: int = field(default=10, metadata={"help": "Population size"})
    fe_max: int = field(default=100, metadata={"help": "Maximum function evaluations"})
    
    operators: List[str] = field(default_factory=lambda: ["mutate", "crossover", "local_search"])
    operator_weights: List[int] = field(default_factory=lambda: [3, 2, 1])

class MCTSNode:
    def __init__(self, config: Dict[str, Any], objective: float, depth: int = 0, 
                 is_root: bool = False, parent: Optional['MCTSNode'] = None, 
                 visits: int = 0, Q: float = 0):
        self.config = config
        self.objective = objective
        self.parent = parent
        self.depth = depth
        self.children = []
        self.children_info = []
        self.visits = visits
        self.subtree = []
        self.Q = Q
        self.reward = -1 * objective

    def add_child(self, child_node: 'MCTSNode'):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(config={self.config}, Q={self.Q:.2f}, visits={self.visits})"

class MCTS:
    def __init__(self, root_config: Dict[str, Any], args: MCTSOptimizationArgs):
        self.args = args
        self.exploration_constant_0 = args.exploration_constant_0
        self.alpha = args.alpha
        self.max_depth = args.max_depth
        self.epsilon = 1e-10
        self.discount_factor = args.discount_factor
        self.q_min = 0
        self.q_max = -10000
        self.rank_list = []

        self.root = MCTSNode(config=root_config, objective=0, depth=0, is_root=True)

        self.critiques = []
        self.refinements = []
        self.rewards = []
        self.selected_nodes = []

    def backpropagate(self, node: MCTSNode):
        if node.Q not in self.rank_list:
            self.rank_list.append(node.Q)
            self.rank_list.sort()
        self.q_min = min(self.q_min, node.Q)
        self.q_max = max(self.q_max, node.Q)
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = parent.Q * (1 - self.discount_factor) + best_child_Q * self.discount_factor
            parent.visits += 1
            if parent.parent and parent.parent == self.root:
                parent.subtree.append(node)
            parent = parent.parent

    def uct(self, node: MCTSNode, eval_remain: float):
        self.exploration_constant = self.exploration_constant_0 * eval_remain
        if self.q_max == self.q_min:
            normalized_q = 0.5
        else:
            normalized_q = (node.Q - self.q_min) / (self.q_max - self.q_min)
        
        exploration_term = self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        )
        return normalized_q + exploration_term

    def is_fully_expanded(self, node: MCTSNode):
        return (len(node.children) >= self.args.max_children or 
                any(child.Q > node.Q for child in node.children))

class MCTSOptimizer:
    """MCTS-based optimizer for model configurations and hyperparameters."""
    
    def __init__(self, args: MCTSOptimizationArgs, objective_function):
        self.args = args
        self.objective_function = objective_function
        self.eval_times = 0
        
        self.config_space = {
            'learning_rate': [1e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            'batch_size': [8, 16, 32, 64, 128],
            'hidden_size': [256, 512, 768, 1024],
            'num_layers': [2, 4, 6, 8, 12],
            'num_heads': [4, 8, 12, 16],
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'weight_decay': [0.0, 0.01, 0.1],
            'warmup_ratio': [0.0, 0.1, 0.2]
        }

    def generate_random_config(self) -> Dict[str, Any]:
        """Generate a random configuration from the search space."""
        config = {}
        for param, values in self.config_space.items():
            config[param] = random.choice(values)
        return config

    def mutate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a configuration by changing one parameter."""
        new_config = copy.deepcopy(config)
        param = random.choice(list(self.config_space.keys()))
        new_config[param] = random.choice(self.config_space[param])
        return new_config

    def crossover_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new configuration by crossing over two parent configurations."""
        new_config = {}
        for param in self.config_space.keys():
            new_config[param] = random.choice([config1[param], config2[param]])
        return new_config

    def local_search(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform local search around a configuration."""
        new_config = copy.deepcopy(config)
        
        for param, value in config.items():
            if param in ['learning_rate', 'dropout', 'weight_decay', 'warmup_ratio']:
                values = self.config_space[param]
                current_idx = values.index(value) if value in values else 0
                
                candidates = []
                if current_idx > 0:
                    candidates.append(values[current_idx - 1])
                if current_idx < len(values) - 1:
                    candidates.append(values[current_idx + 1])
                
                if candidates:
                    new_config[param] = random.choice(candidates)
        
        return new_config

    def evaluate_config(self, config: Dict[str, Any]) -> float:
        """Evaluate a configuration using the objective function."""
        self.eval_times += 1
        try:
            return self.objective_function(config)
        except Exception as e:
            print(f"Error evaluating config {config}: {e}")
            return float('inf')

    def expand_node(self, mcts: MCTS, node: MCTSNode, operator: str) -> Optional[MCTSNode]:
        """Expand a node using the specified operator."""
        if operator == "mutate":
            new_config = self.mutate_config(node.config)
        elif operator == "crossover" and len(mcts.root.children) > 0:
            other_node = random.choice(mcts.root.children)
            new_config = self.crossover_configs(node.config, other_node.config)
        elif operator == "local_search":
            new_config = self.local_search(node.config)
        else:
            new_config = self.generate_random_config()

        objective = self.evaluate_config(new_config)
        
        if objective != float('inf'):
            new_node = MCTSNode(
                config=new_config,
                objective=objective,
                parent=node,
                depth=node.depth + 1,
                visits=1,
                Q=-1 * objective
            )
            node.add_child(new_node)
            node.children_info.append({
                'config': new_config,
                'objective': objective,
                'operator': operator
            })
            mcts.backpropagate(new_node)
            return new_node
        
        return None

    def optimize(self, initial_config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], float]:
        """Run MCTS optimization to find the best configuration."""
        if initial_config is None:
            initial_config = self.generate_random_config()
        
        mcts = MCTS(initial_config, self.args)
        
        initial_objective = self.evaluate_config(initial_config)
        mcts.root.objective = initial_objective
        mcts.root.Q = -1 * initial_objective
        
        print(f"Starting MCTS optimization with initial objective: {initial_objective:.4f}")
        
        for i in range(self.args.init_size):
            if i == 0:
                config = initial_config
                objective = initial_objective
            else:
                config = self.generate_random_config()
                objective = self.evaluate_config(config)
            
            child_node = MCTSNode(
                config=config,
                objective=objective,
                parent=mcts.root,
                depth=1,
                visits=1,
                Q=-1 * objective
            )
            mcts.root.add_child(child_node)
            mcts.root.children_info.append({
                'config': config,
                'objective': objective,
                'operator': 'init'
            })
            mcts.backpropagate(child_node)
            child_node.subtree.append(child_node)
        
        print(f"Initialized population with {len(mcts.root.children)} configurations")
        
        while self.eval_times < self.args.fe_max:
            print(f"Evaluation {self.eval_times}/{self.args.fe_max}, Best Q: {mcts.q_max:.4f}")
            
            current_node = mcts.root
            while len(current_node.children) > 0 and current_node.depth < mcts.max_depth:
                eval_remain = max(1 - self.eval_times / self.args.fe_max, 0)
                uct_scores = [mcts.uct(node, eval_remain) for node in current_node.children]
                selected_idx = uct_scores.index(max(uct_scores))
                
                if int(current_node.visits ** mcts.alpha) > len(current_node.children):
                    operator_idx = random.choices(
                        range(len(self.args.operators)),
                        weights=self.args.operator_weights,
                        k=1
                    )[0]
                    operator = self.args.operators[operator_idx]
                    
                    new_node = self.expand_node(mcts, current_node, operator)
                    if new_node:
                        print(f"Expanded with {operator}: objective={new_node.objective:.4f}")
                
                current_node = current_node.children[selected_idx]
            
            for operator, weight in zip(self.args.operators, self.args.operator_weights):
                for _ in range(weight):
                    if self.eval_times >= self.args.fe_max:
                        break
                    self.expand_node(mcts, current_node, operator)
        
        best_node = mcts.root
        best_objective = float('inf')
        
        def find_best_recursive(node):
            nonlocal best_node, best_objective
            if node.objective < best_objective:
                best_objective = node.objective
                best_node = node
            for child in node.children:
                find_best_recursive(child)
        
        find_best_recursive(mcts.root)
        
        print(f"MCTS optimization completed. Best objective: {best_objective:.4f}")
        print(f"Best configuration: {best_node.config}")
        
        return best_node.config, best_objective

@dataclass
class NeuralGuidedMCTSArgs(MCTSOptimizationArgs):
    """Enhanced MCTS arguments with neural network guidance."""
    use_neural_guidance: bool = field(default=True, metadata={"help": "Enable neural network guidance"})
    entropy_weight: float = field(default=0.1, metadata={"help": "Weight for entropy-guided exploration"})
    pruning_threshold: float = field(default=0.01, metadata={"help": "Threshold for move pruning"})
    time_management: bool = field(default=True, metadata={"help": "Enable adaptive time management"})
    policy_temperature: float = field(default=1.0, metadata={"help": "Temperature for policy softmax"})
    neural_guidance_weight: float = field(default=0.3, metadata={"help": "Weight for neural guidance in selection"})

class NeuralGuidedMCTS(MCTS):
    """Enhanced MCTS with neural network guidance and advanced pruning."""
    
    def __init__(self, root_config: Dict[str, Any], args: NeuralGuidedMCTSArgs, 
                 policy_network=None, value_network=None):
        super().__init__(root_config, args)
        self.args = args
        self.policy_network = policy_network
        self.value_network = value_network
        self.move_counts = {}
        self.entropy_history = []
        self.pruned_moves = 0
        self.total_evaluations = 0
    
    def get_policy_priors(self, node: MCTSNode) -> List[float]:
        """Get policy priors from neural network."""
        if self.policy_network is None:
            return [1.0] * len(node.children)
        
        try:
            config_tensor = self._config_to_tensor(node.config)
            with torch.no_grad():
                policy_logits = self.policy_network(config_tensor)
                policy_probs = torch.softmax(policy_logits / self.args.policy_temperature, dim=-1)
            
            probs_list = policy_probs.cpu().numpy().flatten().tolist()
            
            num_children = len(node.children)
            if len(probs_list) >= num_children:
                return [float(p) for p in probs_list[:num_children]]
            else:
                uniform_prob = 1.0 / num_children
                return [float(p) for p in probs_list] + [uniform_prob] * (num_children - len(probs_list))
        except Exception as e:
            return [1.0] * len(node.children)
    
    def _config_to_tensor(self, config: Dict[str, Any]) -> torch.Tensor:
        """Convert configuration to tensor for neural network input."""
        features = []
        for key in ['learning_rate', 'batch_size', 'hidden_size', 'num_layers', 'num_heads', 'dropout', 'weight_decay', 'warmup_ratio']:
            if key in config:
                if key == 'learning_rate':
                    features.append(math.log10(config[key] + 1e-10))
                elif key in ['batch_size', 'hidden_size', 'num_layers', 'num_heads']:
                    features.append(math.log2(config[key] + 1))
                else:
                    features.append(config[key])
            else:
                features.append(0.0)
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def neural_guided_selection(self, node: MCTSNode, eval_remain: float) -> MCTSNode:
        """Select child using neural network policy guidance."""
        if not node.children:
            return node
            
        policy_priors = self.get_policy_priors(node) if self.args.use_neural_guidance else [1.0] * len(node.children)
        
        best_child = None
        best_score = float('-inf')
        
        for i, child in enumerate(node.children):
            uct_score = self.uct(child, eval_remain)
            
            if self.args.use_neural_guidance:
                policy_bonus = policy_priors[i] * self.args.neural_guidance_weight
                entropy_bonus = self.calculate_entropy_bonus(child)
                total_score = uct_score + policy_bonus + entropy_bonus
            else:
                total_score = uct_score
            
            if self.should_prune_move(child, total_score):
                self.pruned_moves += 1
                continue
                
            if total_score > best_score:
                best_score = total_score
                best_child = child
        
        return best_child if best_child else node.children[0]
    
    def calculate_entropy_bonus(self, node: MCTSNode) -> float:
        """Calculate entropy-based exploration bonus."""
        if node.visits == 0 or not node.parent:
            return self.args.entropy_weight
        
        total_visits = sum(c.visits for c in node.parent.children)
        if total_visits == 0:
            return self.args.entropy_weight
            
        visit_prob = node.visits / total_visits
        entropy = -visit_prob * math.log(visit_prob + 1e-8)
        return entropy * self.args.entropy_weight
    
    def should_prune_move(self, node: MCTSNode, score: float) -> bool:
        """Determine if a move should be pruned based on score and visits."""
        if not self.args.time_management:
            return False
            
        if node.visits < 2:
            return False
            
        avg_score = sum(c.Q for c in node.parent.children) / len(node.parent.children)
        return score < avg_score - self.args.pruning_threshold
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'pruned_moves': self.pruned_moves,
            'total_evaluations': self.total_evaluations,
            'pruning_rate': self.pruned_moves / max(1, self.total_evaluations),
            'entropy_history': self.entropy_history[-10:],
            'neural_guidance_enabled': self.args.use_neural_guidance
        }

class EnhancedMCTSOptimizer(MCTSOptimizer):
    """Enhanced MCTS optimizer with neural guidance and advanced features."""
    
    def __init__(self, args: NeuralGuidedMCTSArgs, objective_function, 
                 policy_network=None, value_network=None):
        super().__init__(args, objective_function)
        self.args = args
        self.policy_network = policy_network
        self.value_network = value_network
        self.optimization_history = []
    
    def optimize(self, initial_config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], float]:
        """Run enhanced MCTS optimization."""
        if initial_config is None:
            initial_config = self.generate_random_config()
        
        mcts = NeuralGuidedMCTS(initial_config, self.args, self.policy_network, self.value_network)
        
        initial_objective = self.evaluate_config(initial_config)
        mcts.root.objective = initial_objective
        mcts.root.Q = -1 * initial_objective
        
        print(f"Starting Enhanced MCTS optimization with initial objective: {initial_objective:.4f}")
        
        for i in range(self.args.init_size):
            if i == 0:
                config = initial_config
                objective = initial_objective
            else:
                config = self.generate_random_config()
                objective = self.evaluate_config(config)
            
            child_node = MCTSNode(
                config=config,
                objective=objective,
                parent=mcts.root,
                depth=1,
                visits=1,
                Q=-1 * objective
            )
            mcts.root.add_child(child_node)
            mcts.root.children_info.append({
                'config': config,
                'objective': objective,
                'operator': 'init'
            })
            mcts.backpropagate(child_node)
            child_node.subtree.append(child_node)
        
        print(f"Initialized population with {len(mcts.root.children)} configurations")
        
        while self.eval_times < self.args.fe_max:
            eval_remain = max(1 - self.eval_times / self.args.fe_max, 0)
            
            if self.eval_times % 10 == 0:
                stats = mcts.get_optimization_stats()
                print(f"Evaluation {self.eval_times}/{self.args.fe_max}, Best Q: {mcts.q_max:.4f}, "
                      f"Pruning Rate: {stats['pruning_rate']:.2%}")
            
            current_node = mcts.root
            path = []
            
            while len(current_node.children) > 0 and current_node.depth < mcts.max_depth:
                selected_child = mcts.neural_guided_selection(current_node, eval_remain)
                if selected_child is None:
                    break
                path.append(selected_child)
                current_node = selected_child
                
                if int(current_node.visits ** mcts.alpha) > len(current_node.children):
                    operator_idx = random.choices(
                        range(len(self.args.operators)),
                        weights=self.args.operator_weights,
                        k=1
                    )[0]
                    operator = self.args.operators[operator_idx]
                    
                    new_node = self.expand_node(mcts, current_node, operator)
                    if new_node:
                        mcts.total_evaluations += 1
                        if self.eval_times % 20 == 0:
                            print(f"Expanded with {operator}: objective={new_node.objective:.4f}")
            
            for operator, weight in zip(self.args.operators, self.args.operator_weights):
                for _ in range(weight):
                    if self.eval_times >= self.args.fe_max:
                        break
                    expanded = self.expand_node(mcts, current_node, operator)
                    if expanded:
                        mcts.total_evaluations += 1
        
        best_node = mcts.root
        best_objective = float('inf')
        
        def find_best_recursive(node):
            nonlocal best_node, best_objective
            if node.objective < best_objective:
                best_objective = node.objective
                best_node = node
            for child in node.children:
                find_best_recursive(child)
        
        find_best_recursive(mcts.root)
        
        final_stats = mcts.get_optimization_stats()
        print(f"Enhanced MCTS optimization completed. Best objective: {best_objective:.4f}")
        print(f"Optimization stats: {final_stats}")
        print(f"Best configuration: {best_node.config}")
        
        self.optimization_history.append({
            'best_objective': best_objective,
            'best_config': best_node.config,
            'stats': final_stats
        })
        
        return best_node.config, best_objective

def create_mcts_optimizer(objective_function, args: Optional[MCTSOptimizationArgs] = None):
    """Factory function to create MCTS optimizer."""
    if args is None:
        args = MCTSOptimizationArgs()
    return MCTSOptimizer(args, objective_function)

def create_enhanced_mcts_optimizer(objective_function, args: Optional[NeuralGuidedMCTSArgs] = None,
                                 policy_network=None, value_network=None):
    """Factory function to create enhanced MCTS optimizer."""
    if args is None:
        args = NeuralGuidedMCTSArgs()
    return EnhancedMCTSOptimizer(args, objective_function, policy_network, value_network)

def example_objective_function(config: Dict[str, Any]) -> float:
    """Example objective function for model optimization."""
    penalty = 0
    if config['learning_rate'] > 1e-3:
        penalty += 0.1
    if config['dropout'] > 0.2:
        penalty += 0.05
    if config['num_layers'] > 8:
        penalty += 0.1
    
    base_loss = 0.5 + penalty + random.gauss(0, 0.1)
    return max(0.1, base_loss)
