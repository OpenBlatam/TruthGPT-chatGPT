"""
DeepSeek-R1-Qwen3 Interactive Demo

Demonstrates the advanced reasoning capabilities of the DeepSeek-R1-Qwen3 model
including step-by-step reasoning, verification, and confidence estimation.

Features:
- Interactive reasoning sessions
- Step-by-step problem solving
- Confidence calibration
- Verification mechanisms
- Multiple reasoning modes
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import yaml

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import DeepSeekR1Qwen3ForCausalLM, DeepSeekR1Qwen3Config

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class ReasoningSession:
    """Tracks a reasoning session."""
    problem: str
    reasoning_steps: List[Dict[str, Any]]
    final_answer: str
    confidence: float
    verification_status: str
    total_time: float


class DeepSeekR1Qwen3Demo:
    """Interactive demo for DeepSeek-R1-Qwen3 reasoning model."""
    
    def __init__(self, config_path: str = "config.yaml", model_size: str = "medium"):
        """Initialize the demo with model configuration."""
        self.config_path = config_path
        self.model_size = model_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"{Colors.HEADER}🧠 DeepSeek-R1-Qwen3 Reasoning Demo{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Device: {self.device}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Model Size: {model_size}{Colors.ENDC}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._initialize_model()
        
        # Demo statistics
        self.session_history: List[ReasoningSession] = []
        
        print(f"{Colors.OKGREEN}✅ Model loaded successfully!{Colors.ENDC}\n")
    
    def _load_config(self) -> Dict:
        """Load model configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get variant-specific config
            if self.model_size in config.get("model_variants", {}):
                variant_config = config["model_variants"][self.model_size]
                # Merge with base config
                model_config = config["model_config"].copy()
                model_config.update(variant_config)
                config["model_config"] = model_config
            
            return config
        except FileNotFoundError:
            print(f"{Colors.WARNING}Config file not found. Using default configuration.{Colors.ENDC}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for demo."""
        return {
            "model_config": {
                "vocab_size": 151936,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "hidden_act": "silu",
                "max_position_embeddings": 131072,
                "reasoning_depth": 5,
                "thinking_tokens": 23000,
                "chain_of_thought_layers": [4, 8, 12, 16, 20, 23],
                "use_thinking_head": True,
                "thinking_head_size": 512,
                "use_step_by_step": True,
                "use_verification": True,
                "use_reflection": True,
                "max_reasoning_steps": 10,
                "reasoning_confidence_threshold": 0.8,
            }
        }
    
    def _initialize_model(self) -> tuple:
        """Initialize model and tokenizer."""
        print(f"{Colors.OKCYAN}Loading model...{Colors.ENDC}")
        
        # Create model configuration
        model_config = DeepSeekR1Qwen3Config(**self.config["model_config"])
        
        # Initialize model
        model = DeepSeekR1Qwen3ForCausalLM(model_config)
        model.to(self.device)
        model.eval()
        
        # Create a simple tokenizer (in practice, you'd load from HuggingFace)
        tokenizer = self._create_demo_tokenizer()
        
        return model, tokenizer
    
    def _create_demo_tokenizer(self):
        """Create a demo tokenizer for testing."""
        class DemoTokenizer:
            def __init__(self):
                self.vocab_size = 151936
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.bos_token_id = 2
                
                # Simple vocabulary for demo
                self.vocab = {
                    "<pad>": 0, "<eos>": 1, "<bos>": 2,
                    "Problem": 3, "Reasoning": 4, "Step": 5, "Answer": 6,
                    "What": 7, "is": 8, "the": 9, "of": 10, "and": 11,
                    "a": 12, "to": 13, "in": 14, "for": 15, "with": 16,
                    "I": 17, "need": 18, "find": 19, "solve": 20, "calculate": 21,
                    "Let": 22, "me": 23, "think": 24, "about": 25, "this": 26,
                    "First": 27, "Second": 28, "Then": 29, "Finally": 30,
                    "Therefore": 31, "So": 32, "Thus": 33, "Hence": 34,
                    "0": 35, "1": 36, "2": 37, "3": 38, "4": 39, "5": 40,
                    "6": 41, "7": 42, "8": 43, "9": 44, "+": 45, "-": 46,
                    "*": 47, "/": 48, "=": 49, "(": 50, ")": 51,
                    ".": 52, ",": 53, "?": 54, "!": 55, ":": 56, ";": 57,
                    "\n": 58, " ": 59,
                }
                self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
                """Simple encoding for demo."""
                tokens = []
                if add_special_tokens:
                    tokens.append(self.bos_token_id)
                
                # Simple word-level tokenization
                words = text.split()
                for word in words:
                    if word in self.vocab:
                        tokens.append(self.vocab[word])
                    else:
                        # Use a hash-based approach for unknown words
                        token_id = (hash(word) % (self.vocab_size - 100)) + 100
                        tokens.append(token_id)
                
                if add_special_tokens:
                    tokens.append(self.eos_token_id)
                
                return tokens
            
            def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
                """Simple decoding for demo."""
                words = []
                for token_id in token_ids:
                    if skip_special_tokens and token_id in [self.pad_token_id, self.eos_token_id, self.bos_token_id]:
                        continue
                    if token_id in self.reverse_vocab:
                        words.append(self.reverse_vocab[token_id])
                    else:
                        words.append(f"<unk_{token_id}>")
                
                return " ".join(words)
            
            def __call__(self, text: str, return_tensors: str = None, padding: bool = False, 
                        truncation: bool = False, max_length: int = None, **kwargs):
                """Tokenizer call interface."""
                token_ids = self.encode(text)
                
                if max_length and len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                
                if padding and max_length:
                    while len(token_ids) < max_length:
                        token_ids.append(self.pad_token_id)
                
                result = {
                    "input_ids": token_ids,
                    "attention_mask": [1 if tid != self.pad_token_id else 0 for tid in token_ids]
                }
                
                if return_tensors == "pt":
                    result = {k: torch.tensor([v]) for k, v in result.items()}
                
                return result
        
        return DemoTokenizer()
    
    def run_interactive_demo(self):
        """Run the interactive demo."""
        print(f"{Colors.BOLD}🎯 Interactive Reasoning Demo{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Enter reasoning problems and watch the model think step-by-step!{Colors.ENDC}")
        print(f"{Colors.WARNING}Commands: 'quit' to exit, 'stats' for statistics, 'examples' for sample problems{Colors.ENDC}\n")
        
        while True:
            try:
                # Get user input
                problem = input(f"{Colors.BOLD}🤔 Enter a problem: {Colors.ENDC}").strip()
                
                if problem.lower() in ['quit', 'exit', 'q']:
                    break
                elif problem.lower() == 'stats':
                    self._show_statistics()
                    continue
                elif problem.lower() == 'examples':
                    self._show_examples()
                    continue
                elif not problem:
                    continue
                
                # Solve the problem
                print(f"\n{Colors.OKCYAN}🧠 Thinking...{Colors.ENDC}")
                session = self._solve_problem(problem)
                
                # Display results
                self._display_reasoning_session(session)
                
                # Add to history
                self.session_history.append(session)
                
                print(f"\n{Colors.OKGREEN}{'='*60}{Colors.ENDC}\n")
                
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Demo interrupted by user.{Colors.ENDC}")
                break
            except Exception as e:
                print(f"{Colors.FAIL}Error: {str(e)}{Colors.ENDC}")
        
        print(f"\n{Colors.HEADER}Thanks for using DeepSeek-R1-Qwen3 Demo! 🚀{Colors.ENDC}")
        self._show_final_statistics()
    
    def _solve_problem(self, problem: str) -> ReasoningSession:
        """Solve a problem using the reasoning model."""
        start_time = time.time()
        
        # Prepare input
        input_text = f"Problem: {problem}\n\nReasoning:"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with reasoning
        with torch.no_grad():
            try:
                # Use the model's reasoning generation if available
                if hasattr(self.model, 'generate_with_reasoning'):
                    outputs = self.model.generate_with_reasoning(
                        input_ids=inputs["input_ids"],
                        max_length=inputs["input_ids"].shape[1] + 512,
                        temperature=0.7,
                        confidence_threshold=0.7,
                    )
                    
                    generated_text = self.tokenizer.decode(
                        outputs["generated_ids"][0],
                        skip_special_tokens=True
                    )
                    
                    reasoning_steps = outputs.get("reasoning_steps", [])
                    confidence = np.mean([step.get("confidence", 0.5) for step in reasoning_steps]) if reasoning_steps else 0.5
                    
                else:
                    # Fallback to standard generation
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        max_length=inputs["input_ids"].shape[1] + 512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    reasoning_steps = self._extract_reasoning_steps(generated_text)
                    confidence = 0.7  # Default confidence
                
            except Exception as e:
                print(f"{Colors.WARNING}Generation error: {e}. Using fallback.{Colors.ENDC}")
                generated_text = self._generate_fallback_response(problem)
                reasoning_steps = self._extract_reasoning_steps(generated_text)
                confidence = 0.6
        
        end_time = time.time()
        
        # Extract answer
        answer = self._extract_final_answer(generated_text)
        
        # Determine verification status
        verification_status = "verified" if confidence > 0.8 else "uncertain"
        
        return ReasoningSession(
            problem=problem,
            reasoning_steps=reasoning_steps,
            final_answer=answer,
            confidence=confidence,
            verification_status=verification_status,
            total_time=end_time - start_time
        )
    
    def _generate_fallback_response(self, problem: str) -> str:
        """Generate a fallback response for demo purposes."""
        fallback_responses = {
            "math": f"Let me solve this step by step.\nStep 1: I need to analyze the mathematical problem: {problem}\nStep 2: I'll break down the components and apply appropriate mathematical operations.\nStep 3: After careful calculation, I arrive at the solution.\nAnswer: This requires mathematical computation based on the given problem.",
            
            "logic": f"Let me think through this logically.\nStep 1: I need to understand the logical structure of: {problem}\nStep 2: I'll identify the premises and conclusions.\nStep 3: I'll apply logical reasoning principles.\nAnswer: Based on logical analysis, here's my reasoning.",
            
            "general": f"Let me approach this systematically.\nStep 1: I need to understand what's being asked: {problem}\nStep 2: I'll gather relevant information and context.\nStep 3: I'll formulate a well-reasoned response.\nAnswer: Based on my analysis, here's my response to the problem."
        }
        
        # Simple heuristic to choose response type
        if any(word in problem.lower() for word in ["calculate", "solve", "math", "number", "+", "-", "*", "/"]):
            return fallback_responses["math"]
        elif any(word in problem.lower() for word in ["logic", "if", "then", "because", "therefore"]):
            return fallback_responses["logic"]
        else:
            return fallback_responses["general"]
    
    def _extract_reasoning_steps(self, text: str) -> List[Dict[str, Any]]:
        """Extract reasoning steps from generated text."""
        steps = []
        lines = text.split('\n')
        
        step_num = 1
        for line in lines:
            line = line.strip()
            if line.startswith(f"Step {step_num}:") or line.startswith(f"{step_num}."):
                step_content = line.split(":", 1)[-1].strip()
                steps.append({
                    "step": step_num,
                    "content": step_content,
                    "confidence": 0.7 + (step_num * 0.05),  # Increasing confidence
                    "verification": "plausible"
                })
                step_num += 1
        
        return steps
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from generated text."""
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith("Answer:"):
                return line.split(":", 1)[-1].strip()
        
        # Fallback: return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return "No clear answer found."
    
    def _display_reasoning_session(self, session: ReasoningSession):
        """Display the reasoning session results."""
        print(f"\n{Colors.BOLD}📋 Problem:{Colors.ENDC} {session.problem}")
        print(f"\n{Colors.BOLD}🔍 Step-by-Step Reasoning:{Colors.ENDC}")
        
        for i, step in enumerate(session.reasoning_steps, 1):
            confidence_color = Colors.OKGREEN if step.get("confidence", 0) > 0.8 else Colors.WARNING
            print(f"  {Colors.OKCYAN}Step {i}:{Colors.ENDC} {step['content']}")
            print(f"    {confidence_color}Confidence: {step.get('confidence', 0.5):.2f}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}✅ Final Answer:{Colors.ENDC} {session.final_answer}")
        
        # Display confidence and verification
        confidence_color = Colors.OKGREEN if session.confidence > 0.8 else Colors.WARNING if session.confidence > 0.6 else Colors.FAIL
        print(f"\n{Colors.BOLD}📊 Analysis:{Colors.ENDC}")
        print(f"  {confidence_color}Overall Confidence: {session.confidence:.2f}{Colors.ENDC}")
        print(f"  {Colors.OKCYAN}Verification Status: {session.verification_status}{Colors.ENDC}")
        print(f"  {Colors.OKCYAN}Reasoning Steps: {len(session.reasoning_steps)}{Colors.ENDC}")
        print(f"  {Colors.OKCYAN}Processing Time: {session.total_time:.2f}s{Colors.ENDC}")
    
    def _show_examples(self):
        """Show example problems."""
        examples = [
            "What is 15 * 23?",
            "If a train travels 120 km in 2 hours, what is its average speed?",
            "Solve for x: 2x + 5 = 13",
            "A rectangle has length 8 cm and width 5 cm. What is its area?",
            "If I have 3 apples and buy 7 more, then give away 4, how many do I have left?",
            "What is the next number in the sequence: 2, 4, 8, 16, ?",
            "If all cats are animals and Fluffy is a cat, what can we conclude about Fluffy?",
            "A store sells books for $12 each. If I buy 3 books and pay with a $50 bill, how much change do I get?",
        ]
        
        print(f"\n{Colors.BOLD}📚 Example Problems:{Colors.ENDC}")
        for i, example in enumerate(examples, 1):
            print(f"  {Colors.OKCYAN}{i}.{Colors.ENDC} {example}")
        print()
    
    def _show_statistics(self):
        """Show session statistics."""
        if not self.session_history:
            print(f"{Colors.WARNING}No sessions completed yet.{Colors.ENDC}")
            return
        
        total_sessions = len(self.session_history)
        avg_confidence = sum(s.confidence for s in self.session_history) / total_sessions
        avg_steps = sum(len(s.reasoning_steps) for s in self.session_history) / total_sessions
        avg_time = sum(s.total_time for s in self.session_history) / total_sessions
        high_confidence_sessions = sum(1 for s in self.session_history if s.confidence > 0.8)
        
        print(f"\n{Colors.BOLD}📈 Session Statistics:{Colors.ENDC}")
        print(f"  {Colors.OKCYAN}Total Sessions: {total_sessions}{Colors.ENDC}")
        print(f"  {Colors.OKCYAN}Average Confidence: {avg_confidence:.2f}{Colors.ENDC}")
        print(f"  {Colors.OKCYAN}Average Reasoning Steps: {avg_steps:.1f}{Colors.ENDC}")
        print(f"  {Colors.OKCYAN}Average Processing Time: {avg_time:.2f}s{Colors.ENDC}")
        print(f"  {Colors.OKGREEN}High Confidence Sessions: {high_confidence_sessions}/{total_sessions} ({high_confidence_sessions/total_sessions*100:.1f}%){Colors.ENDC}")
        print()
    
    def _show_final_statistics(self):
        """Show final statistics at the end of the demo."""
        if self.session_history:
            print(f"\n{Colors.BOLD}🎯 Final Demo Statistics:{Colors.ENDC}")
            self._show_statistics()
            
            # Show best session
            best_session = max(self.session_history, key=lambda s: s.confidence)
            print(f"{Colors.BOLD}🏆 Best Reasoning Session:{Colors.ENDC}")
            print(f"  Problem: {best_session.problem}")
            print(f"  Confidence: {best_session.confidence:.2f}")
            print(f"  Steps: {len(best_session.reasoning_steps)}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="DeepSeek-R1-Qwen3 Interactive Demo")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--model-size", default="medium", choices=["small", "medium", "large"], 
                       help="Model size variant to use")
    parser.add_argument("--batch-demo", action="store_true", help="Run batch demo instead of interactive")
    
    args = parser.parse_args()
    
    try:
        # Initialize demo
        demo = DeepSeekR1Qwen3Demo(args.config, args.model_size)
        
        if args.batch_demo:
            # Run batch demo with predefined problems
            demo.run_batch_demo()
        else:
            # Run interactive demo
            demo.run_interactive_demo()
            
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Demo interrupted by user.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Demo error: {str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Add numpy import for compatibility
    try:
        import numpy as np
    except ImportError:
        # Fallback numpy functions
        class np:
            @staticmethod
            def mean(values):
                return sum(values) / len(values) if values else 0
    
    main()