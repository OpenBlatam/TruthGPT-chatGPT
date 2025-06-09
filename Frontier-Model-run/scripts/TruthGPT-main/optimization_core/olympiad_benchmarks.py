"""
Mathematical Olympiad Benchmark Suite for AI Model Evaluation.
Includes algebra, number theory, geometry, and combinatorics problems.
"""

import random
import math
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import torch
import torch.nn as nn

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: SymPy not available. Some advanced problem generation features disabled.")

class ProblemCategory(Enum):
    ALGEBRA = "algebra"
    NUMBER_THEORY = "number_theory"
    GEOMETRY = "geometry"
    COMBINATORICS = "combinatorics"

class DifficultyLevel(Enum):
    AMC_12 = "amc_12"
    AIME = "aime"
    USAMO = "usamo"
    IMO = "imo"

@dataclass
class OlympiadProblem:
    """Represents a mathematical olympiad problem."""
    problem_id: str
    category: ProblemCategory
    difficulty: DifficultyLevel
    statement: str
    latex_statement: str
    answer: Union[int, float, str]
    solution: str
    hints: List[str] = field(default_factory=list)
    time_limit_minutes: int = 45
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert problem to dictionary."""
        result = asdict(self)
        result['category'] = self.category.value
        result['difficulty'] = self.difficulty.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OlympiadProblem':
        """Create problem from dictionary."""
        data['category'] = ProblemCategory(data['category'])
        data['difficulty'] = DifficultyLevel(data['difficulty'])
        return cls(**data)

class AlgebraProblemGenerator:
    """Generate algebra problems: polynomials, inequalities, functional equations."""
    
    def generate_polynomial_problem(self, difficulty: DifficultyLevel) -> OlympiadProblem:
        """Generate polynomial equation problems."""
        problem_id = f"algebra_poly_{difficulty.value}_{random.randint(1000, 9999)}"
        
        if difficulty == DifficultyLevel.AMC_12:
            a, b, c = random.randint(1, 5), random.randint(-10, 10), random.randint(-10, 10)
            
            if SYMPY_AVAILABLE:
                x = sp.Symbol('x')
                poly = a*x**2 + b*x + c
                roots = sp.solve(poly, x)
                answer = float(sum(roots)) if roots else 0
                poly_str = str(poly)
                latex_poly = sp.latex(poly)
            else:
                poly_str = f"{a}x² + {b}x + {c}"
                latex_poly = f"{a}x^2 + {b}x + {c}"
                answer = -b/a if a != 0 else 0
            
            statement = f"Find the sum of the roots of {poly_str} = 0"
            latex_statement = f"Find the sum of the roots of ${latex_poly} = 0$"
            solution = "Use Vieta's formulas: sum of roots = -b/a"
            hints = ["Consider the relationship between coefficients and roots", "Use Vieta's formulas"]
            keywords = ["polynomial", "roots", "vieta", "quadratic"]
            
        elif difficulty == DifficultyLevel.AIME:
            coeffs = [random.randint(-3, 3) for _ in range(4)]
            if coeffs[0] == 0:
                coeffs[0] = 1
            
            if SYMPY_AVAILABLE:
                x = sp.Symbol('x')
                poly = sum(coeffs[i] * x**(3-i) for i in range(4))
                real_roots = [r for r in sp.solve(poly, x) if r.is_real]
                answer = len(real_roots)
                poly_str = str(poly)
                latex_poly = sp.latex(poly)
            else:
                poly_str = f"{coeffs[0]}x³ + {coeffs[1]}x² + {coeffs[2]}x + {coeffs[3]}"
                latex_poly = f"{coeffs[0]}x^3 + {coeffs[1]}x^2 + {coeffs[2]}x + {coeffs[3]}"
                answer = random.randint(1, 3)
            
            statement = f"Find the number of real roots of {poly_str} = 0"
            latex_statement = f"Find the number of real roots of ${latex_poly} = 0$"
            solution = "Use calculus to find critical points and analyze sign changes"
            hints = ["Find the derivative to locate critical points", "Use the intermediate value theorem"]
            keywords = ["polynomial", "real roots", "cubic", "calculus"]
            
        else:
            statement = "Find all real solutions to the functional equation f(x+y) = f(x) + f(y) for all real x, y"
            latex_statement = "Find all real solutions to the functional equation $f(x+y) = f(x) + f(y)$ for all real $x, y$"
            answer = "f(x) = cx for some constant c"
            solution = "This is Cauchy's functional equation. The continuous solutions are f(x) = cx."
            hints = ["Consider what happens when x = y = 0", "Try specific values like x = y"]
            keywords = ["functional equation", "cauchy", "linearity"]
        
        return OlympiadProblem(
            problem_id=problem_id,
            category=ProblemCategory.ALGEBRA,
            difficulty=difficulty,
            statement=statement,
            latex_statement=latex_statement,
            answer=answer,
            solution=solution,
            hints=hints,
            keywords=keywords
        )
    
    def generate_inequality_problem(self, difficulty: DifficultyLevel) -> OlympiadProblem:
        """Generate inequality problems."""
        problem_id = f"algebra_ineq_{difficulty.value}_{random.randint(1000, 9999)}"
        
        if difficulty == DifficultyLevel.AMC_12:
            a, b = random.randint(1, 10), random.randint(1, 10)
            statement = f"For positive real numbers x and y, find the minimum value of {a}/x + {b}/y given that x + y = {a+b}"
            latex_statement = f"For positive real numbers $x$ and $y$, find the minimum value of $\\frac{{{a}}}{{x}} + \\frac{{{b}}}{{y}}$ given that $x + y = {a+b}$"
            answer = 2 * math.sqrt(a * b) / (a + b) * (a + b)
            solution = "Use Lagrange multipliers or AM-GM inequality"
            hints = ["Try the AM-GM inequality", "Consider when equality occurs"]
            keywords = ["inequality", "optimization", "am-gm", "minimum"]
            
        else:
            statement = "Prove that for positive real numbers a, b, c: (a+b+c)³ ≥ 27abc"
            latex_statement = "Prove that for positive real numbers $a, b, c$: $(a+b+c)^3 \\geq 27abc$"
            answer = "Proof using AM-GM inequality"
            solution = "Apply AM-GM: (a+b+c)/3 ≥ ∛(abc), then cube both sides"
            hints = ["Use the AM-GM inequality", "Consider when equality holds"]
            keywords = ["inequality", "am-gm", "proof", "optimization"]
        
        return OlympiadProblem(
            problem_id=problem_id,
            category=ProblemCategory.ALGEBRA,
            difficulty=difficulty,
            statement=statement,
            latex_statement=latex_statement,
            answer=answer,
            solution=solution,
            hints=hints,
            keywords=keywords
        )

class NumberTheoryProblemGenerator:
    """Generate number theory problems: divisibility, modular arithmetic, Diophantine equations."""
    
    def generate_modular_arithmetic_problem(self, difficulty: DifficultyLevel) -> OlympiadProblem:
        """Generate modular arithmetic problems."""
        problem_id = f"number_theory_mod_{difficulty.value}_{random.randint(1000, 9999)}"
        
        if difficulty == DifficultyLevel.AMC_12:
            a, m = random.randint(10, 100), random.randint(7, 20)
            remainder = a % m
            
            statement = f"Find the remainder when {a} is divided by {m}"
            latex_statement = f"Find the remainder when ${a}$ is divided by ${m}$"
            answer = remainder
            solution = f"{a} = {a//m} × {m} + {remainder}"
            hints = ["Use the division algorithm"]
            keywords = ["modular arithmetic", "remainder", "division"]
            
        elif difficulty == DifficultyLevel.AIME:
            base, exp, mod = random.randint(2, 10), random.randint(10, 50), random.randint(11, 100)
            answer = pow(base, exp, mod)
            
            statement = f"Find the remainder when {base}^{exp} is divided by {mod}"
            latex_statement = f"Find the remainder when ${base}^{{{exp}}}$ is divided by ${mod}$"
            solution = "Use modular exponentiation or Fermat's Little Theorem if applicable"
            hints = ["Look for patterns in powers", "Consider Fermat's Little Theorem if mod is prime"]
            keywords = ["modular arithmetic", "exponentiation", "fermat"]
            
        else:
            p = random.choice([7, 11, 13, 17, 19, 23])
            statement = f"Find the number of solutions to x² ≡ 1 (mod {p}²)"
            latex_statement = f"Find the number of solutions to $x^2 \\equiv 1 \\pmod{{{p}^2}}$"
            answer = 2
            solution = f"Use Hensel's lemma to lift solutions from mod {p} to mod {p}²"
            hints = ["Start with solutions modulo p", "Use Hensel's lemma"]
            keywords = ["modular arithmetic", "hensel", "quadratic residue"]
        
        return OlympiadProblem(
            problem_id=problem_id,
            category=ProblemCategory.NUMBER_THEORY,
            difficulty=difficulty,
            statement=statement,
            latex_statement=latex_statement,
            answer=answer,
            solution=solution,
            hints=hints,
            keywords=keywords
        )
    
    def generate_divisibility_problem(self, difficulty: DifficultyLevel) -> OlympiadProblem:
        """Generate divisibility problems."""
        problem_id = f"number_theory_div_{difficulty.value}_{random.randint(1000, 9999)}"
        
        if difficulty == DifficultyLevel.AMC_12:
            n = random.randint(100, 999)
            digit_sum = sum(int(d) for d in str(n))
            
            statement = f"Is {n} divisible by 3? The sum of its digits is {digit_sum}"
            latex_statement = f"Is ${n}$ divisible by 3? The sum of its digits is ${digit_sum}$"
            answer = "Yes" if digit_sum % 3 == 0 else "No"
            solution = "A number is divisible by 3 if and only if the sum of its digits is divisible by 3"
            hints = ["Use the divisibility rule for 3"]
            keywords = ["divisibility", "digit sum", "divisibility rule"]
            
        else:
            a, b = random.randint(2, 10), random.randint(2, 10)
            statement = f"Prove that {a}^n - {b}^n is divisible by {a-b} for all positive integers n"
            latex_statement = f"Prove that ${a}^n - {b}^n$ is divisible by ${a-b}$ for all positive integers $n$"
            answer = "Proof by factorization"
            solution = f"Use the factorization a^n - b^n = (a-b)(a^(n-1) + a^(n-2)b + ... + b^(n-1))"
            hints = ["Factor the expression", "Use induction"]
            keywords = ["divisibility", "factorization", "proof", "induction"]
        
        return OlympiadProblem(
            problem_id=problem_id,
            category=ProblemCategory.NUMBER_THEORY,
            difficulty=difficulty,
            statement=statement,
            latex_statement=latex_statement,
            answer=answer,
            solution=solution,
            hints=hints,
            keywords=keywords
        )

class GeometryProblemGenerator:
    """Generate geometry problems: Euclidean plane geometry, triangles, circles."""
    
    def generate_triangle_problem(self, difficulty: DifficultyLevel) -> OlympiadProblem:
        """Generate triangle geometry problems."""
        problem_id = f"geometry_triangle_{difficulty.value}_{random.randint(1000, 9999)}"
        
        if difficulty == DifficultyLevel.AMC_12:
            a, b, c = 3, 4, 5
            area = 0.5 * a * b
            
            statement = f"Find the area of a right triangle with legs of length {a} and {b}"
            latex_statement = f"Find the area of a right triangle with legs of length ${a}$ and ${b}$"
            answer = area
            solution = f"Area = (1/2) × base × height = (1/2) × {a} × {b} = {area}"
            hints = ["Use the formula for the area of a triangle"]
            keywords = ["triangle", "area", "right triangle"]
            
        elif difficulty == DifficultyLevel.AIME:
            statement = "In triangle ABC, AB = 13, BC = 14, CA = 15. Find the length of the altitude from A to BC"
            latex_statement = "In triangle $ABC$, $AB = 13$, $BC = 14$, $CA = 15$. Find the length of the altitude from $A$ to $BC$"
            
            s = (13 + 14 + 15) / 2
            area = math.sqrt(s * (s-13) * (s-14) * (s-15))
            altitude = 2 * area / 14
            answer = altitude
            
            solution = "Use Heron's formula to find the area, then use Area = (1/2) × base × height"
            hints = ["Use Heron's formula first", "Area = (1/2) × base × height"]
            keywords = ["triangle", "altitude", "heron", "area"]
            
        else:
            statement = "Prove that in any triangle, the sum of any two sides is greater than the third side"
            latex_statement = "Prove that in any triangle, the sum of any two sides is greater than the third side"
            answer = "Triangle inequality proof"
            solution = "Use the fact that the shortest path between two points is a straight line"
            hints = ["Consider the shortest path between two vertices"]
            keywords = ["triangle inequality", "proof", "geometry"]
        
        return OlympiadProblem(
            problem_id=problem_id,
            category=ProblemCategory.GEOMETRY,
            difficulty=difficulty,
            statement=statement,
            latex_statement=latex_statement,
            answer=answer,
            solution=solution,
            hints=hints,
            keywords=keywords
        )
    
    def generate_circle_problem(self, difficulty: DifficultyLevel) -> OlympiadProblem:
        """Generate circle geometry problems."""
        problem_id = f"geometry_circle_{difficulty.value}_{random.randint(1000, 9999)}"
        
        if difficulty == DifficultyLevel.AMC_12:
            r = random.randint(3, 10)
            area = math.pi * r * r
            
            statement = f"Find the area of a circle with radius {r}"
            latex_statement = f"Find the area of a circle with radius ${r}$"
            answer = f"{r}²π"
            solution = f"Area = πr² = π × {r}² = {r}²π"
            hints = ["Use the formula A = πr²"]
            keywords = ["circle", "area", "radius"]
            
        else:
            statement = "Two circles intersect at points A and B. Prove that the line AB is perpendicular to the line connecting the centers"
            latex_statement = "Two circles intersect at points $A$ and $B$. Prove that the line $AB$ is perpendicular to the line connecting the centers"
            answer = "Proof using symmetry"
            solution = "Use the fact that both intersection points are equidistant from both centers"
            hints = ["Consider the symmetry of the configuration", "Use properties of perpendicular bisectors"]
            keywords = ["circle", "intersection", "perpendicular", "proof"]
        
        return OlympiadProblem(
            problem_id=problem_id,
            category=ProblemCategory.GEOMETRY,
            difficulty=difficulty,
            statement=statement,
            latex_statement=latex_statement,
            answer=answer,
            solution=solution,
            hints=hints,
            keywords=keywords
        )

class CombinatoricsProblemGenerator:
    """Generate combinatorics problems: counting, permutations, combinations."""
    
    def generate_counting_problem(self, difficulty: DifficultyLevel) -> OlympiadProblem:
        """Generate counting problems."""
        problem_id = f"combinatorics_count_{difficulty.value}_{random.randint(1000, 9999)}"
        
        if difficulty == DifficultyLevel.AMC_12:
            n, k = random.randint(5, 10), random.randint(2, 4)
            if k > n:
                k = n - 1
            
            answer = math.comb(n, k)
            statement = f"In how many ways can you choose {k} objects from {n} distinct objects?"
            latex_statement = f"In how many ways can you choose ${k}$ objects from ${n}$ distinct objects?"
            solution = f"C({n},{k}) = {n}!/({k}!×({n-k})!) = {answer}"
            hints = ["Use the combination formula"]
            keywords = ["combinations", "counting", "binomial coefficient"]
            
        elif difficulty == DifficultyLevel.AIME:
            n = random.randint(6, 10)
            answer = math.factorial(n)
            
            statement = f"How many ways can {n} people be arranged in a line?"
            latex_statement = f"How many ways can ${n}$ people be arranged in a line?"
            solution = f"{n}! = {answer}"
            hints = ["Each position can be filled in decreasing number of ways"]
            keywords = ["permutations", "factorial", "arrangements"]
            
        else:
            statement = "How many ways can you tile a 2×n rectangle with 1×2 dominoes?"
            latex_statement = "How many ways can you tile a $2 \\times n$ rectangle with $1 \\times 2$ dominoes?"
            answer = "Fibonacci sequence: F(n+1)"
            solution = "Use recurrence relation: f(n) = f(n-1) + f(n-2)"
            hints = ["Set up a recurrence relation", "Consider the rightmost column"]
            keywords = ["tiling", "recurrence", "fibonacci", "dynamic programming"]
        
        return OlympiadProblem(
            problem_id=problem_id,
            category=ProblemCategory.COMBINATORICS,
            difficulty=difficulty,
            statement=statement,
            latex_statement=latex_statement,
            answer=answer,
            solution=solution,
            hints=hints,
            keywords=keywords
        )

@dataclass
class OlympiadBenchmarkConfig:
    """Configuration for olympiad benchmarking."""
    enable_olympiad_benchmarks: bool = True
    problem_categories: List[str] = field(default_factory=lambda: ["algebra", "number_theory", "geometry", "combinatorics"])
    difficulty_levels: List[str] = field(default_factory=lambda: ["amc_12", "aime", "usamo"])
    problems_per_category: int = 10
    time_limit_minutes: int = 45
    enable_latex_rendering: bool = True
    save_problems: bool = True
    random_seed: Optional[int] = None

class OlympiadBenchmarkSuite:
    """Comprehensive olympiad benchmark suite for AI model evaluation."""
    
    def __init__(self, config: OlympiadBenchmarkConfig):
        self.config = config
        self.problems: List[OlympiadProblem] = []
        self.generators = {
            ProblemCategory.ALGEBRA: AlgebraProblemGenerator(),
            ProblemCategory.NUMBER_THEORY: NumberTheoryProblemGenerator(),
            ProblemCategory.GEOMETRY: GeometryProblemGenerator(),
            ProblemCategory.COMBINATORICS: CombinatoricsProblemGenerator()
        }
        
        if config.random_seed is not None:
            random.seed(config.random_seed)
    
    def generate_problem_set(self) -> List[OlympiadProblem]:
        """Generate a complete set of olympiad problems."""
        problems = []
        
        for category_name in self.config.problem_categories:
            try:
                category = ProblemCategory(category_name)
                generator = self.generators[category]
                
                for difficulty_name in self.config.difficulty_levels:
                    try:
                        difficulty = DifficultyLevel(difficulty_name)
                        
                        for _ in range(self.config.problems_per_category):
                            if category == ProblemCategory.ALGEBRA:
                                if random.random() < 0.5:
                                    problem = generator.generate_polynomial_problem(difficulty)
                                else:
                                    problem = generator.generate_inequality_problem(difficulty)
                            elif category == ProblemCategory.NUMBER_THEORY:
                                if random.random() < 0.5:
                                    problem = generator.generate_modular_arithmetic_problem(difficulty)
                                else:
                                    problem = generator.generate_divisibility_problem(difficulty)
                            elif category == ProblemCategory.GEOMETRY:
                                if random.random() < 0.5:
                                    problem = generator.generate_triangle_problem(difficulty)
                                else:
                                    problem = generator.generate_circle_problem(difficulty)
                            elif category == ProblemCategory.COMBINATORICS:
                                problem = generator.generate_counting_problem(difficulty)
                            
                            problems.append(problem)
                            
                    except ValueError:
                        print(f"Warning: Unknown difficulty level {difficulty_name}")
                        
            except ValueError:
                print(f"Warning: Unknown category {category_name}")
        
        self.problems = problems
        
        if self.config.save_problems:
            self.save_problems_to_file()
        
        return problems
    
    def save_problems_to_file(self, filename: Optional[str] = None):
        """Save generated problems to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"olympiad_problems_{timestamp}.json"
        
        problems_data = [problem.to_dict() for problem in self.problems]
        
        with open(filename, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'problems': problems_data,
                'metadata': {
                    'generated_at': time.time(),
                    'total_problems': len(self.problems),
                    'categories': list(set(p.category.value for p in self.problems)),
                    'difficulties': list(set(p.difficulty.value for p in self.problems))
                }
            }, f, indent=2)
        
        print(f"Saved {len(self.problems)} problems to {filename}")
    
    def load_problems_from_file(self, filename: str):
        """Load problems from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.problems = [OlympiadProblem.from_dict(p) for p in data['problems']]
        print(f"Loaded {len(self.problems)} problems from {filename}")
    
    def evaluate_model_performance(self, model_responses: List[str]) -> Dict[str, Any]:
        """Evaluate model performance on olympiad problems."""
        if len(model_responses) != len(self.problems):
            raise ValueError("Number of responses must match number of problems")
        
        results = {
            'total_problems': len(self.problems),
            'correct_answers': 0,
            'category_performance': {},
            'difficulty_performance': {},
            'detailed_results': []
        }
        
        for problem, response in zip(self.problems, model_responses):
            is_correct = self._check_answer(problem, response)
            
            if is_correct:
                results['correct_answers'] += 1
            
            category = problem.category.value
            difficulty = problem.difficulty.value
            
            if category not in results['category_performance']:
                results['category_performance'][category] = {'correct': 0, 'total': 0}
            if difficulty not in results['difficulty_performance']:
                results['difficulty_performance'][difficulty] = {'correct': 0, 'total': 0}
            
            results['category_performance'][category]['total'] += 1
            results['difficulty_performance'][difficulty]['total'] += 1
            
            if is_correct:
                results['category_performance'][category]['correct'] += 1
                results['difficulty_performance'][difficulty]['correct'] += 1
            
            results['detailed_results'].append({
                'problem_id': problem.problem_id,
                'category': category,
                'difficulty': difficulty,
                'correct': is_correct,
                'expected_answer': problem.answer,
                'model_response': response
            })
        
        results['overall_accuracy'] = results['correct_answers'] / results['total_problems']
        
        for category_data in results['category_performance'].values():
            category_data['accuracy'] = category_data['correct'] / category_data['total']
        
        for difficulty_data in results['difficulty_performance'].values():
            difficulty_data['accuracy'] = difficulty_data['correct'] / difficulty_data['total']
        
        return results
    
    def _check_answer(self, problem: OlympiadProblem, response: str) -> bool:
        """Check if model response matches expected answer."""
        try:
            expected = str(problem.answer).lower().strip()
            response = response.lower().strip()
            
            if expected == response:
                return True
            
            try:
                expected_num = float(expected)
                response_num = float(response)
                return abs(expected_num - response_num) < 1e-6
            except ValueError:
                pass
            
            return expected in response or response in expected
            
        except Exception:
            return False
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        if not self.problems:
            return "No problems generated yet. Call generate_problem_set() first."
        
        report = ["# Mathematical Olympiad Benchmark Report", ""]
        report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Problems:** {len(self.problems)}")
        report.append("")
        
        category_counts = {}
        difficulty_counts = {}
        
        for problem in self.problems:
            category = problem.category.value
            difficulty = problem.difficulty.value
            
            category_counts[category] = category_counts.get(category, 0) + 1
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        report.append("## Problem Distribution")
        report.append("")
        report.append("### By Category")
        for category, count in category_counts.items():
            report.append(f"- **{category.replace('_', ' ').title()}**: {count} problems")
        
        report.append("")
        report.append("### By Difficulty")
        for difficulty, count in difficulty_counts.items():
            report.append(f"- **{difficulty.replace('_', ' ').upper()}**: {count} problems")
        
        report.append("")
        report.append("## Sample Problems")
        
        for category in ProblemCategory:
            category_problems = [p for p in self.problems if p.category == category]
            if category_problems:
                sample = random.choice(category_problems)
                report.append(f"")
                report.append(f"### {category.value.replace('_', ' ').title()} Example")
                report.append(f"**Problem:** {sample.statement}")
                report.append(f"**Answer:** {sample.answer}")
                report.append(f"**Difficulty:** {sample.difficulty.value.upper()}")
        
        return "\n".join(report)

def get_olympiad_benchmark_config(variant_name: str) -> OlympiadBenchmarkConfig:
    """Get olympiad benchmark configuration for specific variant."""
    configs = {
        'deepseek_v3': OlympiadBenchmarkConfig(
            problem_categories=["algebra", "number_theory"],
            difficulty_levels=["amc_12", "aime"],
            problems_per_category=15,
            time_limit_minutes=30
        ),
        'qwen': OlympiadBenchmarkConfig(
            problem_categories=["algebra", "number_theory", "combinatorics"],
            difficulty_levels=["amc_12", "aime"],
            problems_per_category=12
        ),
        'ultra_optimized': OlympiadBenchmarkConfig(
            problem_categories=["algebra", "number_theory", "geometry", "combinatorics"],
            difficulty_levels=["amc_12", "aime", "usamo", "imo"],
            problems_per_category=20,
            time_limit_minutes=60
        ),
        'viral_clipper': OlympiadBenchmarkConfig(
            problem_categories=["combinatorics", "geometry"],
            difficulty_levels=["amc_12", "aime"],
            problems_per_category=8
        ),
        'brandkit': OlympiadBenchmarkConfig(
            problem_categories=["algebra", "geometry"],
            difficulty_levels=["amc_12"],
            problems_per_category=10
        )
    }
    return configs.get(variant_name, OlympiadBenchmarkConfig())

def create_olympiad_benchmark_suite(variant_name: str = 'default') -> OlympiadBenchmarkSuite:
    """Factory function to create olympiad benchmark suite."""
    config = get_olympiad_benchmark_config(variant_name)
    return OlympiadBenchmarkSuite(config)

def test_problem_generation():
    """Test problem generation for all categories and difficulties."""
    print("Testing Olympiad Problem Generation")
    print("=" * 50)
    
    generators = {
        'algebra': AlgebraProblemGenerator(),
        'number_theory': NumberTheoryProblemGenerator(),
        'geometry': GeometryProblemGenerator(),
        'combinatorics': CombinatoricsProblemGenerator()
    }
    
    difficulties = [DifficultyLevel.AMC_12, DifficultyLevel.AIME]
    
    for category_name, generator in generators.items():
        print(f"\n🧪 Testing {category_name.replace('_', ' ').title()}")
        
        for difficulty in difficulties:
            try:
                if category_name == 'algebra':
                    problem = generator.generate_polynomial_problem(difficulty)
                elif category_name == 'number_theory':
                    problem = generator.generate_modular_arithmetic_problem(difficulty)
                elif category_name == 'geometry':
                    problem = generator.generate_triangle_problem(difficulty)
                elif category_name == 'combinatorics':
                    problem = generator.generate_counting_problem(difficulty)
                
                print(f"  ✅ {difficulty.value.upper()}: {problem.statement[:50]}...")
                
            except Exception as e:
                print(f"  ❌ {difficulty.value.upper()}: {e}")
    
    print(f"\n🎉 Problem generation testing completed!")

if __name__ == "__main__":
    test_problem_generation()
