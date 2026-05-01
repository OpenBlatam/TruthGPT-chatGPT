"""
OpenClaw Embodied RL Agent — Pydantic-First Architecture.

An autonomous agent that uses the LLM as a "Policy" to decide which action
to take, interacting with a continuous environment and maximizing reward.
"""

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from ..arquitecturas_fundamentales.base_agent import BaseAgent
from ..models import AgentResponse, AgentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class RLConfig(BaseModel):
    """Configuration for the Embodied RL Agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_steps: int = Field(default=5, description="Maximum steps per episode")
    epsilon: float = Field(default=0.1, description="Epsilon-greedy exploration rate")
    target_performance: int = Field(default=100, description="Performance target to complete episode")
    initial_performance: int = Field(default=50, description="Starting performance metric")
    valid_actions: List[str] = Field(
        default=["optimize", "explore", "stop"],
        description="Allowed actions in the environment",
    )


class EnvTransition(BaseModel):
    """Typed result of an environment step."""
    next_state: str
    reward: float
    done: bool
    performance_metric: int


class TrajectoryStep(BaseModel):
    """A single step in the RL trajectory for structured logging."""
    step_num: int
    state_before: str
    performance_before: int
    action: str
    reward: float
    performance_after: int
    done: bool
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Structured Environment
# ---------------------------------------------------------------------------

class SimpleEnv(BaseModel):
    """A Pydantic-validated simulated environment for the RL Agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: str = "idle"
    performance_metric: int = 50

    def step(self, action: str) -> EnvTransition:
        """Execute action and return a typed transition."""
        prev_perf = self.performance_metric

        if action == "optimize":
            self.performance_metric += random.randint(5, 15)
            self.state = "optimizing"
            reward = 10.0
        elif action == "explore":
            self.state = "exploring"
            reward = 5.0
            if random.random() < 0.2:
                self.performance_metric -= 10
        elif action == "stop":
            self.state = "stopped"
            return EnvTransition(
                next_state=self.state, reward=0.0, done=True,
                performance_metric=self.performance_metric,
            )
        else:
            self.state = "error_state"
            reward = -5.0

        done = self.performance_metric >= 100
        return EnvTransition(
            next_state=self.state, reward=reward, done=done,
            performance_metric=self.performance_metric,
        )

    def reset(self, initial_performance: int = 50) -> None:
        """Reset environment to initial state."""
        self.state = "idle"
        self.performance_metric = initial_performance


# ---------------------------------------------------------------------------
# RL Agent
# ---------------------------------------------------------------------------

class RLAgent(BaseAgent):
    """
    OpenClaw Embodied RL Agent.

    Uses the LLM as a Policy to select actions in a structured environment,
    records typed trajectory steps, and returns Pydantic-validated results.
    """

    def __init__(
        self,
        config: AgentConfig = None,
        llm_engine: Optional[Any] = None,
        env: Any = None,
        rl_config: Optional[RLConfig] = None,
    ):
        super().__init__(
            name="EmbodiedRLAgent",
            role="Agente de Aprendizaje por Refuerzo y Optimización",
        )
        self.config = config
        self.rl_config = rl_config or RLConfig()
        self.llm = llm_engine or (config.llm_engine if config else None)
        self.env = env or SimpleEnv(performance_metric=self.rl_config.initial_performance)
        self.cumulative_reward = 0.0

    def _parse_action(self, raw_response: str) -> str:
        """Parse the LLM response into a valid action with epsilon-greedy fallback."""
        # Epsilon-greedy exploration
        if random.random() < self.rl_config.epsilon:
            action = random.choice(self.rl_config.valid_actions)
            logger.debug(f"[{self.name}] Epsilon-greedy exploration: {action}")
            return action

        try:
            clean_str = raw_response.replace("```json", "").replace("```", "").strip()
            decision = json.loads(clean_str)
            parsed = decision.get("action", "").lower()
            if parsed in self.rl_config.valid_actions:
                return parsed
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"[{self.name}] JSON parse failure. Fallback to explore. Raw: {raw_response[:80]}")

        return "explore"

    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Execute an RL episode and return structured results."""
        logger.info(f"{self.name} starting RL cycle with objective: {query}")

        trajectory: List[TrajectoryStep] = []

        for step_num in range(self.rl_config.max_steps):
            state_desc = f"State=({self.env.state}), Metric={self.env.performance_metric}"
            logger.info(f"[{self.name}] Observation step {step_num + 1}: {state_desc}")

            prompt = (
                f"Eres un agente inteligente RL optimizando un sistema autónomo.\n"
                f"Objetivo global: {query}\n"
                f"Observación actual: {state_desc}\n"
                "Responde ESTRICTAMENTE con un bloque JSON en el siguiente formato:\n"
                "{\"action\": \"optimize\" | \"explore\" | \"stop\"}"
            )

            try:
                perf_before = self.env.performance_metric
                state_before = self.env.state

                raw_action = await self.llm(prompt)
                action = self._parse_action(raw_action)
                logger.info(f"[{self.name}] Decoded action: {action}")

                transition = self.env.step(action)
                self.cumulative_reward += transition.reward

                trajectory.append(TrajectoryStep(
                    step_num=step_num + 1,
                    state_before=state_before,
                    performance_before=perf_before,
                    action=action,
                    reward=transition.reward,
                    performance_after=transition.performance_metric,
                    done=transition.done,
                ))

                if transition.done:
                    logger.info(f"[{self.name}] Episode completed. Metric >= {self.rl_config.target_performance}.")
                    break

            except Exception as e:
                return AgentResponse(
                    content=f"Critical error during RL simulation: {str(e)}",
                    action_type="final_answer",
                )

        # Build structured summary
        traj_lines = [
            f"[Step {t.step_num}: {t.performance_before}→{t.performance_after} | "
            f"Act: {t.action} | R: {t.reward}]"
            for t in trajectory
        ]

        summary = (
            f"**Simulación Embodied RL Completada**\n"
            f"- Objetivo Original: {query}\n"
            f"- Recompensa Total Acumulada: {self.cumulative_reward}\n"
            f"- Métrica de Rendimiento Final: {self.env.performance_metric}/{self.rl_config.target_performance}\n"
            f"- Pasos ejecutados: {len(trajectory)}/{self.rl_config.max_steps}\n\n"
            f"**Trayectoria de Decisiones:**\n" + " \n ".join(traj_lines)
        )

        self.add_to_memory("user", query)
        self.add_to_memory("assistant", summary)

        return AgentResponse(
            content=summary,
            action_type="final_answer",
            metadata={
                "cumulative_reward": self.cumulative_reward,
                "final_performance": self.env.performance_metric,
                "steps": len(trajectory),
                "trajectory": [t.model_dump() for t in trajectory],
            },
        )


