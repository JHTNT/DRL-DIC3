"""
Shared Bandit Strategies Module
================================

Common base classes and implementations for all bandit strategies.
Used by both streamlit_app.py and mab_algorithms_comparison.py.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseBanditStrategy(ABC):
    """Abstract base class for all bandit strategies."""

    def __init__(self, means: Dict[str, float], total_pulls: int, seed: int = 0):
        self.means = means
        self.total_pulls = total_pulls
        self.arms = list(means.keys())
        self.rng = random.Random(seed)

        self.counts = {a: 0 for a in self.arms}
        self.sum_rewards = {a: 0.0 for a in self.arms}
        self.total_reward = 0.0
        self.average_reward_curve: List[float] = []

    def sample_reward(self, arm: str) -> float:
        """Sample Bernoulli reward from arm."""
        return 1.0 if self.rng.random() < self.means[arm] else 0.0

    def empirical_mean(self, arm: str) -> float:
        """Compute empirical mean for arm."""
        n = self.counts[arm]
        return self.sum_rewards[arm] / n if n > 0 else 0.0

    @abstractmethod
    def select_arm(self, t: int) -> str:
        """Select which arm to pull at time step t."""
        pass

    def update(self, arm: str, reward: float) -> None:
        """Update counts and rewards after pulling arm."""
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.total_reward += reward
        step_count = sum(self.counts.values())
        self.average_reward_curve.append(self.total_reward / step_count)

    def run(self) -> dict:
        """Execute the strategy for total_pulls steps."""
        for t in range(1, self.total_pulls + 1):
            arm = self.select_arm(t)
            reward = self.sample_reward(arm)
            self.update(arm, reward)

        return {
            "total_reward": self.total_reward,
            "average_reward_curve": self.average_reward_curve,
            "pulls": self.counts.copy(),
        }


class ABTestingStrategy(BaseBanditStrategy):
    """A/B Testing: Equal split between A and B during test, then exploit best."""

    def __init__(
        self, means: Dict[str, float], total_pulls: int, ab_test_pulls: int = 2000, seed: int = 0
    ):
        super().__init__(means, total_pulls, seed)
        self.ab_test_pulls = ab_test_pulls
        self.test_arms = ["A", "B"]
        self.best_after_test: str | None = None

    def select_arm(self, t: int) -> str:
        if t <= self.ab_test_pulls:
            return self.test_arms[(t - 1) % 2]
        if self.best_after_test is None:
            mean_a = self.empirical_mean("A")
            mean_b = self.empirical_mean("B")
            self.best_after_test = "A" if mean_a >= mean_b else "B"
        return self.best_after_test


class OptimisticInitialValuesStrategy(BaseBanditStrategy):
    """Optimistic Initial Values: Initialize Q-values optimistically."""

    def __init__(
        self, means: Dict[str, float], total_pulls: int, initial_value: float = 1.0, seed: int = 0
    ):
        super().__init__(means, total_pulls, seed)
        self.q = {a: initial_value for a in self.arms}

    def select_arm(self, t: int) -> str:
        return max(self.arms, key=lambda a: self.q[a])

    def update(self, arm: str, reward: float) -> None:
        super().update(arm, reward)
        n = self.counts[arm]
        self.q[arm] += (reward - self.q[arm]) / n


class EpsilonGreedyStrategy(BaseBanditStrategy):
    """ε-Greedy: Explore with probability ε, exploit with probability 1-ε."""

    def __init__(self, means: Dict[str, float], total_pulls: int, epsilon: float = 0.1, seed: int = 0):
        super().__init__(means, total_pulls, seed)
        self.epsilon = epsilon

    def select_arm(self, t: int) -> str:
        if t <= len(self.arms):
            return self.arms[t - 1]
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.arms)
        return max(self.arms, key=lambda a: self.empirical_mean(a))


class SoftmaxBoltzmannStrategy(BaseBanditStrategy):
    """Softmax (Boltzmann): Probabilistic action selection with temperature."""

    def __init__(
        self, means: Dict[str, float], total_pulls: int, temperature: float = 0.1, seed: int = 0
    ):
        super().__init__(means, total_pulls, seed)
        self.temperature = max(temperature, 1e-6)

    def select_arm(self, t: int) -> str:
        if t <= len(self.arms):
            return self.arms[t - 1]
        logits = [self.empirical_mean(a) / self.temperature for a in self.arms]
        max_logit = max(logits)
        exps = [math.exp(x - max_logit) for x in logits]
        z = sum(exps)
        probs = [e / z for e in exps]
        r = self.rng.random()
        cdf = 0.0
        for arm, p in zip(self.arms, probs):
            cdf += p
            if r <= cdf:
                return arm
        return self.arms[-1]


class UCB1Strategy(BaseBanditStrategy):
    """Upper Confidence Bound: Optimism under uncertainty."""

    def __init__(self, means: Dict[str, float], total_pulls: int, c: float = 2.0, seed: int = 0):
        super().__init__(means, total_pulls, seed)
        self.c = c

    def select_arm(self, t: int) -> str:
        if t <= len(self.arms):
            return self.arms[t - 1]

        def ucb_value(a: str) -> float:
            avg = self.empirical_mean(a)
            bonus = self.c * math.sqrt(math.log(t) / self.counts[a])
            return avg + bonus

        return max(self.arms, key=ucb_value)


class ThompsonSamplingStrategy(BaseBanditStrategy):
    """Thompson Sampling: Bayesian posterior sampling."""

    def __init__(self, means: Dict[str, float], total_pulls: int, seed: int = 0):
        super().__init__(means, total_pulls, seed)
        self.alpha = {a: 1.0 for a in self.arms}
        self.beta = {a: 1.0 for a in self.arms}

    def select_arm(self, t: int) -> str:
        samples = {
            a: self.rng.betavariate(self.alpha[a], self.beta[a]) for a in self.arms
        }
        return max(self.arms, key=lambda a: samples[a])

    def update(self, arm: str, reward: float) -> None:
        super().update(arm, reward)
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward
