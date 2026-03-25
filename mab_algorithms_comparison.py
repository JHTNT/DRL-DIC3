"""
Bandit Strategies Comparison with Chart Visualization
====================================================

This module compares the following strategies under a stochastic Bernoulli setup:
1. A/B Testing
2. Optimistic Initial Values
3. epsilon-Greedy
4. Softmax (Boltzmann)
5. Upper Confidence Bound (UCB1)
6. Thompson Sampling

Bandit means:
- A: 0.8
- B: 0.7
- C: 0.5

Budget:
- Total pulls: 10,000 (1 pull = allocate $1)

Output:
- Console summary table
- Charts saved to ./charts/
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Type

import matplotlib.pyplot as plt

from strategies import (
    ABTestingStrategy,
    EpsilonGreedyStrategy,
    OptimisticInitialValuesStrategy,
    SoftmaxBoltzmannStrategy,
    ThompsonSamplingStrategy,
    UCB1Strategy,
)


@dataclass
class StrategyRunResult:
    name: str
    total_reward: float
    average_reward_curve: List[float]
    pulls: Dict[str, int]


@dataclass
class StrategyAggregateResult:
    name: str
    mean_total_reward: float
    std_total_reward: float
    regret_vs_optimal: float
    mean_pulls: Dict[str, float]
    mean_average_reward_curve: List[float]





class BanditComparison:
    def __init__(self, means: Dict[str, float], total_pulls: int, n_runs: int = 200):
        self.means = means
        self.total_pulls = total_pulls
        self.n_runs = n_runs
        self.optimal_arm = max(means, key=lambda a: means[a])
        self.optimal_expected_reward = total_pulls * means[self.optimal_arm]

    def evaluate_strategy(
        self,
        name: str,
        strategy_cls: Type,
        **kwargs,
    ) -> StrategyAggregateResult:
        run_rewards: List[float] = []
        run_pulls: List[Dict[str, int]] = []
        avg_curve_sum = [0.0] * self.total_pulls

        for i in range(self.n_runs):
            seed = 20260325 + i
            strategy = strategy_cls(self.means, self.total_pulls, seed=seed, **kwargs)
            result = strategy.run()

            run_rewards.append(result["total_reward"])
            run_pulls.append(result["pulls"])
            for t, value in enumerate(result.average_reward_curve):
                avg_curve_sum[t] += value

        mean_reward = sum(run_rewards) / self.n_runs
        variance = sum((x - mean_reward) ** 2 for x in run_rewards) / self.n_runs
        std_reward = math.sqrt(variance)

        mean_pulls = {
            a: sum(p[a] for p in run_pulls) / self.n_runs for a in self.means.keys()
        }
        mean_average_curve = [x / self.n_runs for x in avg_curve_sum]

        return StrategyAggregateResult(
            name=name,
            mean_total_reward=mean_reward,
            std_total_reward=std_reward,
            regret_vs_optimal=self.optimal_expected_reward - mean_reward,
            mean_pulls=mean_pulls,
            mean_average_reward_curve=mean_average_curve,
        )

    def run_all(self) -> List[StrategyAggregateResult]:
        specs = [
            ("A/B Testing", ABTestingStrategy, {"ab_test_pulls": 2000}),
            (
                "Optimistic Initial Values",
                OptimisticInitialValuesStrategy,
                {"initial_value": 1.0},
            ),
            ("epsilon-Greedy", EpsilonGreedyStrategy, {"epsilon": 0.1}),
            ("Softmax (Boltzmann)", SoftmaxBoltzmannStrategy, {"temperature": 0.1}),
            ("UCB", UCB1Strategy, {"c": 2.0}),
            ("Thompson Sampling", ThompsonSamplingStrategy, {}),
        ]

        results: List[StrategyAggregateResult] = []
        for name, cls, kwargs in specs:
            print(f"Running: {name}")
            results.append(self.evaluate_strategy(name, cls, **kwargs))

        results.sort(key=lambda x: x.mean_total_reward, reverse=True)
        return results

    def print_summary(self, results: List[StrategyAggregateResult]) -> None:
        print("\n" + "=" * 95)
        print("Strategy Comparison (Monte Carlo Average)")
        print("=" * 95)
        print(
            f"Bandit means: {self.means}, total pulls: {self.total_pulls}, runs: {self.n_runs}"
        )
        print(f"Optimal expected reward: {self.optimal_expected_reward:.2f}\n")

        header = (
            f"{'Strategy':30s} {'Mean Reward':>12s} {'Std':>10s} "
            f"{'Regret':>10s} {'A pulls':>10s} {'B pulls':>10s} {'C pulls':>10s}"
        )
        print(header)
        print("-" * len(header))

        for r in results:
            print(
                f"{r.name:30s} "
                f"{r.mean_total_reward:12.2f} "
                f"{r.std_total_reward:10.2f} "
                f"{r.regret_vs_optimal:10.2f} "
                f"{r.mean_pulls['A']:10.1f} "
                f"{r.mean_pulls['B']:10.1f} "
                f"{r.mean_pulls['C']:10.1f}"
            )

    def plot_results(self, results: List[StrategyAggregateResult]) -> None:
        os.makedirs("charts", exist_ok=True)

        names = [r.name for r in results]
        rewards = [r.mean_total_reward for r in results]
        stds = [r.std_total_reward for r in results]
        regrets = [r.regret_vs_optimal for r in results]

        # Chart 1: Mean reward + optimal line.
        plt.figure(figsize=(11, 6))
        bars = plt.bar(range(len(names)), rewards, yerr=stds, capsize=5)
        plt.axhline(
            self.optimal_expected_reward,
            color="red",
            linestyle="--",
            label=f"Optimal expected reward = {self.optimal_expected_reward:.0f}",
        )
        plt.xticks(range(len(names)), names, rotation=20, ha="right")
        plt.ylabel("Mean Total Reward")
        plt.title("Bandit Strategy Comparison: Mean Reward")
        plt.legend()
        for b, v in zip(bars, rewards):
            plt.text(b.get_x() + b.get_width() / 2, v + 15, f"{v:.0f}", ha="center")
        plt.tight_layout()
        plt.savefig("charts/strategy_mean_reward.png", dpi=150)
        plt.close()

        # Chart 2: Regret bar chart.
        plt.figure(figsize=(11, 6))
        bars = plt.bar(range(len(names)), regrets)
        plt.xticks(range(len(names)), names, rotation=20, ha="right")
        plt.ylabel("Regret vs Optimal")
        plt.title("Bandit Strategy Comparison: Regret")
        for b, v in zip(bars, regrets):
            plt.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.0f}", ha="center")
        plt.tight_layout()
        plt.savefig("charts/strategy_regret.png", dpi=150)
        plt.close()

        # Chart 3: Mean average reward over time.
        plt.figure(figsize=(11, 6))
        for r in results:
            plt.plot(r.mean_average_reward_curve, label=r.name, linewidth=1.5)
        plt.xlabel("Pull (Time Step)")
        plt.ylabel("Mean Reward per Pull")
        plt.title("Learning Curves: Mean Average Reward per Pull")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("charts/learning_curves.png", dpi=150)
        plt.close()

        # Chart 4: Average pulls per arm for each strategy.
        plt.figure(figsize=(11, 6))
        x = list(range(len(names)))
        width = 0.25
        pulls_a = [r.mean_pulls["A"] for r in results]
        pulls_b = [r.mean_pulls["B"] for r in results]
        pulls_c = [r.mean_pulls["C"] for r in results]

        plt.bar([i - width for i in x], pulls_a, width=width, label="A")
        plt.bar(x, pulls_b, width=width, label="B")
        plt.bar([i + width for i in x], pulls_c, width=width, label="C")
        plt.xticks(x, names, rotation=20, ha="right")
        plt.ylabel("Average Pull Count")
        plt.title("Action Allocation by Strategy")
        plt.legend()
        plt.tight_layout()
        plt.savefig("charts/arm_allocation.png", dpi=150)
        plt.close()

        print("\nCharts saved to ./charts/")
        print("- charts/strategy_mean_reward.png")
        print("- charts/strategy_regret.png")
        print("- charts/learning_curves.png")
        print("- charts/arm_allocation.png")


def main() -> None:
    means = {"A": 0.8, "B": 0.7, "C": 0.5}
    total_pulls = 10000
    n_runs = 200

    print("=" * 95)
    print("Bandit Strategy Benchmark with Charts")
    print("=" * 95)

    comparison = BanditComparison(means, total_pulls, n_runs=n_runs)
    results = comparison.run_all()
    comparison.print_summary(results)
    comparison.plot_results(results)


if __name__ == "__main__":
    main()
