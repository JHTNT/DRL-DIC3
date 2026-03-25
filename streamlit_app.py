"""
Streamlit Multi-Armed Bandit Interactive Visualization
======================================================

An interactive dashboard for comparing bandit strategies with adjustable parameters.

Run with:
    streamlit run streamlit_app.py
"""

import math
import os
import random
from typing import Dict, List

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from abc import ABC, abstractmethod

# Set matplotlib backend for Streamlit
matplotlib.use("Agg")


# ============================================================================
# Strategy Classes (identical to mab_algorithms_comparison.py)
# ============================================================================

class BaseBanditStrategy(ABC):
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
        return 1.0 if self.rng.random() < self.means[arm] else 0.0

    def empirical_mean(self, arm: str) -> float:
        n = self.counts[arm]
        return self.sum_rewards[arm] / n if n > 0 else 0.0

    @abstractmethod
    def select_arm(self, t: int) -> str:
        pass

    def update(self, arm: str, reward: float) -> None:
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.total_reward += reward
        step_count = sum(self.counts.values())
        self.average_reward_curve.append(self.total_reward / step_count)

    def run(self) -> dict:
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
    def __init__(
        self,
        means: Dict[str, float],
        total_pulls: int,
        ab_test_pulls: int = 2000,
        seed: int = 0,
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
    def __init__(
        self,
        means: Dict[str, float],
        total_pulls: int,
        initial_value: float = 1.0,
        seed: int = 0,
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
    def __init__(
        self,
        means: Dict[str, float],
        total_pulls: int,
        epsilon: float = 0.1,
        seed: int = 0,
    ):
        super().__init__(means, total_pulls, seed)
        self.epsilon = epsilon

    def select_arm(self, t: int) -> str:
        if t <= len(self.arms):
            return self.arms[t - 1]
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.arms)
        return max(self.arms, key=lambda a: self.empirical_mean(a))


class SoftmaxBoltzmannStrategy(BaseBanditStrategy):
    def __init__(
        self,
        means: Dict[str, float],
        total_pulls: int,
        temperature: float = 0.1,
        seed: int = 0,
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
    def __init__(
        self,
        means: Dict[str, float],
        total_pulls: int,
        c: float = 2.0,
        seed: int = 0,
    ):
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


# ============================================================================
# Streamlit App
# ============================================================================


def main():
    st.set_page_config(page_title="Bandit Strategy Comparison", layout="wide")
    
    st.title("🎰 Multi-Armed Bandit Strategy Comparison")
    st.markdown(
        """
        Interactive comparison of different bandit strategies with real-time parameter tuning.
        Adjust settings in the sidebar and click **Run Simulation** to see results.
        """
    )

    # ========================================================================
    # SIDEBAR: Parameter Controls
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Bandit Parameters")
        mu_a = st.slider("μ_A (Bandit A mean)", 0.0, 1.0, 0.8, 0.05)
        mu_b = st.slider("μ_B (Bandit B mean)", 0.0, 1.0, 0.7, 0.05)
        mu_c = st.slider("μ_C (Bandit C mean)", 0.0, 1.0, 0.5, 0.05)
        
        st.subheader("Simulation Settings")
        total_pulls = st.slider(
            "Total Pulls (Budget)",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
        )
        n_runs = st.slider(
            "Number of Runs (Monte Carlo)",
            min_value=10,
            max_value=500,
            value=200,
            step=10,
        )
        
        st.subheader("Strategy Hyperparameters")
        
        col1, col2 = st.columns(2)
        with col1:
            ab_test_frac = st.slider(
                "A/B Test Fraction",
                0.0,
                1.0,
                0.2,
                0.05,
                help="Fraction of pulls to spend on A/B testing before exploitation",
            )
            eps = st.slider(
                "ε-Greedy ε",
                0.0,
                0.5,
                0.1,
                0.01,
                help="Exploration probability",
            )
            ucb_c = st.slider(
                "UCB Confidence c",
                0.5,
                5.0,
                2.0,
                0.25,
                help="Confidence multiplier for exploration bonus",
            )
        
        with col2:
            oiv_initial = st.slider(
                "OIV Initial Value",
                0.0,
                2.0,
                1.0,
                0.1,
                help="Optimistic initial Q-value",
            )
            softmax_temp = st.slider(
                "Softmax Temperature",
                0.01,
                1.0,
                0.1,
                0.01,
                help="Temperature for softmax action selection",
            )
        
        run_button = st.button(
            "🚀 Run Simulation",
            use_container_width=True,
            type="primary"
        )
    
    # ========================================================================
    # MAIN: Results Display
    # ========================================================================
    if run_button:
        means = {"A": mu_a, "B": mu_b, "C": mu_c}
        ab_test_pulls = int(total_pulls * ab_test_frac)
        
        st.info(f"Running {n_runs} × {total_pulls} pulls simulation with bandits {means}...")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run each strategy
        results_dict = {}
        strategies_to_run = [
            ("A/B Testing", ABTestingStrategy, {"ab_test_pulls": ab_test_pulls}),
            ("Optimistic Initial Values", OptimisticInitialValuesStrategy, {"initial_value": oiv_initial}),
            ("ε-Greedy", EpsilonGreedyStrategy, {"epsilon": eps}),
            ("Softmax (Boltzmann)", SoftmaxBoltzmannStrategy, {"temperature": softmax_temp}),
            ("UCB", UCB1Strategy, {"c": ucb_c}),
            ("Thompson Sampling", ThompsonSamplingStrategy, {}),
        ]
        
        for idx, (name, strategy_cls, kwargs) in enumerate(strategies_to_run):
            status_text.write(f"Running {name}... ({idx+1}/{len(strategies_to_run)})")
            
            run_rewards = []
            run_pulls = []
            avg_curve_sum = [0.0] * total_pulls
            
            for i in range(n_runs):
                seed = 20260325 + i
                strategy = strategy_cls(means, total_pulls, seed=seed, **kwargs)
                result = strategy.run()
                
                run_rewards.append(result["total_reward"])
                run_pulls.append(result["pulls"])
                for t, value in enumerate(result["average_reward_curve"]):
                    avg_curve_sum[t] += value
            
            mean_reward = sum(run_rewards) / n_runs
            variance = sum((x - mean_reward) ** 2 for x in run_rewards) / n_runs
            std_reward = math.sqrt(variance)
            
            mean_pulls = {
                a: sum(p[a] for p in run_pulls) / n_runs for a in means.keys()
            }
            mean_avg_curve = [x / n_runs for x in avg_curve_sum]
            
            results_dict[name] = {
                "mean_total_reward": mean_reward,
                "std_total_reward": std_reward,
                "mean_pulls": mean_pulls,
                "mean_average_curve": mean_avg_curve,
            }
            
            progress_bar.progress((idx + 1) / len(strategies_to_run))
        
        status_text.empty()
        progress_bar.empty()
        
        st.success("✅ Simulation complete!")
        
        # ====================================================================
        # RESULTS DISPLAY
        # ====================================================================
        
        optimal_arm = max(means, key=lambda a: means[a])
        optimal_reward = total_pulls * means[optimal_arm]
        
        # Metrics Cards
        st.subheader("📊 Summary Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Optimal Expected Reward", f"{optimal_reward:.0f}")
        with col2:
            st.metric("Best Strategy", 
                     max(results_dict, key=lambda x: results_dict[x]["mean_total_reward"]))
        with col3:
            best_reward = max(r["mean_total_reward"] for r in results_dict.values())
            st.metric("Best Mean Reward", f"{best_reward:.0f}")
        
        # Results Table
        st.subheader("📈 Strategy Results")
        
        rows = []
        for name, res in results_dict.items():
            regret = optimal_reward - res["mean_total_reward"]
            rows.append({
                "Strategy": name,
                "Mean Reward": f"{res['mean_total_reward']:.2f}",
                "Std Dev": f"{res['std_total_reward']:.2f}",
                "Regret": f"{regret:.2f}",
                "A pulls": f"{res['mean_pulls']['A']:.1f}",
                "B pulls": f"{res['mean_pulls']['B']:.1f}",
                "C pulls": f"{res['mean_pulls']['C']:.1f}",
            })
        
        st.dataframe(rows, use_container_width=True)
        
        # ====================================================================
        # CHARTS
        # ====================================================================
        
        st.subheader("📊 Visualizations")
        
        chart_col1, chart_col2 = st.columns(2)
        
        # Chart 1: Mean Reward Comparison
        with chart_col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            names = list(results_dict.keys())
            rewards = [results_dict[n]["mean_total_reward"] for n in names]
            stds = [results_dict[n]["std_total_reward"] for n in names]
            
            bars = ax.bar(range(len(names)), rewards, yerr=stds, capsize=5, color="steelblue")
            ax.axhline(optimal_reward, color="red", linestyle="--", linewidth=2,
                      label=f"Optimal = {optimal_reward:.0f}")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
            ax.set_ylabel("Mean Total Reward")
            ax.set_title("Strategy Performance: Mean Reward")
            ax.legend()
            
            for b, v in zip(bars, rewards):
                ax.text(b.get_x() + b.get_width() / 2, v + 50, f"{v:.0f}",
                       ha="center", fontsize=9)
            
            st.pyplot(fig)
        
        # Chart 2: Regret Comparison
        with chart_col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            regrets = [optimal_reward - results_dict[n]["mean_total_reward"] for n in names]
            
            bars = ax.bar(range(len(names)), regrets, color="coral")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
            ax.set_ylabel("Regret vs Optimal")
            ax.set_title("Strategy Performance: Regret")
            
            for b, v in zip(bars, regrets):
                ax.text(b.get_x() + b.get_width() / 2, v + 10, f"{v:.0f}",
                       ha="center", fontsize=9)
            
            st.pyplot(fig)
        
        # Chart 3: Learning Curves
        st.subheader("📈 Learning Curves (Mean Average Reward per Pull)")
        fig, ax = plt.subplots(figsize=(12, 5))
        
        for name in names:
            ax.plot(results_dict[name]["mean_average_curve"], label=name, linewidth=1.5)
        
        ax.set_xlabel("Pull (Time Step)")
        ax.set_ylabel("Mean Reward per Pull")
        ax.set_title("Learning Progress: How Average Reward Improves Over Time")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Chart 4: Arm Allocation
        st.subheader("🎯 Action Allocation")
        fig, ax = plt.subplots(figsize=(12, 5))
        
        x = list(range(len(names)))
        width = 0.25
        pulls_a = [results_dict[n]["mean_pulls"]["A"] for n in names]
        pulls_b = [results_dict[n]["mean_pulls"]["B"] for n in names]
        pulls_c = [results_dict[n]["mean_pulls"]["C"] for n in names]
        
        ax.bar([i - width for i in x], pulls_a, width=width, label="A", color="skyblue")
        ax.bar(x, pulls_b, width=width, label="B", color="lightcoral")
        ax.bar([i + width for i in x], pulls_c, width=width, label="C", color="lightgreen")
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Average Pull Count")
        ax.set_title(f"How Each Strategy Allocates {total_pulls} Pulls Across Arms")
        ax.legend()
        
        st.pyplot(fig)
        
        # ====================================================================
        # DETAILED ANALYSIS
        # ====================================================================
        
        st.subheader("🔍 Strategy Rankings")
        
        rankings = sorted(
            results_dict.items(),
            key=lambda x: x[1]["mean_total_reward"],
            reverse=True
        )
        
        for rank, (name, res) in enumerate(rankings, 1):
            regret = optimal_reward - res["mean_total_reward"]
            with st.expander(f"{rank}. {name}"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Reward", f"{res['mean_total_reward']:.1f}")
                with col2:
                    st.metric("Std Dev", f"{res['std_total_reward']:.1f}")
                with col3:
                    st.metric("Regret", f"{regret:.1f}")
                with col4:
                    efficiency = (res['mean_total_reward'] / optimal_reward) * 100
                    st.metric("Efficiency", f"{efficiency:.1f}%")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**A Pulls:** {res['mean_pulls']['A']:.0f}")
                with col2:
                    st.write(f"**B Pulls:** {res['mean_pulls']['B']:.0f}")
                with col3:
                    st.write(f"**C Pulls:** {res['mean_pulls']['C']:.0f}")


if __name__ == "__main__":
    main()
