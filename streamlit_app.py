"""
Streamlit Multi-Armed Bandit Interactive Visualization
======================================================

An interactive dashboard for comparing bandit strategies with adjustable parameters.

Run with:
    streamlit run streamlit_app.py
"""

import math
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

from strategies import (
    ABTestingStrategy,
    EpsilonGreedyStrategy,
    OptimisticInitialValuesStrategy,
    SoftmaxBoltzmannStrategy,
    ThompsonSamplingStrategy,
    UCB1Strategy,
)

# Set matplotlib backend for Streamlit
matplotlib.use("Agg")


# ============================================================================
# Streamlit App
# ==============================================================================


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
        
        # Chart 1: Performance-Stability Tradeoff (keep learning curve unchanged below)
        with chart_col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            names = list(results_dict.keys())
            rewards = [results_dict[n]["mean_total_reward"] for n in names]
            stds = [results_dict[n]["std_total_reward"] for n in names]
            efficiencies = [(r / optimal_reward) * 100 for r in rewards]

            scatter = ax.scatter(
                stds,
                rewards,
                c=efficiencies,
                cmap="viridis",
                s=140,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.6,
            )
            ax.axhline(
                optimal_reward,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Optimal = {optimal_reward:.0f}",
            )

            for i, name in enumerate(names):
                ax.annotate(
                    name,
                    (stds[i], rewards[i]),
                    xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Efficiency vs Optimal (%)")
            ax.set_xlabel("Std Dev of Total Reward (lower is more stable)")
            ax.set_ylabel("Mean Total Reward")
            ax.set_title("Performance vs Stability")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=9)
            
            st.pyplot(fig)
        
        # Chart 2: Efficiency Ranking
        with chart_col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            regrets = [optimal_reward - results_dict[n]["mean_total_reward"] for n in names]

            ranking = sorted(
                zip(names, rewards, regrets), key=lambda x: x[1], reverse=True
            )
            rank_names = [x[0] for x in ranking]
            rank_eff = [(x[1] / optimal_reward) * 100 for x in ranking]
            rank_regret = [x[2] for x in ranking]
            y = list(range(len(rank_names)))

            bars = ax.barh(y, rank_eff, color="teal", alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(rank_names, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 105)
            ax.set_xlabel("Efficiency (%)")
            ax.set_title("Efficiency Ranking (with Regret)")
            ax.grid(axis="x", alpha=0.25)

            for bar, eff, reg in zip(bars, rank_eff, rank_regret):
                ax.text(
                    min(eff + 0.8, 103.5),
                    bar.get_y() + bar.get_height() / 2,
                    f"{eff:.1f}% | R={reg:.0f}",
                    va="center",
                    fontsize=8,
                )
            
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
        
        # Chart 4: Arm Allocation Mix (100% stacked)
        st.subheader("🎯 Action Allocation Mix")
        fig, ax = plt.subplots(figsize=(12, 5))
        
        x = list(range(len(names)))
        pulls_a = [results_dict[n]["mean_pulls"]["A"] for n in names]
        pulls_b = [results_dict[n]["mean_pulls"]["B"] for n in names]
        pulls_c = [results_dict[n]["mean_pulls"]["C"] for n in names]

        totals = [a + b + c for a, b, c in zip(pulls_a, pulls_b, pulls_c)]
        share_a = [100 * a / t for a, t in zip(pulls_a, totals)]
        share_b = [100 * b / t for b, t in zip(pulls_b, totals)]
        share_c = [100 * c / t for c, t in zip(pulls_c, totals)]
        bottom_c = [a + b for a, b in zip(share_a, share_b)]

        ax.bar(x, share_a, label=f"A (mu={mu_a:.2f})", color="#4e79a7")
        ax.bar(x, share_b, bottom=share_a, label=f"B (mu={mu_b:.2f})", color="#f28e2b")
        ax.bar(x, share_c, bottom=bottom_c, label=f"C (mu={mu_c:.2f})", color="#59a14f")
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Allocation Share (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Exploration/Exploitation Mix by Strategy")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        
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
