"""
Mathematical Foundations and Regret Theory
===========================================

This module provides the theoretical foundations and mathematical formulas
for regret analysis in multi-armed bandits.

REFERENCES:
- Bubeck & Cesa-Bianchi "Regret Analysis of Stochastic and Nonstochastic 
  Multi-armed Bandit Problems" (2012)
- Lattimore & Szepesvári "Bandit Algorithms" (2020)
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class RegretTheory:
    """Mathematical analysis of regret in bandits."""
    
    @staticmethod
    def regret_definition():
        """Explain the mathematical definition of regret."""
        print("="*80)
        print("DEFINITION: REGRET IN MULTI-ARMED BANDITS")
        print("="*80)
        
        formula = r"""
REGRET(T) = E[μ* · T - Σ(t=1 to T) r_t]

Where:
  T      = Total number of pulls (time horizon)
  μ*     = Expected reward of optimal bandit (best action)
  r_t    = Reward received at time t
  E[ ]   = Expectation over randomness in algorithm & bandits

In simpler terms:
  REGRET = (Optimal reward if we knew best bandit) - (Actual reward earned)

PSEUDOREGRET (expected regret per pull):
  R̄(T) = Σ(i≠ i*) Δ_i · E[T_i(T)]
  
  Where:
    Δ_i     = μ* - μ_i = gap between optimal and bandit i
    E[T_i(T)] = expected number of times bandit i is pulled
    
This decomposes regret into:
  - How much worse each bandit is compared to best (Δ_i)
  - How many times we pull each suboptimal bandit (E[T_i(T)])
        """
        
        print(formula)
        
        return {
            'definition': 'REGRET(T) = E[μ* · T - Σ r_t]',
            'pseudoregret': 'R̄(T) = Σ(i≠i*) Δ_i · E[T_i(T)]'
        }
    
    @staticmethod
    def calculate_ab_test_regret_theoretical(
        mu_star: float,
        gaps: Dict[str, float],
        test_time: int,
        test_allocation: Dict[str, float],
        total_time: int
    ) -> Dict:
        """
        Calculate theoretical regret for A/B testing strategy.
        
        Args:
            mu_star: Expected return of best bandit
            gaps: Dictionary of regret gaps for each suboptimal bandit (μ* - μ_i)
            test_time: Number of time steps spent in testing phase
            test_allocation: Fraction of test time allocated to each bandit
            total_time: Total time horizon
        
        Returns:
            Regret calculation details
        """
        
        print("\n" + "="*80)
        print("A/B TESTING REGRET CALCULATION")
        print("="*80)
        
        print(f"\nParameters:")
        print(f"  Best bandit expected return: μ* = {mu_star:.2f}")
        print(f"  Gaps (μ* - μ_i): {gaps}")
        print(f"  Test time: {test_time}")
        print(f"  Test allocations: {test_allocation}")
        print(f"  Total time: {total_time}")
        
        # Phase 1: Testing phase regret
        test_regret = 0
        print(f"\nPhase 1: Testing Phase (t = 0 to {test_time})")
        
        for bandit, fraction in test_allocation.items():
            if fraction > 0:
                pulls = test_time * fraction
                gap = gaps[bandit]
                regret = pulls * gap
                test_regret += regret
                print(f"  {bandit}: {pulls:.0f} pulls × gap {gap:.3f} = regret {regret:.2f}")
        
        print(f"  Total testing phase regret: {test_regret:.2f}")
        
        # Phase 2: Exploitation phase regret
        # After testing, we select the best-observed bandit
        # In deterministic case, we definitely select the actually best one
        # But in stochastic case, we might misselect
        exploit_time = total_time - test_time
        exploit_regret = 0
        
        print(f"\nPhase 2: Exploitation Phase (t = {test_time} to {total_time})")
        print(f"  If selection is correct:")
        print(f"    Pulls of optimal bandit: {exploit_time}")
        print(f"    Regret in exploitation: 0")
        exploit_regret = 0
        
        print(f"  Total exploitation phase regret: {exploit_regret:.2f}")
        
        total_regret = test_regret + exploit_regret
        
        print(f"\nTotal Regret (A/B Testing): {total_regret:.2f}")
        
        return {
            'test_phase_regret': test_regret,
            'exploit_phase_regret': exploit_regret,
            'total_regret': total_regret,
            'regret_per_time': total_regret / total_time
        }
    
    @staticmethod
    def ucb_regret_bounds():
        """Prove and explain UCB regret bounds."""
        
        print("\n" + "="*80)
        print("UCB REGRET BOUNDS (THEORETICAL)")
        print("="*80)
        
        theorem = r"""
THEOREM (Auer et al. 2002): UCB Algorithm
For the UCB algorithm with parameter c, the regret satisfies:

    R(T) ≤ (5 + π²/3) · Σ(i: Δ_i > 0) (ln T) / Δ_i + O(Δ · ln ln T)

Where:
    Δ_i = μ* - μ_i (the gap for suboptimal bandit i)
    T = total time horizon
    ln T = natural logarithm of T

KEY PROPERTIES:
1. Logarithmic Regret: R(T) = O(log T)
   - This is OPTIMAL for the stochastic setting
   - Grows much slower than linear O(T) or polynomial regret

2. Gap-Dependent: Scales inversely with Δ_i
   - Larger gap between bandits → lower regret
   - Easier to eliminate bad bandits

3. Exploration Bonus: c · sqrt(ln(t) / n) structure
   - Balances empirical payoff with uncertainty
   - Automatically adjusts as we learn

COMPARISON WITH A/B TESTING:
- A/B tests bandit i for fixed time: Τ_i = T/k pulls
- Regret per suboptimal bandit: Δ_i · T/k
- Total regret: O(T) - linear in time!
- Much worse than O(log T) of UCB
        """
        
        print(theorem)
        
        return {
            'regret_bound': 'R(T) ≤ (5 + π²/3) · Σ (ln T) / Δ_i',
            'type': 'Logarithmic O(log T)',
            'optimality': 'Theoretically optimal for stochastic bandits'
        }
    
    @staticmethod
    def thompson_sampling_regret_bounds():
        """Explain Thompson Sampling regret bounds."""
        
        print("\n" + "="*80)
        print("THOMPSON SAMPLING REGRET BOUNDS (THEORETICAL)")
        print("="*80)
        
        theorem = r"""
THEOREM (Agrawal & Goyal 2012): Thompson Sampling
For Thompson Sampling with Beta-Bernoulli model:

    R(T) = O(k · ln(T) / Δ_min)

Where:
    k = number of bandits
    Δ_min = minimum gap (min over all i: μ* - μ_i)
    T = time horizon

KEY PROPERTIES:
1. Logarithmic Regret: Same O(log T) as UCB
   - Both are order-optimal

2. Posterior Updates: Bayesian inference
   - Maintains posterior distribution over arm rewards
   - Samples from posterior and picks highest sample
   - Automatically biased towards exploration when uncertain

3. Empirical Success: Often outperforms UCB in practice
   - Better constant factors
   - Handles non-stationary environments well
   - More stable in practice

INTUITION:
- Early on: Posterior has high variance → explores more
- Later: Posterior concentrates → exploits more
- No need for tuning c or ε parameters
        """
        
        print(theorem)
        
        return {
            'regret_bound': 'R(T) = O(k · ln(T) / Δ_min)',
            'type': 'Logarithmic O(log T)',
            'optimality': 'Order-optimal, often better in practice than UCB'
        }
    
    @staticmethod
    def regret_example_analysis():
        """Detailed example analysis with our specific bandits."""
        
        print("\n" + "="*80)
        print("REGRET ANALYSIS: OUR SCENARIO")
        print("="*80)
        
        # Our parameters
        mu_A = 0.8
        mu_B = 0.7
        mu_C = 0.5
        mu_star = mu_A
        
        delta_B = mu_star - mu_B
        delta_C = mu_star - mu_C
        
        T = 10000  # total budget
        
        print(f"\nBandits:")
        print(f"  A: μ = {mu_A} (optimal)")
        print(f"  B: μ = {mu_B}, Δ = {delta_B}")
        print(f"  C: μ = {mu_C}, Δ = {delta_C}")
        print(f"\nTime Horizon T = {T}")
        
        print("\n" + "-"*80)
        print("STRATEGY 1: A/B Testing")
        print("-"*80)
        
        # A/B test: $2000 allocated (1000 to A, 1000 to B), then $8000 to A
        test_phase_A = 1000
        test_phase_B = 1000
        test_phase = test_phase_A + test_phase_B
        exploit_phase = 8000
        
        # Regret from testing B
        regret_ab_test_B = test_phase_B * delta_B
        print(f"\nTest Phase Regret (B tested instead of A):")
        print(f"  {test_phase_B} pulls of B × Δ_B({delta_B}) = {regret_ab_test_B:.2f}")
        
        # Regret from not testing C (but C is not tested anyway, so no regret credit)
        print(f"\nExploitation Phase Regret:")
        print(f"  All {exploit_phase} pulls go to A (correct!)")
        print(f"  Regret = 0")
        
        total_regret_ab = regret_ab_test_B
        regret_proportion_ab = (total_regret_ab / (mu_star * T)) * 100
        
        print(f"\nTotal Regret (A/B): {total_regret_ab:.2f}")
        print(f"  As percentage of optimal reward: {regret_proportion_ab:.3f}%")
        
        print("\n" + "-"*80)
        print("STRATEGY 2: ε-Greedy (ε=0.1)")
        print("-"*80)
        
        # Approximate regret for ε-greedy
        # Mostly exploits, occasionally explores
        # Proportion of pulls of B ≈ ε × T / (k-1) where k is number of bandits after learning
        
        # Rough approximation: after learning, ε fraction goes to exploration
        # Of that, roughly distributed equally
        epsilon = 0.1
        exploration_budget = T * epsilon
        exploration_per_bad_bandit = exploration_budget / 2  # Split between B and C
        
        # But we learn early that B is worse, so convergence is faster
        # Conservative estimate: still test B quite a bit due to ε
        regret_epsilon_test_B = exploration_per_bad_bandit * delta_B
        
        total_regret_epsilon = regret_epsilon_test_B
        regret_proportion_epsilon = (total_regret_epsilon / (mu_star * T)) * 100
        
        print(f"\nEstimated regret from testing B:")
        print(f"  ~{exploration_per_bad_bandit:.0f} pulls × Δ_B({delta_B}) = {regret_epsilon_test_B:.2f}")
        print(f"\nTotal Regret (ε-Greedy): {total_regret_epsilon:.2f}")
        print(f"  As percentage of optimal reward: {regret_proportion_epsilon:.3f}%")
        
        print("\n" + "-"*80)
        print("STRATEGY 3: UCB (c=1.0)")
        print("-"*80)
        
        # UCB regret bound: (5 + π²/3) · Σ (ln T) / Δ_i
        c_const = 5 + (np.pi**2)/3  # ≈ 8.28
        ln_T = np.log(T)
        
        ucb_bound = c_const * (ln_T / delta_B + ln_T / delta_C)
        
        print(f"\nUCB Regret Bound: {c_const:.2f} · (ln({T})/{delta_B} + ln({T})/{delta_C})")
        print(f"  = {c_const:.2f} · ({ln_T:.2f}/{delta_B} + {ln_T:.2f}/{delta_C})")
        print(f"  = {c_const:.2f} · ({ln_T/delta_B:.2f} + {ln_T/delta_C:.2f})")
        print(f"  = {c_const:.2f} · {ln_T/delta_B + ln_T/delta_C:.2f}")
        print(f"  = {ucb_bound:.2f}")
        
        regret_proportion_ucb = (ucb_bound / (mu_star * T)) * 100
        
        print(f"\nTotal Regret (UCB Bound): {ucb_bound:.2f}")
        print(f"  As percentage of optimal reward: {regret_proportion_ucb:.3f}%")
        
        print("\n" + "-"*80)
        print("COMPARISON SUMMARY")
        print("-"*80)
        
        comparison_df = pd.DataFrame({
            'Strategy': ['A/B Testing', 'ε-Greedy', 'UCB'],
            'Regret': [total_regret_ab, total_regret_epsilon, ucb_bound],
            'As % of Optimal': [regret_proportion_ab, regret_proportion_epsilon, regret_proportion_ucb]
        })
        
        print("\n" + comparison_df.to_string(index=False))
        
        print(f"\nOptimal Reward: ${mu_star * T:,.2f}")
        
        return comparison_df


def main():
    """Run theoretical analysis."""
    
    # Part 1: Define regret
    RegretTheory.regret_definition()
    
    # Part 2: A/B Testing regret
    mu_star = 0.8
    gaps = {'B': 0.1, 'C': 0.3}
    test_time = 2000
    test_allocation = {'A': 0.5, 'B': 0.5}
    total_time = 10000
    
    RegretTheory.calculate_ab_test_regret_theoretical(
        mu_star, gaps, test_time, test_allocation, total_time
    )
    
    # Part 3: UCB bounds
    RegretTheory.ucb_regret_bounds()
    
    # Part 4: Thompson Sampling bounds
    RegretTheory.thompson_sampling_regret_bounds()
    
    # Part 5: Detailed example
    RegretTheory.regret_example_analysis()


if __name__ == "__main__":
    main()
