"""
Multi-Armed Bandit with A/B Testing Analysis
==============================================

This module analyzes a bandit problem where we have a total budget of $10,000
to allocate across three bandits (A, B, C) with different expected returns.

We compare an A/B testing strategy with an optimal strategy that knows
the true returns of all bandits.

TRUE PARAMETERS (Known for this analysis):
- Bandit A: μ_A = 0.8 (80% expected return per dollar invested)
- Bandit B: μ_B = 0.7 (70% expected return per dollar invested)  
- Bandit C: μ_C = 0.5 (50% expected return per dollar invested)

TOTAL BUDGET: $10,000

ASSUMPTIONS:
1. Returns are deterministic (no variance) for analytical estimates
2. Each dollar invested yields the expected return multiplied by the investment
3. A/B test phase: allocate $2,000 equally between A and B ($1,000 each)
4. After A/B test, use pure exploitation (allocate all remaining budget to best performer)
5. Exploration of C is ignored in the A/B phase
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


class MultiArmedBanditAnalysis:
    """Analyzes different strategies for allocating budget across bandits."""
    
    def __init__(self, true_returns: Dict[str, float], total_budget: float):
        """
        Initialize the MAB analysis.
        
        Args:
            true_returns: Dictionary with bandit names and their true expected returns
            total_budget: Total budget to allocate
        """
        self.true_returns = true_returns
        self.total_budget = total_budget
        self.bandits = list(true_returns.keys())
        
    def ab_test_phase(self, ab_budget: float, test_split: Dict[str, float]) -> Dict:
        """
        Simulate the A/B testing phase.
        
        Args:
            ab_budget: Budget allocated for A/B testing
            test_split: Dictionary specifying budget split (e.g., {'A': 0.5, 'B': 0.5})
        
        Returns:
            Dictionary with test results including rewards and observations
        """
        print("\n" + "="*70)
        print("TASK 1: A/B TEST PHASE ANALYSIS")
        print("="*70)
        
        print(f"\nPhase Budget: ${ab_budget:,.2f}")
        print(f"Budget Split: {test_split}")
        
        results = {
            'phase': 'A/B Testing',
            'budget': ab_budget,
            'allocations': {},
            'rewards': {},
            'observations': {}
        }
        
        total_reward = 0
        
        for bandit, fraction in test_split.items():
            if fraction > 0:
                allocation = ab_budget * fraction
                # Deterministic reward: investment × expected return
                reward = allocation * self.true_returns[bandit]
                
                results['allocations'][bandit] = allocation
                results['rewards'][bandit] = reward
                results['observations'][bandit] = {
                    'allocation': allocation,
                    'true_return_rate': self.true_returns[bandit],
                    'observed_return': reward,
                    'estimated_mean': self.true_returns[bandit]  # With deterministic returns
                }
                
                total_reward += reward
                
                print(f"\n{bandit}:")
                print(f"  Allocation: ${allocation:,.2f}")
                print(f"  True Return Rate: {self.true_returns[bandit]:.1%}")
                print(f"  Reward Earned: ${reward:,.2f}")
        
        results['total_reward'] = total_reward
        
        print(f"\nTotal Reward from A/B Phase: ${total_reward:,.2f}")
        
        return results
    
    def select_best_from_test(self, ab_results: Dict) -> Tuple[str, float]:
        """
        Task 2: Determine which bandit to exploit based on A/B test results.
        
        Args:
            ab_results: Results dictionary from ab_test_phase
        
        Returns:
            Tuple of (best_bandit_name, estimated_return_rate)
        """
        print("\n" + "="*70)
        print("TASK 2: BANDIT SELECTION FROM A/B TEST")
        print("="*70)
        
        # With deterministic returns, we select based on observed performance
        best_bandit = None
        best_reward_rate = -1
        
        for bandit, obs in ab_results['observations'].items():
            estimated_mean = obs['estimated_mean']
            print(f"\n{bandit}: Estimated return rate = {estimated_mean:.1%}")
            
            if estimated_mean > best_reward_rate:
                best_reward_rate = estimated_mean
                best_bandit = bandit
        
        print(f"\n>>> Selected Bandit for Exploitation: {best_bandit}")
        print(f">>> Estimated Return Rate: {best_reward_rate:.1%}")
        
        return best_bandit, best_reward_rate
    
    def exploitation_phase(self, best_bandit: str, remaining_budget: float) -> Dict:
        """
        Task 3: Allocate remaining budget using pure exploitation.
        
        Args:
            best_bandit: Name of bandit to exploit
            remaining_budget: Budget left after A/B testing
        
        Returns:
            Dictionary with exploitation phase results
        """
        print("\n" + "="*70)
        print("TASK 3: EXPLOITATION PHASE")
        print("="*70)
        
        print(f"\nRemaining Budget: ${remaining_budget:,.2f}")
        print(f"Exploitation Bandit: {best_bandit}")
        print(f"Expected Return Rate: {self.true_returns[best_bandit]:.1%}")
        
        # Allocate all remaining budget to the best bandit
        reward = remaining_budget * self.true_returns[best_bandit]
        
        exploitation_results = {
            'phase': 'Pure Exploitation',
            'bandit': best_bandit,
            'budget': remaining_budget,
            'expected_return_rate': self.true_returns[best_bandit],
            'reward': reward
        }
        
        print(f"\nAllocation to {best_bandit}: ${remaining_budget:,.2f}")
        print(f"Reward Earned: ${reward:,.2f}")
        
        return exploitation_results
    
    def ab_testing_strategy(self, ab_budget: float, ab_split: Dict[str, float]) -> Dict:
        """
        Complete A/B testing strategy: exploration phase + exploitation phase.
        
        Returns:
            Dictionary with complete results
        """
        # Phase 1: A/B Testing
        ab_results = self.ab_test_phase(ab_budget, ab_split)
        
        # Phase 2: Select best
        best_bandit, _ = self.select_best_from_test(ab_results)
        
        # Phase 3: Exploitation
        remaining_budget = self.total_budget - ab_budget
        exp_results = self.exploitation_phase(best_bandit, remaining_budget)
        
        total_reward = ab_results['total_reward'] + exp_results['reward']
        
        return {
            'strategy': 'A/B Testing',
            'ab_phase': ab_results,
            'exploitation_phase': exp_results,
            'total_reward': total_reward
        }
    
    def optimal_strategy(self) -> Dict:
        """
        Task 4: Optimal strategy - allocate all budget to best bandit from the start.
        
        Returns:
            Dictionary with optimal strategy results
        """
        print("\n" + "="*70)
        print("TASK 4: OPTIMAL STRATEGY (ORACLE)")
        print("="*70)
        
        print(f"\nAssuming we know true returns from the start:")
        
        best_bandit = max(self.true_returns, key=self.true_returns.get)
        best_return = self.true_returns[best_bandit]
        
        print(f"Best Bandit: {best_bandit} (return rate: {best_return:.1%})")
        print(f"Total Budget: ${self.total_budget:,.2f}")
        
        optimal_reward = self.total_budget * best_return
        
        print(f"\nOptimal Strategy: Allocate all ${self.total_budget:,.2f} to {best_bandit}")
        print(f"Optimal Total Reward: ${optimal_reward:,.2f}")
        
        return {
            'strategy': 'Optimal (Oracle)',
            'best_bandit': best_bandit,
            'allocation': {best_bandit: self.total_budget},
            'total_reward': optimal_reward
        }
    
    def compute_regret(self, ab_strategy_result: Dict, optimal_result: Dict) -> Dict:
        """
        Task 5: Compute the regret of A/B testing vs optimal strategy.
        
        Regret = Optimal Reward - Actual Reward
        
        Args:
            ab_strategy_result: Results from ab_testing_strategy
            optimal_result: Results from optimal_strategy
        
        Returns:
            Dictionary with regret analysis
        """
        print("\n" + "="*70)
        print("TASK 5: REGRET ANALYSIS")
        print("="*70)
        
        optimal_reward = optimal_result['total_reward']
        actual_reward = ab_strategy_result['total_reward']
        
        regret = optimal_reward - actual_reward
        regret_percent = (regret / optimal_reward) * 100
        
        print(f"\nOptimal Reward: ${optimal_reward:,.2f}")
        print(f"A/B Testing Reward: ${actual_reward:,.2f}")
        print(f"Regret (Absolute): ${regret:,.2f}")
        print(f"Regret (Percentage): {regret_percent:.2f}%")
        
        # Detailed breakdown
        print(f"\n--- Regret Breakdown ---")
        
        # What was the cost of testing C-bandits instead of focusing on best?
        ab_phase_budget = ab_strategy_result['ab_phase']['budget']
        ab_bandits_tested = list(ab_strategy_result['ab_phase']['observations'].keys())
        exploited_bandit = ab_strategy_result['exploitation_phase']['bandit']
        
        print(f"\nA/B Testing Phase:")
        print(f"  Allocated: ${ab_phase_budget:,.2f} to test B bandits {ab_bandits_tested}")
        print(f"  Optimal would have: ${ab_phase_budget * optimal_result['total_reward'] / self.total_budget:,.2f}")
        
        optimal_allocation_to_best = ab_phase_budget * self.true_returns[optimal_result['best_bandit']]
        actual_ab_to_best = ab_strategy_result['ab_phase']['total_reward']
        exploration_cost = optimal_allocation_to_best - actual_ab_to_best
        
        print(f"  Exploration Cost: ${exploration_cost:,.2f}")
        
        return {
            'optimal_reward': optimal_reward,
            'actual_reward': actual_reward,
            'regret_absolute': regret,
            'regret_percentage': regret_percent,
            'exploration_cost': exploration_cost
        }
    
    def print_summary(self, ab_result: Dict, optimal_result: Dict, regret_result: Dict):
        """Print a comprehensive summary of all results."""
        
        print("\n" + "="*70)
        print("COMPREHENSIVE SUMMARY")
        print("="*70)
        
        summary_df = pd.DataFrame({
            'Strategy': ['A/B Testing', 'Optimal (Oracle)'],
            'Total Reward': [f"${ab_result['total_reward']:,.2f}", 
                            f"${optimal_result['total_reward']:,.2f}"],
            'Regret': [f"${regret_result['regret_absolute']:,.2f}", '$0.00']
        })
        
        print("\n" + summary_df.to_string(index=False))
        
        print(f"\nRegret Analysis:")
        print(f"  Absolute Regret: ${regret_result['regret_absolute']:,.2f}")
        print(f"  Regret %: {regret_result['regret_percentage']:.2f}%")
        print(f"  Exploration Cost: ${regret_result['exploration_cost']:,.2f}")
        
        print("\n" + "="*70)


def main():
    """Run the complete analysis."""
    
    # Define parameters
    true_returns = {
        'A': 0.8,
        'B': 0.7,
        'C': 0.5
    }
    total_budget = 10000.0
    ab_budget = 2000.0
    ab_split = {'A': 0.5, 'B': 0.5}  # Equal split between A and B
    
    # Initialize analysis
    analysis = MultiArmedBanditAnalysis(true_returns, total_budget)
    
    # Run A/B testing strategy
    ab_result = analysis.ab_testing_strategy(ab_budget, ab_split)
    
    # Run optimal strategy
    optimal_result = analysis.optimal_strategy()
    
    # Compute regret
    regret_result = analysis.compute_regret(ab_result, optimal_result)
    
    # Print summary
    analysis.print_summary(ab_result, optimal_result, regret_result)


if __name__ == "__main__":
    main()
