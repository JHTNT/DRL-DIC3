"""
Master Runner: Complete Multi-Armed Bandit Analysis
====================================================

This script runs the complete analysis pipeline covering all tasks:
1. ✓ A/B Test phase analysis
2. ✓ Bandit selection from test results  
3. ✓ Exploitation phase with remaining budget
4. ✓ Comparison with optimal strategy
5. ✓ Regret calculation and analysis
6. ✓ Algorithm comparison and chart visualization
   (A/B Testing, Optimistic Initial Values, epsilon-Greedy,
    Softmax, UCB, Thompson Sampling)

QUICK START:
    python mab_master.py

This runs all analyses and generates a comprehensive report.
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_analysis():
    """Run complete analysis pipeline."""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "MULTI-ARMED BANDIT ANALYSIS: A/B TESTING VS OPTIMAL STRATEGIES".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\n")
    print("RUNNING ANALYSIS PIPELINE...")
    print("-" * 80)
    
    # Analysis 1: A/B Testing Strategy with Analytical Estimates
    print("\n[1/3] Loading A/B Testing Analysis Module...")
    try:
        from mab_ab_testing_analysis import main as ab_main
        ab_main()
    except Exception as e:
        print(f"❌ Error in A/B Testing analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Analysis 2: Bandit Algorithms Comparison
    print("\n" + "="*80)
    print("\n[2/3] Loading Bandit Algorithms Comparison Module...")
    try:
        from mab_algorithms_comparison import main as comparison_main
        comparison_main()
    except Exception as e:
        print(f"❌ Error in algorithms comparison: {e}")
        import traceback
        traceback.print_exc()
    
    # Analysis 3: Mathematical Foundations and Regret Theory
    print("\n" + "="*80)
    print("\n[3/3] Loading Mathematical Foundations Module...")
    try:
        from mab_regret_theory import main as theory_main
        theory_main()
    except Exception as e:
        print(f"❌ Error in regret theory: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print("""
FILES CREATED:
    1. mab_ab_testing_analysis.py  - Tasks 1-5: A/B testing analysis
    2. mab_algorithms_comparison.py - Task 6 (Part 1): Algorithm comparison  
    3. mab_regret_theory.py        - Task 6 (Part 2): Theoretical analysis

EACH FILE CONTAINS:
    ✓ Step-by-step calculations
    ✓ Clear assumptions documented
    ✓ Mathematical formulas
    ✓ Comparison tables
    ✓ Detailed code comments
    ✓ Runnable independently

TO VIEW DETAILED ANALYSIS:
    $ python mab_ab_testing_analysis.py         # Tasks 1-5
    $ python mab_algorithms_comparison.py       # Task 6 algorithms
    $ python mab_regret_theory.py               # Mathematical foundations

KEY TAKEAWAYS:
    1. A/B Testing Reward: $7,600
    2. Optimal Reward: $8,000
    3. Regret: $400 (5% loss)
    4. Adaptive algorithms (UCB, TS) achieve logarithmic O(log T) regret
    5. Fixed A/B testing achieves linear O(T) regret in gap-dependent analysis
    6. Thompson Sampling often has best practical performance
    7. Strategy comparison charts are saved in ./charts/

THEORETICAL RESULT:
    R_UCB(T) ≤ (5 + π²/3) · Σ_i (ln T) / Δ_i  [O(log T)]
    R_AB(T) ≤ Σ_i Δ_i · T / k                  [O(T)]
    
    The gap between strategies grows significantly with budget T.
    """)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    run_analysis()
