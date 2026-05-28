"""
Quick Reference: Statistical Testing Functions

=== WILCOXON SIGNED-RANK TEST ===
Purpose: Non-parametric paired test for metric differences
Formula: W = min(W+, W-) where W+/W- are sums of signed ranks
Use when: Comparing scores between two models per query
Function: wilcoxon_signed_rank_test(differences, alpha=0.05, alternative="two-sided")
Returns: {'n', 'W_plus', 'W_minus', 'W', 'expected_W', 'variance_W', 'z_score', 'p_value', 'is_significant', 'method'}

Quick test:
```python
import numpy as np
from src.evaluation.statistical_comparison import wilcoxon_signed_rank_test

diffs = np.array([0.05, -0.02, 0.08, 0.00, -0.01])
result = wilcoxon_signed_rank_test(diffs, alpha=0.05)
print(f"p-value: {result['p_value']:.4f}, Significant: {result['is_significant']}")
```

=== GOODMAN-KRUSKAL GAMMA ===
Purpose: Measure ordinal association between rank assignments
Formula: γ = (C - D) / (C + D)
Use when: Comparing difficulty rankings between models
Function: goodman_kruskal_gamma_test(ranks_A, ranks_B)
Returns: {'n_queries', 'n_pairs', 'C', 'D', 'gamma', 'interpretation'}

Quick test:
```python
from src.evaluation.statistical_comparison import goodman_kruskal_gamma_test
ranks_A = [1, 2, 3, 4]
ranks_B = [1, 2, 4, 3]
result = goodman_kruskal_gamma_test(ranks_A, ranks_B)
print(f"γ = {result['gamma']:.3f}: {result['interpretation']}")
```

=== KAPPA COEFFICIENT ===
Purpose: Measure agreement between multiple runs beyond chance
Formula: κ = (p_o - p_e) / (1 - p_e)
Use when: Evaluating stability across multiple generations
Function: kappa_coefficient(retrieval_sets_i, retrieval_sets_j, relevant_sets=None)
Returns: {'n_queries', 'kappa_mean', 'kappa_min', 'kappa_max', 'kappas'}

Quick test:
```python
from src.evaluation.statistical_comparison import kappa_coefficient
gen1 = [{'url1', 'url2', 'url3'}]
gen2 = [{'url1', 'url2', 'url3'}]
result = kappa_coefficient(gen1, gen2)
print(f"κ = {result['kappa_mean']:.3f}")
```

=== PAIRED PERMUTATION TEST ===
Purpose: Non-parametric sign-flip test for significance
Method: Resampling with random sign flips
Use when: Non-parametric alternative to Wilcoxon for verification
Function: paired_permutation_test(scores_A, scores_B, n_permutations=10000, seed=67, alternative="two-sided")
Returns: {'observed_diff', 'p_value', 'n_permutations', 'perm_stats', 'alternative', 'interpretation'}

Quick test:
```python
from src.evaluation.statistical_comparison import paired_permutation_test
A = [0.8, 0.75, 0.85, 0.70]
B = [0.70, 0.65, 0.75, 0.60]
result = paired_permutation_test(A, B, seed=42)
print(f"p-value: {result['p_value']:.4f}")
```

=== HELPER FUNCTIONS ===

_rank_with_ties(values):
    Rank array values with tie-averaging per thesis definition
    Returns: ranks array where tied values have average rank

wilcoxon_signed_rank_test() / goodman_kruskal_gamma_test() / kappa_coefficient() / paired_permutation_test()
    Main statistical test implementations

=== DATASET PREPARATION ===

For Wilcoxon test:
    Input: differences array (scores_A - scores_B) per query

For Gamma test:
    Input: ranks of correct answer position (1-5 or ∞)
    
For Kappa:
    Input: sets of retrieved URLs per query, optionally relevant URLs

For Permutation test:
    Input: scores_A and scores_B arrays

=== INTERPRETATION GUIDE ===

p-value < 0.05:
    Difference is statistically significant at α=0.05 level

γ close to 1:
    Models strongly agree on query ranking

γ close to 0:
    No association between model rankings

κ close to 1:
    High agreement, method is stable

κ close to 0:
    Low agreement, method may be unstable

=== FILE LOCATIONS ===

Implementation: src/evaluation/statistical_comparison.py
Tests: src/evaluation/tests/test_statistical_comparison.py
Full docs: src/evaluation/STATISTICAL_TESTING.md

=== RUNNING TESTS ===

All tests:
    pytest src/evaluation/tests/test_statistical_comparison.py -v -o addopts=""

Specific test:
    pytest src/evaluation/tests/test_statistical_comparison.py::TestWilcoxonSignedRankTest -v -o addopts=""

With coverage:
    pytest src/evaluation/tests/test_statistical_comparison.py -o addopts="" --cov=src/evaluation

=== THESIS ALIGNMENT ===

✓ Wilcoxon: Exact tie-averaging implementation per Section 5.1
✓ Gamma: Concordant/discordant pair calculation per Section 5.2  
✓ Kappa: Contingency table with observed/expected agreement per Section 5.3
✓ Permutation: Sign-flip resampling per Section 5.4

All implementations match mathematical definitions from thesis exactly.
Test cases verify each formula component independently.
"""
