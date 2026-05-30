# <span id="page-21-0"></span>H Sensitivity to random seed

The experiments we report throughout Section [5](#page-7-0) use one fixed random seed (the default value from the supplementary code). To verify that our results are robust to randomness, we run SpQR with 5 random seeds (0-5) and measure the adjusted standard deviation.

For this evaluation, we compress LLaMA-65B with SpQR using b<sup>w</sup> = b<sup>z</sup> = b<sup>s</sup> = 3 and β<sup>1</sup> = β<sup>2</sup> = 16, which corresponds to 3.625 bits per parameter. The resulting perplexity scores are 3.75 ± 0.003 (WikiText2), 7.03 ± 0.01 (Penn Treebank) and 5.75 ± 0.00086 (C4). In addition to the chosen random seed, these standard deviations can be affected by the inherent nondeterminism of GPU computation. Overall, the standard deviations are at least one order of magnitude smaller than the difference between SpQR, GPTQ, and RTN.

