# B Pruning criteria further insights

In this work, we experiment with several pruning criteria, including magnitude pruning (Fig. 2), Hessian-based pruning [29] (Fig. 3bc; see Appendix for structured pruning adaptation), and Taylor expansion-based contribution approximation pruning [45] (see Appendix).

Our primary focus is on magnitude pruning due to its simplicity, effectiveness, reliability, and low computational cost, allowing for extensive benchmarking and experimentation [24]. Magnitude pruning is one of the most widely used pruning criteria across a variety of methods [15, 16, 47].

In this section, however, we want to address the potential similarities and differences in evaluating pruning regimes when alternative criteria are applied. Overall, we find that the relative performance of pruning regimes is largely consistent across different criteria. With the appropriate retraining duration, one-shot pruning performs best up to 80% of the original parameter count, while iterative pruning is preferable at higher compression ratios. Notably, for Hessian-based criteria, one-shot pruning at high pruning rates results in a significant accuracy drop, suggesting iterative pruning may be a more stable solution for second-derivative-based methods.

For second-derivative pruning, the Hessian matrix, which captures the curvature of the loss function, identifies weights in low-curvature regions (small eigenvalues) as good pruning candidates. The experimental results may be explained by the fact that single-step pruning can dramatically alter the loss landscape, rendering the pre-pruning Hessian less accurate in assessing remaining weights. In contrast, iterative pruning enables recalculating the Hessian at each step, ensuring a more precise sensitivity evaluation of the weights retained.

We then expand on the main text, which compares different pruning criteria under various training regimes. We present additional results on structured pruning using two criteria: Hessian-based pruning [29] and Taylor expansion-based contribution approximation pruning [45]. The results, shown in Fig. 8, are largely consistent with the conclusions drawn in the main paper. Specifically, one-shot pruning performs better or comparably up to about a 90% pruning rate, whereas iterative pruning yields better performance at higher compression ratios.

