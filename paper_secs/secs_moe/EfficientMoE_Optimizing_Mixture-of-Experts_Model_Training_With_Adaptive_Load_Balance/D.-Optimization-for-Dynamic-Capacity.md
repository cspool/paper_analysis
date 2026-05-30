# *D. Optimization for Dynamic Capacity*

Limited by the design of static graphs, all experts had the same expert capacity, and a large number of tokens were discarded because of the small expert capacity of hot experts, resulting in a reduction in training accuracy. By contrast, cold experts need to fill too many zero vectors because of their large capacity, which leads to a waste of hardware. Therefore, EfficientMoE constructs an expert capacity model to set appropriate capacity values for the various experts. This can prevent significant token discarding or padding and improve AI-accelerator resource utilization by making the cold and hot experts or replica experts on the same AI-accelerator fully share resources.

To ensure that experts could still process input tokens, the ratio of tokens processed by cold and hot experts was set higher than p%. EfficientMoE introduced a capacity adjustment method based on expert periodic loads. It defines the baseline capacity B and introduces a capacity factor F, which is defined by the experts' computational load, to adjust their capacity. For each cycle j, EfficientMoE calculates a cycle-specific expert capacity C*<sup>j</sup>* by analyzing the token loads and applying a weighted average across recent cycles to account for temporal load trends:

$$C_j = (1 - r) * B + r * \frac{1}{m} \sum_{i=1}^{m} (F_i)$$
 (3)

where B is constructed by statistically analyzing the token counts over a sampling cycle of m iterations for each expert. Extreme values were discarded, and the average of the m iterations was used as the token load for each expert. The token loads of all experts were then sorted, and the value that covered the top p% of the number of tokens was chosen as the baseline capacity B. Factor r is defined as follows:

$$r = \left(\max\left(\overline{T_i}\right) - B\right) * \beta \tag{4}$$

where T*<sup>i</sup>* represents the average number of tokens that must be processed by expert i in m iterations. β (0 ≤ β ≤ 1) is a decay constant used to adjust the size of r to achieve the optimal expert capacity setting for the load impact of different datasets. This term allows C*<sup>j</sup>* to dynamically adjust downwards for experts whose peak load has not persisted over time, freeing capacity for more consistently high-demand experts. The capacity factor F is calculated as follows:

$$F = \gamma * \frac{T_i - B}{Total\_tokens} \tag{5}$$

where T*<sup>i</sup>* represents the token count for experti as determined by the load evaluation model. For cold experts, T*<sup>i</sup>* − B is negative, resulting in a decrease in C*i*. Conversely, F is positive for hot experts, which leads to an increase in C*i*. The weighting coefficient γ (0 ≤ β ≤ 1) can realize the smooth regulation of F and reduce the impact of the abnormal load of a few experts on F. In addition, EfficientMoE must ensure that the total memory requirement M*cost* for all experts on the same AI accelerator is less than the total memory the of AI accelerators. The memory requirement for expert i is expressed as follows:

$$M_{cost,i} = M_p + M_a + M_t \tag{6}$$

where M*p*, M*a*, and M*<sup>t</sup>* represent the cost of memory for the parameters of expert i, activation, and tokens, respectively. The overall implementation can be described using Algorithm [2:](#page-6-0)

Algorithm[2](#page-6-0) dynamically adjusted the capacities of the experts in the MoE model based on the token loads they handled. By

<span id="page-6-0"></span>**Algorithm 2:**Dynamic Adjustment for Capability of Expert.

allocating more capacity to hot experts (those handling more tokens) and less capacity to cold experts, the computational load was balanced across devices. This approach reduces the issues related to token truncation and padding, thereby optimizing resource utilization. Additionally, the memory cost check ensures that the total memory usage remains within hardware limits, which promotes efficient and scalable training of large models.

