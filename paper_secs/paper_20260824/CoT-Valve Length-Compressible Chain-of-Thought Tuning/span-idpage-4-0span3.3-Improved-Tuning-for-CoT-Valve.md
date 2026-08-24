# <span id="page-4-0"></span>3.3 Improved Tuning for CoT-Valve

In this section, we present two enhanced variants of CoT-Valve: one aimed at achieving improved controllability and the other focused on optimizing the compression ratio of the reasoning paths.

A More Precise CoT-Valve Paradigm: CoT-Valve++. In the previously proposed CoT-Valve framework, the training process only constrained ∆θ to satisfy the final objective with α = 1. However, during inference, we expect all positions along this direction to exhibit reasoning trajectories of varying lengths. This leads to the inconsistency between training and inference. With MixChain, we can explicitly incorporate this requirement during training by introducing an additional constraint, ensuring that the model can adapt to reasoning chains of different lengths across all positions in this direction. For each training sample, in addition to the question, answer, and solution, we have introduced a normalized term β, which represents the factor for the length of the reasoning path. Under this dataset, our training objective is modified to find a parameter update ∆θ ′ such that it satisfies:

$$\max_{\Delta \theta'} \mathbb{E}_{(q,a) \sim \mathcal{D}'} p\left(a \mid t_{< m}, q; \theta + \beta \Delta \theta'\right)$$

$$\prod_{i=1}^{m} p(t_i | t_{< i}, q; \theta + \beta \Delta \theta') \quad (3)$$

Where D′ is the Mixchain dataset. Each sample consists of the question q, the answer a, the solution {ti} m <sup>i</sup>=1 and β, where β is calculated as:

$$\beta = 1 - \frac{m - m_{min}}{m_{max} - m_{min}} \tag{4}$$

Here, mmin and mmax is the length of the shortest solution and longest solution for this question. Based on synthetic samples, we introduce additional constraints that enable us to better identify the updated parameter ∆θ ′ , facilitating more precise compressibility and controllability.

Progressive Chain Compression: CoT-Valve+P. The structure of MixChain, which features progressively shorter reasoning paths for each question,

facilitates a progressive chain-length compression strategy. This approach is similar to iterative pruning in model compression [\(Molchanov et al.,](#page-9-18) [2016\)](#page-9-18). In this process, the model is trained with a shorter reasoning path from the dataset at each iteration, rather than training directly with the shortest reasoning CoT. This gradual compression method allows the model to progressively reduce the length of its reasoning paths.

