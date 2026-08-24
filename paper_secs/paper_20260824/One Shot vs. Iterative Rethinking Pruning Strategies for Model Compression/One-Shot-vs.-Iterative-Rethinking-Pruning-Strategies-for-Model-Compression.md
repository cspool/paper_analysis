# One Shot vs. Iterative: Rethinking Pruning Strategies for Model Compression

Mikołaj Janusz<sup>a,\*</sup>, Tomasz Wojnar<sup>a</sup>, Yawei Li<sup>b</sup>, Luca Benini<sup>b</sup> and Kamil Adamczewski<sup>c</sup>

<sup>a</sup>Jagiellonian University

<sup>b</sup>ETH Zürich

<sup>c</sup>Wrocław University of Science and Technology

Abstract. Pruning is a core technique for compressing neural networks to improve computational efficiency. This process is typically approached in two ways: one-shot pruning, which involves a single pass of training and pruning, and iterative pruning, where pruning is performed over multiple cycles for potentially finer network refinement. Although iterative pruning has historically seen broader adoption, this preference is often assumed rather than rigorously tested. Our study presents one of the first systematic and comprehensive comparisons of these methods, providing rigorous definitions, benchmarking both across structured and unstructured settings, and applying different pruning criteria and modalities. We find that each method has specific advantages; one-shot pruning proves more effective at lower pruning ratios, while iterative pruning performs better at higher ratios. Building on these findings, we advocate for patience-based pruning and introduce a hybrid approach that can outperform traditional methods in certain scenarios, providing valuable insights for practitioners selecting a pruning strategy tailored to their goals and constraints. Source code is available at https://github.com/janumiko/pruning-benchmark.

#### Introduction

As the complexity and scale of tasks for machine learning and computer vision continue to grow, so does the size of neural networks designed to address these tasks [10, 20, 21, 25, 28, 48]. Larger models typically achieve superior performance. However, deploying these large models can be computationally expensive and resource-intensive, making them impractical for environments with limited computational power, such as mobile devices or embedded systems [15, 17, 37]. Pruning is a critical technique that reduces network size without significantly compromising performance [5, 9, 22, 31, 40, 41, 45, 55]. By removing redundant parts of a network, pruning can help achieve results similar to those of fully trained networks, especially in inference scenarios where efficiency is essential.

Pruning techniques are broadly divided into two categories: **one-shot pruning** and **iterative pruning** (Figure 1). One-shot pruning consists of a single cycle of training, pruning, and retraining, as detailed in Section 3. In contrast, iterative pruning involves multiple cycles of these steps, leading to potentially more refined and efficient

network structures. Despite their prevalence, a systematic comparison of these two approaches under different regimes is limited in the literature. Prior research has largely focused on either the criteria for pruning [35, 55] or the specific pruning methods [6, 41, 53], leaving a gap in understanding the relative performance of different pruning strategies under varying conditions.

In this work, we conduct a thorough comparison of one-shot versus iterative pruning across multiple settings. Such a comparison requires isolating the pruning regime as a variable, which fits into the broader view of pruning as a multi-objective optimization problem trading off accuracy, model size, and computational cost. Our controlled experimental setup holds the pruning criterion and target sparsity fixed to achieve this isolation. Furthermore, to properly evaluate the retraining cost, we propose patience-based fine-tuning as a more natural way to gauge fine-tuning duration. Our analysis focuses primarily on vision tasks, where the pruning literature is most extensive. To broaden our investigation, we also present an exploratory case study on a text generation task to explore how these regimes translate to the NLP domain.

For iterative pruning, we introduce a geometric pruning ratio scheduler in addition to the constant pruning ratio scheduler. Unlike the constant scheduler, which prunes a fixed percentage of weights across the entire network, the geometric scheduler prunes a fixed percentage of the remaining weights at each step, progressively removing fewer weights as pruning progresses. Our experiments reveal that the geometric pruning scheduler generally outperforms the constant scheduler.

Additionally, inspired by the respective strengths of one-shot and iterative pruning, we propose a **hybrid few-shot pruning** regime, combining aspects of both methods. This hybrid approach shows advantages over the individual techniques in certain scenarios. Notably, this study does not seek to establish one method as universally superior; instead, it aims to provide guidelines to help practitioners choose the most suitable pruning approach based on specific requirements and constraints.

To summarize, the contributions of this work include:

- Defining the problem of pruning regime selection, and the first broad comparison of one-shot and iterative pruning methods.
- The introduction of a geometric pruning ratio scheduler that prunes a fixed percentage of remaining weights in each pruning step, demonstrating benefits for iterative pruning.
- Proposing a hybrid approach that combines aspects of one-shot and iterative pruning to provide flexibility in specific cases.

 $<sup>*</sup> Corresponding \ Author. \ Email: mikolaj1.janusz@student.uj.edu.pl$ 

 Broad empirical results that describe preferred settings for each pruning regime, in particular the impact of pruning regime on pruning criteria.

> **[图片提取文字 (无描述)]:**
> 100 80 Overall Pruning Level 60 40 20 One-Shot Iterative Geometric Iterative Constant - Hybrid Few-Shot 0 2 10 Iteration
![](_page_1_Figure_1.jpeg)

**Figure 1**: A conceptual illustration of the different pruning regimes. This schematic shows the progression of weight removal for one-shot, iterative constant, and iterative geometric pruning, and is not derived from a specific experiment.

#### 2 Pruning regimes in literature

Pruning is one of the earliest and most established techniques for compressing neural networks. This section attempts to show the extent of the bias in the pruning literature towards proposing new pruning criteria. This bias underscores the importance of the proposed problem, which is broadly orthogonal and shows an alternative direction of pruning research.

Early research demonstrated that training large networks and then removing weights can significantly improve computational efficiency and reduce parameter size [19, 29]. Much of the pruning literature focuses on developing new methods to define pruning criteria, with the most common approaches based on the magnitude of weights [15, 23], sensitivity to derivatives [29, 44], or empirical filter sensitivity [3, 36]. Other methods have been introduced that prune based on neuron similarity [49], activation functions [38], or even more unconventional criteria, such as game-theoretic [2], reinforcement learning [22], or genetic algorithms [18, 40].

Despite this diversity of pruning methods, there is no unified procedure for how networks are pruned. Instead, two main pruning strategies are prevalent in the literature: one-shot pruning and iterative pruning. These approaches differ in whether the network is pruned all at once or in successive stages. However, the impact of this choice on pruning outcomes is not well understood. Each approach also comes with a unique set of parameters that can significantly influence final network performance, such as fine-tuning duration and the number of pruning iterations. Details like the exact percentage of weights removed at each step are often not explicitly provided, leaving gaps in the understanding of how specific pruning configurations affect results. In this work, we argue that these choices are as crucial as pruning criteria and can greatly influence the overall performance of the pruned network. Our study aims to address this gap in the literature, conducting a detailed analysis to clarify how different pruning strategies affect network performance.

In the following sections, we discuss studies that employ both oneshot and iterative pruning, and in Table 1, we summarize common pruning methods and their associated training protocols. One-shot pruning. One-shot pruning involves a single cycle of pruning followed by retraining, making it a computationally efficient approach widely appreciated in the literature. In this method, node ranking is computed only once, resulting in the final network structure after a single pruning step.

One-shot pruning is often chosen for methods where pruning costs are high. For instance, He et al. [23] removes filters near the geometric median, arguing that these filters are effectively represented by the remaining ones. Calculating the geometric median, however, is computationally intensive. Similarly, Dirichlet pruning [1] requires a training phase to compute parameters of the Dirichlet distribution, which serve as importance weights. CURL [42], on the other hand, removes all unimportant filters across layers in one step using a KL-divergence-based criterion. Li et al. [34] conducted one-shot pruning according to the gradients of the latent vectors in a hypernetwork, which is introduced to guide network pruning.

After the pruning step, retraining is typically performed. However, the optimal length for retraining has not been thoroughly investigated. Instead, many methods fix the retraining duration arbitrarily; for example, CURL retrains the model for 100 epochs.

**Iterative pruning.** Iterative pruning refines neural network architectures by progressively removing less important parameters or structures while fine-tuning the network to maintain performance. Unlike one-shot pruning, which removes a significant portion of weights or neurons at once, iterative pruning involves multiple cycles of pruning and retraining.

Iterative pruning requires computing parameter importance at every step. Therefore, for large networks, iterative pruning can be computationally prohibitive in cases of unstructured pruning, due to the vast number of individual parameters. However, in smaller networks, iterative pruning dates back to early methods such as Optimal Brain Damage (OBD) and Optimal Brain Surgeon (OBS) [19], which remove unimportant connections based on the second-order derivatives of the loss function with respect to the weights. In second-derivative methods, [19] requires calculating the inverse Hessian matrix, which involves significant computational effort. Given the computational cost of calculating such pruning criteria, only one [52] or a few batches [36] are often used to evaluate parameter sensitivity.

In iterative pruning, a common approach is to use a constant pruning rate—a fixed percentage of weights pruned in each cycle. However, this rate varies across methods. For example, [45] uses Taylor expansions to approximate filter contributions, removing 10 neurons every 30 mini-batches until the target number of pruned neurons is reached, followed by fine-tuning for 25 epochs. In [47], 20% of the lowest-magnitude weights are pruned globally, after which the network is retrained using learning rate rewinding and the original training time. [35] explores sub-architecture optimization by randomly removing 10% of weights, followed by fine-tuning.

The number of elements pruned in each cycle may be determined by various criteria, often treated as hyperparameters. [16] sets a custom threshold, removing elements whose magnitude is below that threshold. [39] prunes channels at each iteration based on Fisher Information scores, allowing the number of pruned channels to vary. [36] uses a hyperharmonic sequence for pruning ratios, where the ith pruning ratio follows  $1 - \frac{1}{(i+1)^{\alpha}}$ . [38] prunes a specified number of channels layer-wise, fine-tuning after each layer's pruning. Additionally, [43] uses statistics from subsequent layers to prune each layer iteratively, fine-tuning for one or two epochs after each layer is pruned. Li et al. [32] applied group sparsity to the sparsity-inducing matrix and conducted a proximal gradient descent algorithm to progressively prune the network during the optimization of the pruning

procedure.

Pruning at initialization and during training. While this work primarily focuses on post-training pruning using a train-prune-retrain cycle, we acknowledge other types of pruning. Pruning at initialization [14, 30, 51] avoids pruning a fully trained model, instead identifying a smaller subnetwork to train from scratch for comparable performance to the larger model.

Another category involves pruning during training through regularization, which encourages certain parameters to approach zero [46, 54]. For example, [54] introduces scaling factors that selectively scale the outputs of certain CNN structures, applying sparse regularization to these scaling factors to progressively reduce their influence during training.

Unstructured and structured pruning. Pruning networks can be done for individual weights or for structures within the network. Each of them is important in its own regard and is broadly researched. Hence, this benchmark includes tests where both unstructured and structured pruning are considered. We build our benchmark around the Torch-Pruning [13] for structured pruning, and extend it with unstructured pruning using the built-in pruning utilities in PyTorch. The details for how unstructured and structured pruning are done can be found in the Appendix.

## 3 Pruning regimes

In the context of neural network optimization, pruning regimes can be split into categories based on the following question: *How should we divide the pruning process?* Should we prune all the redundant weights at once? Or should we divide the pruning between different iterations that are separated by modifying the structure of the network and weight updates? In this section, we present rigorous definitions of pruning criteria.

For the purpose of the comparison of the pruning regimes, assume that W is the total number of weights in the neural network and p is the desired pruning percentage, e.g. p = 0.8 means a pruning of 80% of weights.

- One-shot Pruning: One-shot pruning is a regime where a substantial portion of the network's weights are removed in a single pruning step, following the process:
  - Initial Training: The neural network is first fully trained to ensure it learns the patterns in the data.
  - Pruning Step: A certain percentage of the least important weights or neurons (based on metrics like magnitude or importance) are pruned in one go.
  - Fine-tuning: After pruning, the network may undergo additional fine-tuning to regain any performance lost from pruning.

The least significant weights are eliminated based on a predetermined importance criterion. The number of weights removed after one-shot pruning is given by:

$$p \times W$$
.

• Iterative Pruning: Iterative pruning is a process where weights are pruned over multiple iterations, allowing the network to gradually retrain and recover some performance loss. In each iteration the ranking, pruning, and fine-tuning occurs. We define two common approaches to iterative pruning and name them differently to avoid confusion.

– Iterative Constant: A constant number of parameters is pruned at each step. Let *steps* be the number of iterations, then

$$\frac{p \times W}{\text{steps}}$$

weights are pruned at each stage. The pruned percentage per step is fixed in relation to the initial number of weights.

– Iterative Geometric: A fixed percentage p of the remaining weights is pruned at each step, meaning that as the pruning process progresses, fewer weights are pruned at each iteration. The number of weights at step n is given by the following formula:

$$W \times (1 - \frac{p}{\text{steps}})^n$$

- Hybrid pruning: We propose a new pruning regime, hybrid (fewshot) pruning, which is a combination of the idea of one-shot and geometric regimes. The majority of the weights are removed at the first step and the model is retrained for a longer time. Then for the remaining weights, we perform a more fine-grained geometriclike pruning over several iterations. The hybrid pruning approach can be summarized as follows:
  - Apply large pruning ratio p<sup>k</sup> to the original network and perform a longer fine-tuning phase.
  - Perform iterative geometric pruning with smaller pruning ratio, p<sup>i</sup> ≪ p<sup>k</sup> starting from the state with p<sup>k</sup> parameters removed. Proceed until the desired final pruning percentage p.

We elaborate on the hybrid regime and empirical estimates for p<sup>k</sup> and p<sup>i</sup> in Section 5.5.

## 4 Pruning regime factors

Pruning regimes, whether one-shot or iterative, involve parameters that significantly impact their effectiveness. To properly evaluate these regimes, it is essential to consider these parameters.

## *4.1 Retraining*

Retraining, or fine-tuning, is a critical step in the pruning process, helping the neural network recover performance after a portion of its weights or channels has been removed. Retraining is necessary because the initial pruning can degrade the model's accuracy by eliminating parameters that the network previously relied upon for making predictions.

Both one-shot and iterative pruning require retraining, though the duration may vary. Generally, the larger the drop in accuracy, the longer the retraining phase should be. The accuracy drop typically correlates with the number and importance of pruned parameters, leading to longer fine-tuning for one-shot pruning and shorter, repeated fine-tuning phases for each step in iterative pruning.

However, the optimal length of retraining has not been rigorously studied, with methods often using arbitrary values. For instance, [47] fine-tunes the model for the full original training time, which may be computationally excessive since the network is not being trained from scratch. Conversely, in [7, 43], fine-tuning is limited to one epoch after each pruning step in iterative pruning.

| Method                                | structure    | regime                  | step |
|---------------------------------------|--------------|-------------------------|------|
| HRank [38]                            | structured   | iterative (custom)      |      |
| ThiNet [43]                           | structured   | iterative (custom)      |      |
| SSS [26]                              | structured   | iterative (unspecified) |      |
| Revisiting Random Pruning (RRCP) [35] | structured   | iterative (constant)    | 10%  |
| Fisher information [52]               | structured   | iterative               |      |
| Learning rate rewinding [47]          | both         | iterative (constant)    | 20%  |
| Optimal brain damage [29]             | unstructured | iterative               |      |
| Optimal brain surgeon [19]            | unstructured | iterative               |      |
| Learning weights and connections [16] | unstructured | iterative               |      |
| Taylor Expansion [45]                 | structured   | iterative (constant)    | 2%   |
| Group Fisher Information [32]         | structured   | iterative (custom)      |      |
| Empirical Sensitivity Analysis [36]   | structured   | iterative (custom)      |      |
| CURL (KL-divergence metric) [42]      | structured   | one-shot                |      |
| Dirichlet Pruning [1]                 | structured   | one-shot                |      |
| Geometric Median [23]                 | structured   | one-shot                |      |

Table 1: A list of selected pruning methods and their corresponding training regimes

Patience (early stopping). In this work, we advocate for using patience or early stopping to determine the number of fine-tuning epochs in both one-shot and iterative pruning. Patience allows a dynamic approach to fine-tuning, where the model is trained until a specified criterion (e.g., validation accuracy) no longer improves over a set number of epochs. The best-performing checkpoint is retained. Patience is beneficial because it adapts the fine-tuning duration based on actual performance, whereas fixed epochs may either be insufficient or wasteful. In this study, we use patience as an alternative to a fixed epoch count, defining it as the number of epochs to continue fine-tuning before stopping if no improvement occurs. The details of our patience-based algorithm are outlined in Appendix Algorithm 1. In Appendix Section A we present a comparative study to emphasize the importance of proper selection of both patience-based retraining length and step size.

## *4.2 Iteration Pruning Rate*

In iterative pruning, a key parameter is the iteration pruning rate, which determines the percentage of parameters removed at each step. Assuming a fixed target pruning rate, the iteration pruning rate is defined either by the number of steps or, conversely, the number of steps is determined based on the selected pruning rate.

As discussed in Section 1, the pruning rate varies widely across methods, from as low as 1% to as high as 20–30%. Reflecting on these values, we examine both single-digit and double-digit iterative pruning rates for constant and geometric pruning. In constant pruning, the iterative rate should evenly divide the final pruning rate. In geometric pruning, however, the amount pruned at each step decreases, and the rate is set so that the geometric sum of pruning rates across steps achieves a predefined final pruning level. Further details can be found in the Appendix.

