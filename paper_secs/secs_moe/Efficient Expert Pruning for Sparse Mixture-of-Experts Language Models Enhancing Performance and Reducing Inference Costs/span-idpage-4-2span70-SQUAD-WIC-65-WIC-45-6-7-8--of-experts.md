# <span id="page-4-2"></span>70 SQUAD WIC 65 WIC 45 6 7 8 # of experts

Figure 3: Performance from a single expert to an ensemble of experts.

#### <span id="page-4-0"></span>4.1 Motivation

LLMs based on the SMoE architecture have shown remarkable performance across various natural language

processing tasks [20, 50, 57]. These models leverage multiple experts, activating only a subset for any given input, thus balancing computational efficiency and model capacity. Typically, top-2 experts are activated, striking a balance between performance and computational cost.

Fig. 3 presents our investigation into the activation of different numbers of experts on Mixtral  $8 \times 7B$ -Instruct, revealing the following observations: i) Activating only a single expert does not lead to model collapse and may result in only a minimal performance drop compared to the default setting of using two experts. This suggests that individual experts possess redundant knowledge, enabling them to maintain reasonable performance independently. This redundancy indicates potential for expert pruning. ii) Conversely, activating all 8 experts leads to a noticeable performance gain, highlighting the benefits of expert ensemble. However, the computational cost of such an ensemble is substantially higher. Wortsman et al. [56] have shown that merging differently fine-tuned models can efficiently substitute their ensemble, achieving similar performance with reduced computational overhead.

Building on these insights, we propose a two-step approach involving expert pruning followed by expert merging. Initially, we search for the optimal subset of experts given a fixed size. Subsequently, we employ expert merging to consolidate the knowledge from the pruned experts into the remaining ones. This approach not only restores the knowledge of the pruned experts but also updates the surviving experts to incorporate the collective expertise of the entire SMoE block.

#### <span id="page-4-1"></span>4.2 Parameter space for expert pruning and merging

**Expert Pruning and Merging Matrices.** To efficiently prune and merge experts in each SMoE block  $(l=1\dots L)$ , we introduce two key matrices: the Router Mapping matrix  $(\boldsymbol{W}_{\rm RM}^l)$  and the Expert Merging matrix  $(\boldsymbol{W}_{\rm EM}^l)$ . For clarity, we omit the block index l in this section. A schematic illustration is provided in Fig. 1b. The router mapping matrix  $\boldsymbol{W}_{\rm RM} \in \mathbb{R}^{E' \times E}$ , where E' is the reduced number of experts (i.e., E > E'), is applied to the routing weights  $\boldsymbol{G}$  to reduce the dimensionality and handle fewer experts:

$$G' = W_{\text{RM}} \operatorname{softmax}(ZW_G),$$
 (5)

The expert merging matrix  $W_{\rm EM} \in \mathbb{R}^{E' \times E}$  is applied to the expert weights  $\{\boldsymbol{\theta}_i\}_{i=1}^E$  to merge E experts into E' experts. Each element in  $W_{\rm EM}$  operates blockwise on the parameters of the experts. Denote  $\{\omega_{j1}, \omega_{j2}, \ldots, \omega_{jE}\}$  as the j-th row of  $W_{\rm EM}$  that maps the original E experts to the j-th new expert  $\boldsymbol{\theta}_j'$ . We define merging as follows:

$$\boldsymbol{\theta}_{j}' = \{ \sum_{i=1}^{E} \omega_{ji} \boldsymbol{W}_{1i}, \sum_{i=1}^{E} \omega_{ji} \boldsymbol{W}_{2i}, \sum_{i=1}^{E} \omega_{ji} \boldsymbol{W}_{3i} \},$$
(6)

where the parameters of the experts are defined in Eq. (4).

Expert Pruning Phase. During the expert pruning phase, the low-rank matrices WRM and WEM are initialized with each row as a one-hot vector to ensure that only pruning occurs. Additionally, WRM and WEM are set as to be identical WRM = WEM. Consequently, these matrices only retain the selected expert weights and their corresponding routing weights. During evolutionary search, EEP also maintains the one-hot format of WRM and WEM.

Expert Merging Phase. In the expert merging phase, WRM and WEM are decoupled and initialized from their optimal values obtained during the pruning phase. This decoupling allows for a more flexible transformation where multiple experts can be merged, and the router weights can be updated independently. During this phase, the elements of WRM and WEM transition from discrete 0/1 values to continuous values. This allows the matrices to perform more nuanced transformations.

#### <span id="page-5-0"></span>4.3 Evolutionary search for the router mapping and expert merging matrices

The search space of the router mapping and expert merging matrices is large and complex, making it difficult to design heuristics for determining a solution, as is done in other expert pruning studies [\[37,](#page-13-8) [8,](#page-10-4) [34\]](#page-12-4). Therefore, an efficient optimization strategy is necessary. Given the substantial size of SMoE LLMs, computing gradients for optimization is computationally prohibitive for most users. As a solution, we employ a gradient-free evolutionary strategy, similar to approaches found in previous works [\[30,](#page-12-11) [32\]](#page-12-8). Our algorithm is detailed in Alg. [1.](#page-16-1)

Initially, we populate the search space using random initialization. During the evolutionary search, each set of router mapping and expert merging matrices is treated as an individual. In each iteration, only the top-performing individuals are selected as parents to produce the next generation through crossover and mutation. Specifically, during crossover, we randomly combine the entries of the matrices from two parents or select one parent's matrices entirely. For mutation, we introduce random Gaussian noise to the matrices, ensuring stochastic variations. This process conserves beneficial adaptations while discarding detrimental modifications, enhancing the optimization process.

This evolutionary reproduction process is repeated for a predetermined number of iterations within each search phase, updating the population with newly generated individuals. Upon completion of the search process, the best individual is selected as the output of our search algorithm.

