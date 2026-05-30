# <span id="page-2-0"></span>3 Understanding the role of data order on recurrent models

In this section, we show that the quality of recurrent large language models varies as a function of the order in which data arises in context making them brittle for in-context learning applications.

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2: Data order vs. quality. The x-axis shows the recurrent state size in (bytes) during inference. The y-axis shows the accuracy on the set disjointness task, where the model needs to output the intersecting elements between two sets of tokens A and B (of lengths |A| and |B|) provided in-context. (Left) |A| is longer than |B|; (Middle) |B| is longer than |A|; (Right) Difference in accuracy between the two orderings. We evaluate non-causal and causal versions of the Based recurrent architecture from [\[7\]](#page-12-6). For each, we vary the hyperparameters (e.g., model dimension, feature dimension) that affect the state size. We plot the maximum score for each point across a sweep of three learning rates {1e − 4, 5e − 4, 8e − 4} and two random seeds. The plot shows that the causal recurrent models are more sensitive to the data order than non-causal models.

## 3.1 Analysis of data order and communication complexity

Set disjointness problem. To formalize the impact of data order, we invoke the set disjointness (SD) problem: given two sets of elements, determine if the intersection is empty or not. SD is the quintessential problem for studying the communication complexity of different streaming algorithms (such as recurrent models) over the past several decades [\[38\]](#page-14-6). The hardness for a wide collection of problems reduces to the hardness of SD [\[11\]](#page-12-10). A formal definition of this task is provided in Appendix [G.2.](#page-42-1)

Synthetic formulation. We construct a synthetic task where the model is given input sequences that contain two sets A and B, seperated by a special token that designates the end of set A and start of set B. Set elements are tokens ∈ [0..|V |] for vocabulary size |V | and the model needs to output the tokens in the intersection of A and B. For example, the correct output below would be 6:[1](#page-3-1)

$$\underbrace{7\ 11\ 17\ 16\ 4\ 6\ 9}_{\mathbf{Set}\ \mathbf{A}}\ *\ \underbrace{8\ 1\ 5\ 6}_{\mathbf{Set}\ \mathbf{B}}\to\ ?$$

In Figure [2,](#page-3-0) we vary the state size of the Based recurrent architecture [\[7\]](#page-12-6), which has been demonstrated to outperform prior subquadratic models on recall, on the SD task. We train on sequences where |A| and |B| are between 1 and 1024, and |V | = 2048. In addition to measuring overall accuracy, we consider the sliced accuracy on sequences where |A| < |B| and sequences where |B| < |A|.

We find the causal models achieve better quality when the size of set A is smaller than set B. Figure [2](#page-3-0) (Right) shows the difference in quality between when A is shorter vs. longer than B, reflecting that the gaps tend to be larger at smaller state sizes (x-axis). We additionally evaluate a non-causal variant of the Based architecture and find (1) it outperforms the causal models across state sizes when A is longer than B (Figure [2](#page-3-0) (Left)), and (2) displays less variation in quality as a function of data (set) order Figure [2](#page-3-0) (Right). We release code to reproduce this plot.

Theoretical study: recall and set disjointness. In Appendix [G,](#page-39-0) we perform a systematic theoretical study of the connection between set disjointness and recall as well as the complexity of solving set disjointness in the JRT setting.

<span id="page-3-1"></span><sup>1</sup>Note that we train the model to output the set intersection, of size 1, not binary disjointness result (Algorithm [1\)](#page-21-0). We find explicitly outputting the intersection helps the model avoid the behavior of outputting 0 or 1 with 50% accuracy during training.

First, we show that set disjointness and the "general associative recall" (GAR) problem, which we define in Appendix G [Definition G.24]), are essentially equivalent (see Propositions G.25 and G.26). Roughly speaking, the keys and queries in GAR correspond to sets A and B in set disjointness.

We argue that recurrent models need space  $\Omega(\min(|A|, |B|))$  for solving set disjointness, and hence, GAR (see Proposition G.29 in Appendix G.4.1).

**Proposition 3.1.** Given a JR-p prompt<sup>2</sup>  $u^{JR-p} \in \{0,1\}^{pN\times d}$  for input  $u \in \{0,1\}^{N\times d}$  to the GAR problem, any recurrent model  $\mathcal{M}_{GAR}$  (definition G.12) solving GAR requires its state size to be at least  $\Omega\left(\frac{\min\{|A|,|B|\}}{p}\right)$ -bits.

That is, the lower bound holds even if we allow multiple, but constant, many passes, as opposed to  $\Omega(\max(|A|, |B|))$  lower bound for recurrent models without repeats [7] **Theorem F.3**.

Next, we show we can indeed achieve this lower bound. We show that certain recurrent models (concretely, a slight variant of Based) can solve SD with  $O(\min(|A|, |B|))$  space in the JRT-PROMPT setting (App. G.3).

**Theorem 3.2.** Given a JRT prompt  $\mathbf{u}^{JRT} \in \mathbb{R}^{2N \times (n+1)}$  of the input  $\mathbf{u} \in \mathbb{R}^{N \times (n+1)}$  for the set-disjointness (SD) problem  $(A, B) \subseteq \{0, 1\}^n$ , there exists a Based model (BaseConv + MLP + LinearAttention + MLP)<sup>3</sup> that solves SD with space  $O(\min\{|A|, |B|\} \cdot n)$ .

Finally, we show that this improvement via JRT-prompting is not realizable for all possible architectures. In particular, we show that  $\Omega(\max\{|A|,|B|\}) = \Omega(N)$  lower bounds for the BaseConv model (a model that provably simulates any gated convolution, e.g. Hyena [39], H3 [40], with just poly-log blowup in parameters and depth) (Theorems F.4, F.5, and F.6, [7]) for recall carry over even in the JRT-prompt setting (see Theorems G.6, G.7, and G.11).

