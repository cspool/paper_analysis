# 1 INTRODUCTION

Exploiting sparsity in the weights, activations and input data of deep neural networks (DNNs) is an effective technique for reducing the amount of computation that is needed to achieve a given model quality [\(Han et al.,](#page-11-0) [2015;](#page-11-0) [Gale et al.,](#page-11-0) [2019\)](#page-11-0). The past decade has seen significant progress in algorithms and high-performance software to make sparsity practically useful [\(Gray et al.,](#page-11-0) [2017;](#page-11-0) [Narang et al.,](#page-12-0) [2017;](#page-12-0) [Kalchbrenner et al.,](#page-12-0) [2018;](#page-12-0) [Elsen et al.,](#page-11-0) [2020;](#page-11-0) [Gale](#page-11-0) [et al.,](#page-11-0) [2020\)](#page-11-0). One area that remains a challenge for sparsity is efficient model training on accelerators. DNNs are most commonly trained on hardware accelerators like GPUs [\(NVIDIA,](#page-12-0) [2020\)](#page-12-0) and TPUs [\(Jouppi et al.,](#page-11-0) [2017\)](#page-11-0), which exploit the regularity of dense computation to deliver high performance. Consequently, fine-grained sparse computation is less efficient on these processors. To enable efficient computation on accelerators, structure can be enforced on the sparse matrices [\(Narang et al.,](#page-12-0) [2017;](#page-12-0) [Gray et al.,](#page-11-0) [2017;](#page-11-0) [Yao et al.,](#page-13-0) [2019\)](#page-13-0).

An emerging class of models with underlying structured sparsity is Mixture-of-Experts (MoEs) [\(Shazeer et al.,](#page-13-0) [2017\)](#page-13-0). Each layer in an MoE is a collection of *experts*, which are themselves small DNNs. As data is passed through the MoE layers, each token is dynamically routed to a subset of the experts for computation. By exploiting this sparse computation, MoEs have reduced training times by as much

*Proceedings of the* 6 th *MLSys Conference*, Miami Beach, FL, USA, 2023. Copyright 2023 by the author(s).

as 4× for applications in natural language processing and computer vision [\(Artetxe et al.,](#page-10-0) [2021;](#page-10-0) [Riquelme et al.,](#page-12-0) [2021\)](#page-12-0). These gains have translated to new levels of scale for model training [\(Artetxe et al.,](#page-10-0) [2021;](#page-10-0) [Du et al.,](#page-11-0) [2021;](#page-11-0) [Fedus et al.,](#page-11-0) [2022\)](#page-11-0).

The challenge in computing MoEs efficiently is handling the dynamic routing and load-imbalanced<sup>1</sup> computation that are fundamental to these architectures. However, existing hardware and software for deep learning make it difficult to meet this challenge. For example, TPUs and their XLA compiler require all tensor shapes to be known statically and often struggle with fine-grained operations like scatters and gathers [\(Fedus et al.,](#page-11-0) [2022\)](#page-11-0). These constraints make it difficult to implement MoEs directly on TPUs. While GPUs are more flexible, the sparse computation in MoEs does not map cleanly to the software primitives supported in major frameworks and libraries.

State-of-the-art frameworks for MoE training sidestep these challenges by placing rigid constraints on MoE routing. In order to remove the load imbalance from the computation, the set of tokens mapped to each expert are trimmed or padded to a user-specified size [\(Lepikhin et al.,](#page-12-0) [2020;](#page-12-0) [Fe](#page-11-0)[dus et al.,](#page-11-0) [2022;](#page-11-0) [Hwang et al.,](#page-11-0) [2022\)](#page-11-0). This procrustean formulation introduces a tradeoff between model quality and hardware efficiency, as users must decide whether to drop tokens or waste computation and memory on padding. This decision is often made through hyperparameter tuning, which increases the complexity of using MoEs.

To address these challenges, we develop an approach for

<sup>1</sup> Stanford University, Stanford, California, USA <sup>2</sup>Microsoft Research, Redmond, Washington, USA <sup>3</sup>Google Research, Mountain View, California, USA. Correspondence to: Trevor Gale <tgale@cs.stanford.edu>.

<sup>1</sup>Load imbalance results from different numbers of tokens being routed to different experts. We discuss this in detail in [§2](#page-1-0) and [§3.](#page-2-0)

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 1. A Mixture-of-Experts Layer. Shown for *num\_experts=3*, *top\_k=1* and *capacity\_factor=1* with the prevalent, token dropping formulation. First (1), tokens are mapped to experts by the router. Along with expert assignments, the router produces probabilities that reflect the confidence of the assignments. Second (2), the feature vectors are permuted to group tokens by expert assignment. If the number of tokens assigned to an expert exceeds its capacity, extra tokens are dropped. Third (3), the expert layers are computed for the set of tokens they were assigned as well as any padding needed for unused capacity. Last (4), the results of the expert computation are un-permuted and weighted by the router probabilities. The outputs for dropped tokens are shown here set to zero.

MoE routing and computation *based on sparse primitives*. Our approach never drops tokens and maps efficiently to modern GPUs, enabling end-to-end training speedups of up to 40% and 2.4× over state-of-the-art frameworks for MoE and DNN training, respectively. We make the following specific contributions:

- We show how the computation in an MoE layer can be expressed as block-sparse operations to accommodate imbalanced assignment of tokens to experts. We use this formulation to train *dropless-MoEs* (dMoEs).
- We develop high-performance GPU kernels for blocksparse matrix products that efficiently handle dynamic MoE computation. Our kernels use two techniques, *blocked-CSR-COO* encoding and *transpose indices*, to enable efficient matrix products with sparse inputs and outputs in transposed or non-transposed order.

We have implemented these techniques in a system called MegaBlocks, which builds on the state-of-the-art Megatron-LM library for training Transformer models [\(Shoeybi](#page-13-0) [et al.,](#page-13-0) [2019\)](#page-13-0). We evaluate our system through both microbenchmarks and end-to-end training of Transformer language models. Our code is open source and available at [github.com/stanford-futuredata/megablocks.](https://github.com/stanford-futuredata/megablocks)

## 2 BACKGROUND: MOE LAYERS

MoE layers are made up of many *experts*, which are themselves small neural networks. Each token<sup>2</sup> is dynamically routed to a subset of the experts for computation based on scores computed by a *router*. The experts are commonly

small multi-layer perceptrons (MLPs). It is typical for tokens to be sent to a small number of experts, often between 1 and 4 [\(Fedus et al.,](#page-11-0) [2022\)](#page-11-0).

MoE layers are often interleaved with other DNN layers. In Transformer models, MoE layers are most commonly used to replace feed-forward network (FFN) layers<sup>3</sup> [\(Shazeer](#page-13-0) [et al.,](#page-13-0) [2017;](#page-13-0) [Fedus et al.,](#page-11-0) [2022\)](#page-11-0). This hybrid architecture has demonstrated strong results on both natural language and vision tasks [\(Du et al.,](#page-11-0) [2021;](#page-11-0) [Riquelme et al.,](#page-12-0) [2021\)](#page-12-0). It is conjectured that these improvements are a result of experts specializing to different parts of the data distribution [\(Shazeer et al.,](#page-13-0) [2017\)](#page-13-0).

We illustrate an MoE layer in Figure 1 and describe it in detail in the remainder of this section.

## 2.1 Routing

The first stage of an MoE layer is the router, which is responsible for determining the assignment of tokens to experts. In addition to expert assignments, MoE routers also produce probabilities for each assignment that reflect the confidence of the mapping. These are encoded as a matrix of scores for each token-expert pair, which are used to linearly combine the *top\_k* expert outputs for each token (see [§2.4\)](#page-2-0).

The most common MoE router is the learned router proposed by [Shazeer et al.](#page-13-0) [\(2017\)](#page-13-0). In this router, the tokens are projected from *hidden\_size* elements to *num\_experts* scores by multiplying with a weight matrix that is learned jointly with the other model parameters. The scores are normalized with a softmax and the routing decisions are made by greedily selecting the *top\_k* scoring experts for each token.

For natural language, training data is composed of *tokens*. For vision, the data is typically *pixels* or *patches* [\(Dosovitskiy et al.,](#page-11-0) [2021\)](#page-11-0). For simplicity, we use the term token throughput this paper.

<sup>3</sup>The attention layers are left unchanged.

<span id="page-2-0"></span>![](_page_2_Figure_1.jpeg)

Figure 2. MoEs Trained on The Pile with Different Capacity Factors. The loss reached by the MoE models decreases significantly as expert capacity is increased, but at the cost of additional computation. The lowest loss is achieved by the "max" capacity factor model, which avoids dropping tokens through the dynamic capacity factor mechanism proposed by Hwang et al. (2022).

#### 2.2 Permutation

State-of-the-art MoE implementations compute all expert layers in parallel in order to make effective use of the parallelism available on GPUs and TPUs (Lepikhin et al., 2020; Fedus et al., 2022; Hwang et al., 2022)<sup>4</sup>. The standard primitive used by implementations is batched matrix multiplication, which computes a set of matrix products of the same shape (Figure 3A). However, mapping MoE computation to this primitive is non-trivial. In order to respect the shape constraints of batched matrix multiplication, the experts must have weight matrices of the same shape and the number of tokens assigned to each expert must be equal. The latter constraint is particularly problematic because the learned routing algorithm described above provides no guarantees of a load balanced assignment of tokens to experts.

In order to satisfy this constraint, prior work has defined a fixed expert capacity, which is the number of tokens that each expert can be assigned (Lepikhin et al. (2020); Fedus et al. (2022)). If the number of tokens assigned to an expert exceeds its capacity, the extra tokens are dropped. That is to say, they are not passed to any expert for computation and the model relies on a residual connection to reintroduce the dropped tokens' representations after the MoE layer. If an expert layer is not assigned enough tokens to fill its capacity, its set of tokens is padded to fill the remaining space. Expert capacity is typically specified in terms of a *capacity\_factor* hyperparameter, which is a multiplier on the expected number of tokens that would be assigned to each expert under a perfect uniform distribution:

$$expert\_capacity = \frac{num\_tokens}{num\_experts} \times capacity\_factor$$

Table 1. **Transformer Model Configurations.** These models are based on those used by Vaswani et al. (2017) and Brown et al. (2020). FLOPs were calculated using the expression from Narayanan et al. (2021b) with a single sequence of 1024 tokens. All models use *ffn hidden size=4×hidden size*.

| Transformer | hidden_size | num_layers | Weights (M) | <b>GFLOPs</b> |
|-------------|-------------|------------|-------------|---------------|
| XS          | 512         | 6          | 46          | 316           |
| Small       | 768         | 12         | 125         | 879           |
| Medium      | 1024        | 24         | 356         | 2487          |
| Large       | 1536        | 24         | 760         | 5122          |
| XL          | 2048        | 24         | 1316        | 8684          |

The *capacity\_factor* can be thought of as a parameter that reduces the chance of dropping a token. This hyperparameter represents a tradeoff between additional computation and model quality. As such, it is desirable to minimize the amount of load imbalance in the assignment of tokens to experts (§3). The typical mechanism for doing so is auxiliary *load balancing losses*, which incentivize the router to produce a balanced assignment (Shazeer et al., 2017; Lepikhin et al., 2020; Fedus et al., 2022). These losses additionally help to ensure that all experts see a similar number of tokens during training. This is thought to be important to avoid degenerate states where some experts are assigned zero tokens and stop receiving gradient updates (Zhou et al., 2022).

In addition to enabling batched computation of the expert layers, these constraints allow all tensor shapes to be known statically, which is required by TPUs and XLA.

### 2.3 Computation

Once the data has been permuted, the experts can be computed in parallel. For models where the experts are MLPs, this entails computing each layer for all experts using batched matrix multiplication. For convolutional experts, the layers can be computed with grouped convolutions.

#### 2.4 Un-permutation

After the experts are computed, the resulting feature vectors are un-permuted such that their ordering matches that of the input to the layer. The last step in MoE computation is to scale the output tokens by the scores with which they were assigned to their respective experts. When tokens are routed to more than one expert, these weighted results are summed to produce the final layer output for each token.

