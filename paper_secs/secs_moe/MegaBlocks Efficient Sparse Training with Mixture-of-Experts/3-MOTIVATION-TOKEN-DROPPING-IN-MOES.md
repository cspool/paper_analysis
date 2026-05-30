# 3 MOTIVATION: TOKEN DROPPING IN MOES

Despite the use of load balancing losses, prior work has shown that token routing is still highly imbalanced (Hwang et al., 2022). To quantify the effect of token dropping on model quality, we trained MoE language models on The

<sup>&</sup>lt;sup>4</sup>We benchmark a sequential implementation in Appendix A.

<span id="page-3-0"></span>![](_page_3_Figure_1.jpeg)

Figure 3. Expert Computation in an MoE Layer. Shown with *num\_expert=3*. (A) State-of-the-art MoE implementations use batched matrix multiplication to compute all experts within a layer in parallel. This introduces the constraints that all experts are assigned the same number of tokens and that all experts have the same shape. (B) Expert computation can be analogously posed in terms of block diagonal matrix multiplication with identically sized blocks. (C) In order to relax these constraints, we can construct a block diagonal matrix with variable sized blocks made up of many smaller blocks. We can compute this matrix efficiently using block-sparse matrix multiplication.

Pile [\(Gao et al.,](#page-11-0) [2020\)](#page-11-0) with a range of capacity factors. We train Transformer MoEs similar to those used by [Fedus et al.](#page-11-0) [\(2022\)](#page-11-0), where each model is a Transformer with the FFN layers replaced with 64-expert MoE layers where each expert is a 2-layer MLP matching the original FFN dimensions. We used top-1 routing and based our MoE model dimensions on the Transformer-Small model described in Table [1.](#page-2-0) All models were trained using the tokenization from GPT2 [\(Radford et al.,](#page-12-0) [2019\)](#page-12-0) for 10B tokens with sequence length 1024, the Adam optimizer, and the learning rate and gradient clipping settings from [Shoeybi et al.](#page-13-0) [\(2019\)](#page-13-0). We trained all models on a single A100 GPU with a batch size of 512 sequences. We trained MoEs with capacity factor 1, 1.5, and 2 as well as the dynamic capacity factor technique proposed by Tutel [\(Hwang et al.,](#page-11-0) [2022\)](#page-11-0), where the capacity factor is set dynamically to the minimum value that would avoid token dropping. As a baseline, we trained standard Transformer models across a range of sizes. All Transformer and MoE models have vocabulary size 51200, sequence length 1024 and an attention head size of 64. Our model configurations are summarized in Table [1](#page-2-0) and the results of the experiments are shown in Figure [2.](#page-2-0)

For these models, we observed that the impact of token dropping is significant. While the MoE with capacity factor of 1 achieved a 0.15 reduction in validation loss compared to Transformer-Small, the MoE that avoided dropping tokens provided a reduction of 0.26, 1.73× larger than the gain of the former model and enough to exceed the quality of Transformer-Medium.

While dropping tokens reduces model quality, increasing capacity factor comes at the cost of additional computation and memory. In this example, MoE-layer math operations increased by over 2× in order to avoid dropping tokens. [Hwang et al.](#page-11-0) [\(2022\)](#page-11-0) showed that some MoEs require capacity factors as high as 11 in order to avoid dropping tokens,

and the necessary capacity factor to avoid dropping tokens can spike unpredictably during training.

In addition to the computational overhead of increasing the capacity factor, having to tune an additional hyperparameter can significantly increase the number of models that need to be trained for a target task. This is particularly cumbersome for large neural networks, where the cost to train a single model can run into the hundreds of thousands of dollars [\(MosaicML,](#page-12-0) [2022\)](#page-12-0). Possibly as a result of this, some large studies on MoEs have declined to explore different capacity factors at all [\(Artetxe et al.,](#page-10-0) [2021;](#page-10-0) [Clark et al.,](#page-11-0) [2022\)](#page-11-0).

