# <span id="page-36-2"></span>**D** Additional Experimental Results

#### <span id="page-36-0"></span>D.1 Dataset details

We evaluate on a diverse set of benchmark consisting of 7 popular fine-tuning datasets. Specifically, ARC-c and ARC-e [Clark et al., 2018] evaluate logical reasoning and world knowledge through challenging multiple-choice questions. Commonsense QA [Talmor et al., 2018] assesses a model's grasp of everyday knowledge and implicit relationships. OpenBook QA [Mihaylov et al., 2018] requires multi-step reasoning over scientific facts, while Winogrande [Sakaguchi et al., 2021] measures commonsense pronoun resolution. Accuracy is used as the evaluation metric for all above datasets. In addition, we evaluate the models on 2 more challenging datasets. GSM8K [Cobbe et al., 2021] contains 8.5k high-quality linguistically diverse grade school math word problems. Deriving the correct solution requires multi-step reasoning (2 to 8 steps) by the LLM model. CodeAlpaca [Chaudhary, 2023] contains 20k instruction-following data for fine-tuning LLM's code generation capability. HumanEval [Chen et al., 2021] consists of 164 hand-written programming problems, to access the LLM's capabilities in language comprehension, reasoning, algorithms, and simple mathematics. We train the LLM on CodeAlpaca and then evaluate the checkpoint on HumanEval. We measure the "Pass@1" metric, where we let the fine-tuned model to generate k=1 solution for each problem, and evaluate whether it can pass the unit tests.

#### <span id="page-36-1"></span>D.2 More details on experimental setup

For all models, we insert the adapters to the feed forward networks (FFN) of all transformer layers of the base models. Specifically, each FFN consists of an "up-projection" matrix, a "gate-projection" matrix and a "down-projection" matrix. We insert the adapter to each of the three matrices.

To ensure a fair comparison, we set an equal budget for trainable adapter parameters and compare different model architecture within this constraint. For LoRA [Hu et al., 2021], we vary the rank r in  $\{2^k \mid 0 \leq k \leq 10\}$ , and set the lora\_alpha parameter as  $2 \cdot r$  following standard practice. For MixLoRA [Li et al., 2024a], we adjust the number of experts within  $\{4,8\}$ , keep the number of active experts within  $\{1,2,4\}^{10}$  (while ensuring that it does not exceed half of the total experts), and the expert dimension within  $\{2^k \mid 0 \leq k \leq 6\}$ . For HydraLoRA [Tian et al., 2024], we vary the number of heads in  $\{4,8\}$ , and the rank r in  $\{2^k \mid 0 \leq k \leq 8\}$ . For S'MoRE, in most experiments (except the "scaling-up" study in Table 5), we limit S'MoRE to two layers due to resource constraints. We vary the number of experts  $(s_0,s_1)$  within  $\{(2,2),(4,4)\}$ : the fanout  $(f_0,f_1)$  is (1,1) when  $(s_0,s_1)=(2,2)$  and is (2,2) when  $(s_0,s_1)=(4,4)^{11}$ . We vary the expert dimension  $(r_0,r_1)$  within  $\{(2^k,2^k)\mid 0 \leq k \leq 6\}\cup\{(2^k,2^{k+1})\mid 0 \leq k \leq 5\}\cup\{(2^k,2^{k+2})\mid 0 \leq k \leq 4\}$ . All baselines and S'MoRE are trained with 2 epochs, with learning rate 1e-4. The learning rate follows a cosine schedule.

**Software & hardware.** We implement S'MoRE by adding a customized adapter to the Hugging Face PEFT library [Mangrulkar et al., 2022]. All models are trained via the LLaMA-Factory [Zheng et al., 2024] SFT pipeline, ensuring a consistent execution environment. Similarly, all the evaluations are conducted through OpenCompass [Contributors, 2023b], which is a unified evaluation framework providing a standard API for all considered benchmarks. For the computation hardware, all experiments are run on a single node with 4 NVIDIA A100 80GB GPUs.

<span id="page-36-4"></span><span id="page-36-3"></span> $<sup>^{10}</sup>$  Number of active experts" is only set for the sparse gates ("noisy top-k" and "switch"). For dense gates, the number of active experts equals total number of experts.

<span id="page-36-5"></span><sup>&</sup>lt;sup>11</sup>Same as above, the fanouts are only set for sparse gates. For dense gates, the fanout of layer  $\ell$  equals the total number of experts in layer  $\ell$ 

Table 7: Wall-clock time (second) comparison

<span id="page-37-1"></span>

| Method<br>ARC-c       | ARC-e | CSQA | OBQA | Winogrande | Average                                                                             |
|-----------------------|-------|------|------|------------|-------------------------------------------------------------------------------------|
| MixLoRA 426<br>S'MoRE | 794   | 3343 | 3539 | 3007       | 2222<br>489 (1.15×) 957 (1.21×) 4289 (1.28×) 4406 (1.24×) 4014 (1.33×) 2831 (1.24×) |

### D.3 Wall-clock time & potential system optimizations

While [§3.4](#page-5-4) ensures that S'MoRE theoretically incurs negligible computation overhead, it is true that without system-level optimization, the multi-layer structure may increase the wall-clock time. Yet, such overhead is small.

Measurement. Table [7](#page-37-1) shows the wall-clock time to finish training of MixLoRA and S'MoRE, measured on the same machine (with 4 NVIDIA A100 GPUs) and same software environment (based on LLaMA-Factory). The backbone model is LLaMA 3-8B. Trainable parameters of MixLoRA (8 rank-64 experts) and S'MoRE (2 layers, each with 4 rank-64 experts) are comparable.

On average, S'MoRE incurs 24% wall-clock time overhead, which is relatively small. The above measurement is based on S'MoRE under native PyTorch implementation, without any system optimization. It is reasonable to expect that the wall-clock time overhead can be further reduced by applying standard techniques, such as

- CUDA kernel fusion, which combines the operation of multiple S'MoRE layers into a single CUDA kernel. This can effectively reduce the "kernel launch" overhead associated with deeper S'MoRE (in native PyTorch, each layer may require its own "kernel launch").
- Token-level parallelism, which interleaves the processing of different layers across different tokens. This is achievable by custom Triton kernels or torch.compile(..) optimization. Such parallelism addresses the load-balance between the router and expert layers (since the router is more lightweight than the expert propagation), which improves GPU utilization. Such parallelism can also break the dependency between the top-down routing and bottomup propagation, as these two stages can be interleaved across tokens.

