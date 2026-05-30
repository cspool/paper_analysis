# <span id="page-3-0"></span>3.3 PROGRESSIVE SCALING ROUTING STRATEGY

Another key component of REXMOE is the Progressive Scaling Routing (PSR) strategy, which gradually increases the number of candidate experts during training. When reusing experts from r layers in a TopK MoE with N experts per layer, each router can access up to rN candidates. Instead of training the router to select from all rN candidates from the start, we adopt a progressive scheme: the number of available candidates begins at N and is linearly expanded over the course of training. At iteration t, the candidate expert pool size N<sup>t</sup> is defined as:

<span id="page-4-0"></span>Table 1: Base MoE architectures used in experiments. "MoE-0.5BA0.07B" denotes a MoE model with 0.5B total parameters and 0.07B active parameters per token. "SE" means "Shared Experts". This naming convention applies to all models.

| Model             | Hidden Size | Intermediate Size | #Layers | Heads<br>(Q / KV) | #Experts<br>(Shared + Routed / Total) |
|-------------------|-------------|-------------------|---------|-------------------|---------------------------------------|
| MoE-0.5BA0.07B    | 768         | 384               | 16      | 16/2              | 4/32                                  |
| MoE-0.5BA0.07B-SE | 768         | 384               | 16      | 16/2              | 1 + 3 / 32                            |
| MoE-2.3BA0.3B     | 512         | 744               | 32      | 16/2              | 8 / 64                                |
| MoE-2.3BA0.3B-SE  | 512         | 744               | 32      | 16/2              | 2 + 6 / 64                            |
| MoE-7BA3B-SE      | 2048        | 1408              | 32      | 16 / 4            | 2 + 6 / 64                            |

$$N_{t} = \begin{cases} N, & t \leq t_{s}, \\ \left\lfloor (1 + \frac{(r-1)(t-t_{s})}{t_{e}-t_{s}})N \right\rfloor, & t_{s} < t \leq t_{e}, \\ rN, & t > t_{e}, \end{cases}$$
 (6)

where  $t_s$  and  $t_e$  specify the start and end iterations of the scaling schedule, respectively. At each iteration, we randomly mask  $(rN-N_t)$  experts by setting their gating scores to zero before applying the TopK selection for each token. This design follows the principle of curriculum learning, allowing the model to gradually learn richer and more diverse expert representations.

#### 4 EXPERIMENTS

#### <span id="page-4-2"></span>4.1 EXPERIMENTAL SETUP

**Training environment.** All models are trained with Megatron-LM (Shoeybi et al., 2019), an open-source framework for large-scale language model training. We modified the MoE Block and Topk Router implementations to support cross-layer expert reuse and the Progressive Scaling Routing strategy during training. All models are pre-trained from scratch without instruction tuning, using the same hyperparameters across all runs. The sequence length is 4,096 and the total batch size is 512, resulting in a global batch size of 2M tokens. For optimization, we use AdamW (Loshchilov & Hutter, 2017) with  $\beta_1=0.9,\,\beta_2=0.95$ , weight decay 0.1, and a gradient clipping ratio of 1.0. The learning rate is scheduled to start at  $3\times 10^{-4}$  and decay to  $3\times 10^{-5}$  following a cosine schedule. Further details are provided in Appendix B.2. All training jobs are conducted on 4 nodes, each equipped with  $32\times NVIDIA$  Hopper GPUs.

**Model architecture.** We adopt the widely used Mixture-of-Experts (MoE) transformer architecture with consistent dimensionality settings across all ablation studies. The only differences lie in the router parameters under different reuse configurations. The architectural configurations are summarized in Table 1, where each model name specifies the number of activated and total parameters. The suffix "-SE" indicates that the architecture employs shared experts (Dai et al., 2024; Rajbhandari et al., 2022), and REX models follow the same naming convention. In addition, "-R $\{r\}$ " denotes that experts are reused across r layers.

**Training data.** We use the sample-100BT partition<sup>1</sup> from fineweb-edu dataset (Lozhkov et al., 2024; Penedo et al., 2024). The tokenizer is from LLaMA-2 (Touvron et al., 2023), with a vocabulary size of 32,000. Since the vocabulary is relatively small, the LLaMA-2 tokenizer does not achieve a high compression ratio. As a result, the processed 100B tokens cover around 87% of the original text. To ensure fair comparison, we fixed the data-parallel size and the shuffle seed, so that all experiments were trained on the same tokens in the same order, making the results directly comparable.

**Evaluation metrics.** We use lm-evaluation-harness (Gao et al., 2024) to evaluate performance on downstream tasks. Specifically, we report zero-shot accuracy on ARC-Easy (ARC-E) & ARC-Challenge (ARC-C) (Clark et al., 2018), BoolQ (Clark et al., 2019), HellaSwag (Zellers et al.,

<span id="page-4-1"></span><sup>&</sup>lt;sup>1</sup>https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/sample-100BT

<span id="page-5-0"></span>Table 2: Comparison between REXMOE and vanilla MoE models. All models are trained on 100B tokens. Task abbreviations: Hella. = HellaSwag, LAMB. = LAMBADA, Lg.QA = LogiQA, Op.QA = OpenBookQA, Wino. = WinoGrande. The best accuracy is highlighted in bold.

| Model                |       |       | ARC-E Hella. LAMB. Lg.QA Op.QA PIQA |       |       |       | SciQ  |       | SIQA Wino. Avg.↑ |       |
|----------------------|-------|-------|-------------------------------------|-------|-------|-------|-------|-------|------------------|-------|
| MoE-0.5BA0.07B       | 50.67 | 38.38 | 32.37                               | 28.42 | 31.00 | 65.29 | 71.20 | 38.84 | 53.04            | 45.47 |
| REX-0.5BA0.07B-R2    | 52.31 | 39.06 | 33.75                               | 25.65 | 32.80 | 65.78 | 71.10 | 38.33 | 51.22            | 45.56 |
| REX-0.5BA0.07B-R4    | 53.91 | 39.46 | 32.76                               | 25.35 | 32.80 | 66.81 | 71.00 | 38.38 | 52.17            | 45.85 |
| MoE-0.5BA0.07B-SE    | 51.85 | 38.90 | 33.26                               | 24.88 | 32.00 | 66.05 | 70.60 | 39.05 | 51.54            | 45.35 |
| REX-0.5BA0.07B-SE-R2 | 52.06 | 39.28 | 32.43                               | 26.57 | 35.00 | 66.54 | 71.80 | 37.41 | 51.93            | 45.89 |
| REX-0.5BA0.07B-SE-R4 | 53.11 | 39.39 | 34.00                               | 28.88 | 33.40 | 67.46 | 71.90 | 38.69 | 50.36            | 46.35 |
| MoE-2.3BA0.3B        | 58.42 | 47.14 | 37.55                               | 27.19 | 34.80 | 69.21 | 75.80 | 38.69 | 53.51            | 49.15 |
| REX-2.3BA0.3B-R2     | 61.32 | 46.84 | 37.20                               | 28.57 | 35.00 | 69.48 | 76.50 | 39.61 | 52.33            | 49.65 |
| REX-2.3BA0.3B-R4     | 60.94 | 47.96 | 38.75                               | 28.42 | 37.00 | 70.18 | 76.30 | 39.36 | 53.12            | 50.23 |
| MoE-2.3BA0.3B-SE     | 58.42 | 48.79 | 38.13                               | 25.35 | 37.00 | 69.53 | 75.00 | 40.28 | 52.17            | 49.41 |
| REX-2.3BA0.3B-SE-R2  | 59.09 | 47.99 | 38.54                               | 27.34 | 37.60 | 69.48 | 74.20 | 39.56 | 52.72            | 49.61 |
| REX-2.3BA0.3B-SE-R4  | 58.71 | 48.59 | 39.01                               | 28.26 | 39.00 | 70.67 | 76.10 | 39.66 | 52.80            | 50.31 |

[2019\)](#page-12-0), LAMBADA [\(Paperno et al.,](#page-11-8) [2016\)](#page-11-8), LogiQA [\(Liu et al.,](#page-10-11) [2021\)](#page-10-11), OpenBookQA [\(Mihaylov](#page-10-12) [et al.,](#page-10-12) [2018\)](#page-10-12), PIQA [\(Bisk et al.,](#page-9-11) [2020\)](#page-9-11), SciQ [\(Welbl et al.,](#page-11-9) [2017\)](#page-11-9), SIQA [\(Sap et al.,](#page-11-10) [2019\)](#page-11-10) and WinoGrande [\(Sakaguchi et al.,](#page-11-11) [2021\)](#page-11-11). For evaluation of the impact on inference speed after reusing experts from adjacent layers, we adapted REXMOE to vLLM [\(Kwon et al.,](#page-10-13) [2023\)](#page-10-13) and report the throughput (tokens per second) for prefill and decoding stages. Sampling is disabled in generation.

#### 4.2 MAIN RESULTS

### 4.2.1 EVALUATION ON DOWNSTREAM TASKS

Comparisons to vanilla MoEs. We report the accuracy on downstream benchmarks in [Table 2.](#page-5-0) The results show that the proposed REXMOE models consistently outperform vanilla MoE baselines across different model scales and benchmark tasks. Overall, REXMOE achieves stable improvements in both R2 and R4 configurations, with R4 often delivering the highest average accuracy. For example, compared to the base MoE-2.3BA0.3B, the R4 model attains the best results on tasks such as HellaSwag, LAMBADA, OpenBookQA, PIQA, and SIQA, raising the average score to 50.23%, which clearly surpasses the baseline's 49.15%. Similarly, under the "SE" setting, REXMOE-R4 outperforms the corresponding base MoE-2.3BA0.3B-SE. For smaller models in the MoE-0.5BA0.07B series, the advantage of REXMOE also remains consistent, where both R2 and R4 configurations yield notable gains in average accuracy over the baseline. More detailed task-wise accuracy trends during training can be found in [Figure 6](#page-14-0) and [Figure 7](#page-15-0) in the appendix. In summary, these results demonstrate that REXMOE consistently improves performance across different model scales and architectures, particularly on reasoning and knowledge-intensive tasks, highlighting its robustness, scalability, and general effectiveness.

<span id="page-5-1"></span>Table 3: Comparisons between REXMOE and open-source models. We report results for models with equivalent total or activated parameters on selected language understanding benchmarks. Our method achieves competitive or superior performance across tasks.

| Model                                                                                                         | #Act.            |               | Params Data ARC-E Hella. LAMB. Lg.QA PIQA SciQ Wino. |                      |                |                |                      |                |                      |
|---------------------------------------------------------------------------------------------------------------|------------------|---------------|------------------------------------------------------|----------------------|----------------|----------------|----------------------|----------------|----------------------|
| Llama2-7B (Touvron et al., 2023)<br>MPT-7B-Base (Team, 2023)                                                  | 7B/7B<br>7B/7B   | 2T<br>1T      | 76.4<br>67.3                                         | 78.6<br>76.1         | 73.9<br>70.3   | 30.7<br>-      | 78.1<br>79.9         | 93.7<br>-      | 69.3<br>68.3         |
| DeepSeekMoE-16B (Dai et al., 2024) 3B/16B<br>LLaMA-MoE-8B (Zhu et al., 2024)<br>OpenMoE-8B (Xue et al., 2024) | 3B/8B<br>2.1B/8B | 2T<br>-<br>1T | 68.1<br>60.2<br>64.1                                 | 77.1<br>70.8<br>45.5 | -<br>66.6<br>- | -<br>30.6<br>- | 80.2<br>77.5<br>74.2 | -<br>84.2<br>- | 70.2<br>63.6<br>60.3 |
| REX-7BA3B-SE-R3                                                                                               | 3B/7B            | 1T            | 75.7                                                 | 69.0                 | 63.9           | 33.2           | 75.0                 | 94.2           | 65.9                 |

Comparisons to LLMs with equivalent effective parameters We compare REXMOE with representative open-source dense and MoE models of similar total or activated parameter scales in [Ta-](#page-5-1) ble 3. For a fair comparison, we scale the training data of REX-7BA3B-SE-R3 to 1T tokens sampled from fineweb-edu. The model exhibits well-balanced performance, achieving highest results on LogiQA and SciQ, even when compared to Llama2-7B (Touvron et al., 2023), which uses more activated parameters and is trained on a larger corpus. Meanwhile, REXMOE remains highly competitive across the other benchmarks. These results demonstrate the effectiveness of REXMOE as model size and training data increase, highlighting its scalability for high performance.

#### 4.2.2 IMPACT ON INFERENCE SPEED

<span id="page-6-0"></span>![](_page_6_Figure_3.jpeg)

(a) Prefill goodput under different sequence length. (b) Decode goodput under different sequence length.

Figure 2: Comparison of prefill and decode goodput between base MoE and REX models. Numbers above the bars indicate the relative speedup over the base MoE.

We adapt REXMOE to the vLLM inference engine (Kwon et al., 2023) to evaluate the impact of expert reuse on practical applications. We fix the output length at 128 tokens and vary the input length to assess both prefill and decoding performance across different sequence lengths. The detailed results are shown in Figure 2. Although the computational overhead compared to vanilla MoE is negligible, REXMOE introduces a larger number of experts into each MoE block, which increases I/O operations during the prefill stage. As a result, the inference speed of the reuse scheme experiences a noticeable decline. As shown in Figure 2(a), a larger candidate expert pool leads to slower prefill speed, with the performance degradation being more pronounced when the input length is relatively short. Since the prefill stage usually accounts for only a small portion of the total time, the decoding stage is of greater practical importance. As illustrated in Figure 2(b), REXMOE achieves comparable performance across different sequence lengths in decoding stage.

