# <span id="page-28-0"></span>**B. DeepSeek-V2-Lite: A 16B Model Equipped with MLA and DeepSeekMoE**

## <span id="page-28-1"></span>**B.1. Model Description**

**Architectures.** DeepSeek-V2-Lite has 27 layers and a hidden dimension of 2048. It also employs MLA and has 16 attention heads, where each head has a dimension of 128. Its KV compression dimension is 512, but slightly different from DeepSeek-V2, it does not compress the queries. For the decoupled queries and key, it has a per-head dimension of 64. DeepSeek-V2-Lite also employs DeepSeekMoE, and all FFNs except for the first layer are replaced with MoE layers. Each MoE layer consists of 2 shared experts and 64 routed experts, where the intermediate hidden dimension of each expert is 1408. Among the routed experts, 6 experts will be activated for each token. Under this configuration, DeepSeek-V2-Lite comprises 15.7B total parameters, of which 2.4B are activated for each token.

<span id="page-28-2"></span>

|         | Benchmark          | DeepSeek 7B | DeepSeekMoE 16B | DeepSeek-V2-Lite |
|---------|--------------------|-------------|-----------------|------------------|
|         | Architecture       | MHA+Dense   | MHA+MoE         | MLA+MoE          |
|         | Context Length     | 4K          | 4K              | 32K              |
|         | # Activated Params | 6.9B        | 2.8B            | 2.4B             |
|         | # Total Params     | 6.9B        | 16.4B           | 15.7B            |
|         | # Training Tokens  | 2T          | 2T              | 5.7T             |
|         | MMLU               | 48.2        | 45.0            | 58.3             |
|         | BBH                | 39.5        | 38.9            | 44.1             |
|         | TriviaQA           | 59.7        | 64.8            | 64.2             |
|         | NaturalQuestions   | 22.2        | 25.5            | 26.0             |
| English | ARC-Easy           | 67.9        | 68.1            | 70.9             |
|         | ARC-Challenge      | 48.1        | 49.8            | 51.2             |
|         | AGIEval            | 26.4        | 17.4            | 33.2             |
|         | HumanEval          | 26.2        | 26.8            | 29.9             |
| Code    | MBPP               | 39.0        | 39.2            | 43.2             |
|         | GSM8K              | 17.4        | 18.8            | 41.1             |
|         | MATH               | 3.3         | 4.3             | 17.1             |
| Math    | CMath              | 34.5        | 40.4            | 58.4             |
|         | CLUEWSC            | 73.1        | 72.1            | 74.3             |
| Chinese | C-Eval             | 45.0        | 40.6            | 60.3             |
|         | CMMLU              | 47.2        | 42.5            | 64.3             |

Table 6 | Performance of DeepSeek-V2-Lite, DeepSeekMoE 16B, and DeepSeek 7B.

**Training Details.** DeepSeek-V2-Lite is also trained from scratch on the same pre-training corpus of DeepSeek-V2, which is not polluted by any SFT data. It uses the AdamW optimizer with hyper-parameters set to <sup>1</sup> = 0.9, <sup>2</sup> = 0.95, and weight\_decay = 0.1. The learning rate is scheduled using a warmup-and-step-decay strategy. Initially, the learning rate linearly increases from 0 to the maximum value during the first 2K steps. Subsequently, the learning rate is multiplied by 0.316 after training about 80% of tokens, and again by 0.316 after training about 90% of tokens. The maximum learning rate is set to 4.2 × 10−<sup>4</sup> , and the gradient clipping norm is set to 1.0. We do not employ the batch size scheduling strategy for it, and it is trained with a constant batch size of 4608 sequences. During pre-training, we set the maximum sequence length to 4K, and train DeepSeek-V2-Lite on 5.7T tokens. We leverage pipeline parallelism to deploy different layers of it on different devices, but for each layer, all experts will be deployed on the same device. Therefore, we only employ a small expert-level balance loss with <sup>1</sup> = 0.001, and do not employ device-level balance loss and communication balance loss for it. After pre-training, we also perform long context extension and SFT for DeepSeek-V2-Lite and get a chat model called DeepSeek-V2-Lite Chat.

<span id="page-29-2"></span>

|         | Benchmark          | DeepSeek<br>7B Chat | DeepSeekMoE<br>16B Chat | DeepSeek-V2-Lite<br>Chat |
|---------|--------------------|---------------------|-------------------------|--------------------------|
|         | Architecture       | MHA+Dense           | MHA+MoE                 | MLA+MoE                  |
|         | Context Length     | 4K                  | 4K                      | 32K                      |
|         | # Activated Params | 6.9B                | 2.8B                    | 2.4B                     |
|         | # Total Params     | 6.9B                | 16.4B                   | 15.7B                    |
|         | # Training Tokens  | 2T                  | 2T                      | 5.7T                     |
|         | MMLU               | 49.7                | 47.2                    | 55.7                     |
|         | BBH                | 43.1                | 42.2                    | 48.1                     |
|         | TriviaQA           | 59.5                | 63.3                    | 65.2                     |
|         | NaturalQuestions   | 32.7                | 35.1                    | 35.5                     |
| English | ARC-Easy           | 70.2                | 69.9                    | 74.3                     |
|         | ARC-Challenge      | 50.2                | 50.0                    | 51.5                     |
|         | AGIEval            | 17.6                | 19.7                    | 42.8                     |
|         | HumanEval          | 45.1                | 45.7                    | 57.3                     |
| Code    | MBPP               | 39.0                | 46.2                    | 45.8                     |
|         | GSM8K              | 62.6                | 62.2                    | 72.0                     |
|         | MATH               | 14.7                | 15.2                    | 27.9                     |
| Math    | CMath              | 66.4                | 67.9                    | 71.7                     |
|         | CLUEWSC            | 66.2                | 68.2                    | 80.0                     |
| Chinese | C-Eval             | 44.7                | 40.0                    | 60.1                     |
|         | CMMLU              | 51.2                | 49.3                    | 62.5                     |

Table 7 | Performance of DeepSeek-V2-Lite Chat, DeepSeekMoE 16B Chat, and DeepSeek 7B Chat.

## <span id="page-29-0"></span>**B.2. Performance Evaluation**

**Base Model.** We evaluate the performance of DeepSeek-V2-Lite and compare it with our previous small-size base models in Table [6.](#page-28-2) DeepSeek-V2-Lite exhibits overwhelming performance advantages, especially in reasoning, coding, and math.

<span id="page-29-1"></span>**Chat Model.** We evaluate the performance of DeepSeek-V2-Lite Chat and compare it with our previous small-size chat models in Table [7.](#page-29-2) DeepSeek-V2-Lite also outperforms our previous small-size chat models by a large margin.

## C. Full Formulas of MLA

In order to demonstrate the complete computation process of MLA, we provide its full formulas in the following:

$$\mathbf{c}_{t}^{Q} = W^{DQ} \mathbf{h}_{t}, \tag{37}$$

$$[\mathbf{q}_{t1}^C; \mathbf{q}_{t2}^C; ...; \mathbf{q}_{tnh}^C] = \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q,$$
 (38)

$$[\mathbf{q}_{t,1}^{R}; \mathbf{q}_{t,2}^{R}; ...; \mathbf{q}_{t,n_h}^{R}] = \mathbf{q}_{t}^{R} = \text{RoPE}(W^{QR} \mathbf{c}_{t}^{Q}),$$
 (39)

$$\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R], \tag{40}$$

$$\boxed{\mathbf{c}_t^{KV}} = W^{DKV} \mathbf{h}_t, \tag{41}$$

$$\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R], \tag{40}$$

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t, \tag{41}$$

$$[\mathbf{k}_{t,1}^C; \mathbf{k}_{t,2}^C; ...; \mathbf{k}_{t,n_h}^C] = \mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \tag{42}$$

$$\mathbf{k}_{t}^{R} = \text{RoPE}(W^{KR}\mathbf{h}_{t}),$$

$$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^{C}; \mathbf{k}_{t}^{R}],$$
(43)

$$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R],\tag{44}$$

$$[\mathbf{v}_{t,1}^{C}; \mathbf{v}_{t,2}^{C}; ...; \mathbf{v}_{t,n_h}^{C}] = \mathbf{v}_{t}^{C} = W^{UV} \mathbf{c}_{t}^{KV}, \tag{45}$$

$$\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_{j} \left( \frac{\mathbf{q}_{t,i}^{T} \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}} \right) \mathbf{v}_{j,i}^{C}, \tag{46}$$

$$\mathbf{u}_{t} = W^{O}[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; ...; \mathbf{o}_{t,n_{h}}], \tag{47}$$

where the boxed vectors in blue need to be cached for generation. During inference, the naive formula needs to recover  $\mathbf{k}_t^C$  and  $\mathbf{v}_t^C$  from  $\mathbf{c}_t^{KV}$  for attention. Fortunately, due to the associative law of matrix multiplication, we can absorb  $W^{UK}$  into  $W^{UQ}$ , and  $W^{UV}$  into  $W^O$ . Therefore, we do not need to compute keys and values out for each query. Through this optimization, we avoid the computational overhead for recomputing  $\mathbf{k}_t^C$  and  $\mathbf{v}_t^C$  during inference.

#### <span id="page-30-0"></span>D. Ablation of Attention Mechanisms

#### <span id="page-30-1"></span>D.1. Ablation of MHA, GQA, and MQA

We show the evaluation results for 7B dense models with MHA, GQA, and MQA on four hard benchmarks in Table 8. All of these three models are trained on 1.33T tokens, and share the same architecture except for the attention mechanisms. In addition, for a fair comparison, we align the number of parameters of them to around 7B by adjusting the number of layers. From the table, we can find that MHA demonstrates significant advantages over GQA and MQA on these benchmarks.

#### <span id="page-30-2"></span>D.2. Comparison Between MLA and MHA

<span id="page-30-3"></span>In Table 9, we show the evaluation results for MoE models equipped with MLA and MHA, respectively, on four hard benchmarks. For a solid conclusion, we train and evaluate models across two scales. Two small MoE models comprise about 16B total parameters, and we train them on 1.33T tokens. Two large MoE models comprise about 250B total parameters, and we train them on 420B tokens. Also, two small MoE models and two large MoE models respectively share the same architecture except for the attention mechanisms. From the table, we can observe that MLA shows better performance than MHA. More importantly, MLA requires a significantly smaller amount of KV cache (14% for small MoE models and 4% for large MoE models) than MHA.

<span id="page-31-1"></span>

| Benchmark (Metric) | # Shots | Dense 7B<br>w/ MQA | Dense 7B<br>w/ GQA (8 Groups) | Dense 7B<br>w/ MHA |
|--------------------|---------|--------------------|-------------------------------|--------------------|
| # Params           | -       | 7.1B               | 6.9B                          | 6.9B               |
| BBH (EM)           | 3-shot  | 33.2               | 35.6                          | 37.0               |
| MMLU (Acc.)        | 5-shot  | 37.9               | 41.2                          | 45.2               |
| C-Eval (Acc.)      | 5-shot  | 30.0               | 37.7                          | 42.9               |
| CMMLU (Acc.)       | 5-shot  | 34.6               | 38.4                          | 43.5               |

Table 8 | Comparison among 7B dense models with MHA, GQA, and MQA, respectively. MHA demonstrates significant advantages over GQA and MQA on hard benchmarks.

<span id="page-31-2"></span>

| Benchmark (Metric)             | # Shots | Small MoE<br>w/ MHA | Small MoE<br>w/ MLA | Large MoE<br>w/ MHA | Large MoE<br>w/ MLA |
|--------------------------------|---------|---------------------|---------------------|---------------------|---------------------|
| # Activated Params             | -       | 2.5B                | 2.4B                | 25.0B               | 21.5B               |
| # Total Params                 | -       | 15.8B               | 15.7B               | 250.8B              | 247.4B              |
| KV Cache per Token (# Element) | -       | 110.6K              | 15.6K               | 860.2K              | 34.6K               |
| BBH (EM)                       | 3-shot  | 37.9                | 39.0                | 46.6                | 50.7                |
| MMLU (Acc.)                    | 5-shot  | 48.7                | 50.0                | 57.5                | 59.0                |
| C-Eval (Acc.)                  | 5-shot  | 51.6                | 50.9                | 57.9                | 59.2                |
| CMMLU (Acc.)                   | 5-shot  | 52.3                | 53.4                | 60.7                | 62.5                |

Table 9 | Comparison between MLA and MHA on hard benchmarks. DeepSeek-V2 shows better performance than MHA, but requires a significantly smaller amount of KV cache.

