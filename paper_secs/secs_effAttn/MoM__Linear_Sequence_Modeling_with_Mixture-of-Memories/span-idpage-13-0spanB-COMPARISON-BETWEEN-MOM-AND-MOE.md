# <span id="page-13-0"></span>B COMPARISON BETWEEN MOM AND MOE

While our approach to implementing the Mixture-of-Memories (MoM) draws inspiration from the Mixture-of-Experts (MoE) framework, there are significant differences that distinguish our method from traditional MoE implementations.

• Purpose: The MoE was introduced to scale up the number of parameters without significantly increasing computational resources. It address the limitations of dense models in scaling both parameters and computational demands through sparse activation. However, MoM is designed to expand the memory capacity of linear attention models while preserving their linear time complexity. By sparsely activating memories and using weighed summation to create a mixed memory, MoM effectively address the challenge of forgetting

<span id="page-14-1"></span>Table 7: **Results on Common-Sense Reasoning Tasks.** The performance of linear models and Transformer models is comparable; however, MoM consistently achieves the best average performance across all model sizes.

| Scale          | Model                  | <b>Wiki.</b><br>ppl↓ | <b>Lamb.</b> ppl↓ | ARC-e<br>acc↑ | ARC-c<br>acc <sub>n</sub> ↑ | Hella.<br>acc <sub>n</sub> ↑ | Lamb.<br>acc↑ | PIQA<br>acc↑ | Wino.<br>acc↑ | Avg.  |
|----------------|------------------------|----------------------|-------------------|---------------|-----------------------------|------------------------------|---------------|--------------|---------------|-------|
| 380M Params    | Transformer++          | 26.88                | 76.46             | 44.91         | 25.94                       | 34.95                        | 26.90         | 64.31        | 51.07         | 41.35 |
| 15B Tokens     | RetNet                 | 31.07                | 87.11             | 44.49         | 23.04                       | 33.86                        | 23.93         | 63.49        | 52.33         | 40.19 |
| L=24, $d=1024$ | HGRN2                  | 27.90                | 77.40             | 45.24         | 23.63                       | 35.61                        | 24.74         | 65.45        | 54.06         | 41.46 |
|                | GLA                    | 28.78                | 79.95             | 44.53         | 22.27                       | 34.84                        | 24.94         | 63.93        | 51.38         | 40.32 |
|                | GSA                    | 28.17                | 82.50             | 45.50         | 24.23                       | 35.00                        | 24.02         | 64.85        | 50.43         | 40.67 |
|                | Gated DeltaNet         | 26.47                | 58.59             | 46.04         | 23.55                       | 35.18                        | 27.01         | 66.05        | 50.83         | 41.44 |
|                | MoM                    | 25.86                | 55.41             | 44.65         | 24.74                       | 36.54                        | 27.93         | 66.16        | 51.78         | 41.97 |
| 1.3B Params    | Transformer++†         | 17.61                | 19.29             | 55.01         | 28.07                       | 49.21                        | 40.95         | 70.08        | 56.27         | 49.93 |
| 100B Tokens    | RetNet <sup>†</sup>    | 18.18                | 21.97             | 57.49         | 26.88                       | 48.09                        | 37.75         | 69.37        | 53.28         | 48.81 |
| L=24, $d=2048$ | HGRN2 <sup>†</sup>     | 17.32                | 15.65             | 58.33         | 28.07                       | 51.93                        | 42.31         | 71.33        | 52.01         | 50.66 |
|                | $\mathrm{GLA}^\dagger$ | 17.61                | 19.66             | 55.18         | 27.56                       | 48.89                        | 40.03         | 69.86        | 53.91         | 49.24 |
|                | $GSA^{\dagger}$        | 16.69                | 16.02             | 58.33         | 28.33                       | 50.98                        | 42.03         | 72.25        | 53.43         | 50.89 |
|                | Gated DeltaNet         | 17.14                | 18.80             | 56.82         | 27.39                       | 49.77                        | 39.94         | 71.76        | 51.78         | 49.58 |
|                | MoM                    | 16.64                | 14.83             | 55.35         | 27.99                       | 50.95                        | 43.43         | 71.27        | 56.83         | 50.97 |

historical information in linear attention. Moreover, by separating the memory into distinct states, MoM reduces interference between different pieces of information.

• Structure: In conventional MoE, each expert is a separate neural network within the feed-forward network (FFN) layer such as Qwen-MoE (Team, 2024) and Linear-MoE Sun et al. (2024a). In contrast, in MoM, each memory is an RNN state with unique key-value projection weights to generate different key-value pairs. MoE operates during the channel mixing phase, where each token is processed independently by selected experts. On the other hand, MoM functions during the token mixing phase, where each memory processes different segments of the sequence, preserving inter-token relationships.

#### <span id="page-14-0"></span>C EXPERIMENTS DETAILS

For the 380M models, we train on 15B tokens with a batch size of 0.5M tokens. The warmup tokens count is set to 0.25M. We set the hidden ratio of our model to 3 to keep the activated parameter count approximately the same. For the 1.3B models, we train on 100B tokens with a batch size of 2M tokens. The warmup tokens count is 1B. We employ AdamW optimizer (Loshchilov et al., 2017; Sun et al., 2024b) with learning rate of 3e-4 with cosine learning rate schedule (Zhou et al., 2020). The weight decay is set to 0.01 and gradient clipping is 1.0. Our experiments were conducted using 32 NVIDIA A800 GPUs. Training the 380M parameter model required approximately 10 hours, while the 1.3B parameter model took around 6 days.

#### D COMMONSENSE REASONING TASKS

As shown in Table 7, we report the language modeling perplexity and zero-shot performance of commonsense reasoning tasks following (Zhang et al., 2024) which includes WikiText (Merity et al., 2016), LAMBADA (Paperno et al., 2016), ARC-easy, ARC-challenge (Clark et al., 2018), HellaSwag (Zellers et al., 2019), PiQA (Bisk et al., 2020) and WinoGrande (Sakaguchi et al., 2019). The evaluation results are based on the lm-evaluation-harness (Gao et al., 2024).

Experimental results show that MoM outperforms other linear models and surpassed the Transformer model as well.

#### E Training Loss Comparison

To further assess the learning efficiency of MoM, we compared the training loss curves of MoM with those of other baseline models. As depicted in Figure 7, MoM consistently maintains the lowest loss

<span id="page-15-1"></span>throughout the entire training phase. Even as training nears convergence, MoM continues to exhibit a clear advantage over other methods.

![](_page_15_Figure_1.jpeg)

Figure 7: Training Loss. Loss curves for training 380M models on 15B tokens with a fixed random seed of 42.

