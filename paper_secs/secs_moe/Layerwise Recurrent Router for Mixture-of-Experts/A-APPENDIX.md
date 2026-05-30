# A APPENDIX

## <span id="page-14-0"></span>A.1 MORE RELATED WORKS

Routing Strategies While most MoE works follow the original success and use token choice routing, some works explore different routing approaches. In Expert-Choice Routing [\(Zhou et al.,](#page-13-9) [2022\)](#page-13-9), each expert selects tokens to process across the whole batch input. This method avoids expert imbalance issues and allows different tokens to be processed by a flexible number of experts. Soft Mixture-of-Experts [\(Puigcerver et al.,](#page-12-11) [2023\)](#page-12-11) further assigns token weights for input tokens, weighted-averages them, and passes these merged tokens to different experts. This method moves one step behind the Expert-Choice Routing to allow more precise control. However, their tokenselecting operations are non-causal and thus can't be directly used in the decoder models. Recent works [\(Huang et al.,](#page-11-9) [2024;](#page-11-9) [Yang et al.,](#page-13-10) [2024\)](#page-13-10) introduce dynamic top-k for each input token. While the FLOPs can be reduced, since this dynamic assignment can hurt the parallel computation of experts, more system-level implementation must be optimized to achieve wall-time efficiency. Some works also analyze issues in the routing of standard MoE like uncertain tokens [\(Wu et al.,](#page-13-11) [2024\)](#page-13-11) and lack of expert knowledge transfer [\(Zhao et al.,](#page-13-12) [2024\)](#page-13-12).

Training Strategies Due to the unstable nature of MoE [\(Zoph et al.,](#page-13-5) [2022\)](#page-13-5), some works investigate special training strategies for MoE. EvoMoE [\(Nie et al.,](#page-12-12) [2021\)](#page-12-12) uses a large top-k (even equal to the expert number) at the beginning of training, gradually decreasing k. StableMoE [\(Dai et al.,](#page-10-10) [2022\)](#page-10-10) proposes to freeze the router after training some tokens to avoid token assignment conflicts. Residual Mixture of Experts [\(Wu et al.,](#page-13-8) [2022\)](#page-13-8) initializes MoE from dense training checkpoints and finds it is an efficient method to train MoE models. Later, sparse-upcycling [\(Komatsuzaki et al.,](#page-11-3) [2023\)](#page-11-3) further trains large-scale language models from dense checkpoints, and many works follow this paradigm to efficiently utilize the power of MoE in fine tuning [\(Li et al.,](#page-11-10) [2023\)](#page-11-10), instruction tuning [\(Lin et al.,](#page-11-11) [2024\)](#page-11-11), and visual instruction tuning [\(Ding et al.,](#page-10-11) [2024\)](#page-10-11). Different from directly training MoE models, some works continue training the same pre-trained model on several different datasets to encourage specialization and combine them, either merging them into an MoE-style model [\(Gururangan et al.,](#page-11-12) [2021;](#page-11-12) [Sukhbaatar et al.,](#page-13-13) [2024\)](#page-13-13) or keeping a group of models and introducing a model-level router [\(Li et al.,](#page-11-13) [2022;](#page-11-13) [Gururangan et al.,](#page-11-14) [2023\)](#page-11-14).

Recurrence Controller A series of works introduce recurrent networks for Neural Architecture Search (NAS) [\(Zoph & Le,](#page-13-14) [2016;](#page-13-14) [Ramachandran et al.,](#page-12-13) [2017;](#page-12-13) [Pham et al.,](#page-12-14) [2018;](#page-12-14) [Liu et al.,](#page-11-15) [2018\)](#page-11-15). They introduce a recurrent controller network that predicts the current layer-i's architecture (like CNN filters' number, size, and stride) based on layer-i's input hidden states and previous recurrent states [\(Zoph & Le,](#page-13-14) [2016\)](#page-13-14). While these works use RNN to predict model architecture configurations of each layer for all inputs, RMoE utilizes RNN to help the router select expert combinations for each token, which can be viewed as a dynamic version of NAS.

## <span id="page-14-1"></span>A.2 EXPERIMENT SETUP

<span id="page-14-2"></span>Enwiki8 and WikiText-103 We follow the default configurations in CompeteSMoE [\(Pham et al.,](#page-12-7) [2024\)](#page-12-7). Each model is trained for 80,000 steps with Adam optimizer. The learning rate is 0.0007 with 4000 warmup steps, and the batch size is 48. The main used model is a decoder-only transformerbased architecture with 8 layers and a hidden size of 352. It includes 16 experts, where the top 2 are selected during computation, each with an expert size of 352. The model uses 8 attention heads and handles sequences up to 512 tokens in length, with an attention span of 2048 tokens. It incorporates a dropout rate of 0.1 and a load balancing factor of 0.01 to ensure an even distribution of expert utilization. Computation Cost Each 8-layer model is trained on one NVIDIA-A100 GPU for approximately 21 hours.

<span id="page-14-3"></span>Large Scale Pre-training For model architecture, our 24-layer model employs Rotary Embedding for positional encoding, SwiGLU for activation functions, and RMSNorm to enhance the model's efficiency and performance. Other model configuration includes a hidden size of 1280, 20 attention heads, an initialization method standard deviation of 0.02, a sequence length of 4096, and a maximum positional embedding length of 4096. All dropout rates are set to 0. For the MoE part, we use 16 experts, with each expert having a feedforward network hidden size of 448, following the fine-grained MoE settings, and each token activating 4 experts. We use a tokenizer with a 96512 vocabulary size, which adds approximately 123M embedding parameters and 123M vocabulary projection head parameters. Under this configuration, each model has approximately 664M non-embedding parameters, and every token activates 334M non-embedding parameters. The total parameter is around 910M. For pre-training configurations, we use a global batch size of 1120, a warmup period of 2000 iterations, a learning rate of 4.2e-4, a minimum learning rate of 4.2e-5, cosine learning rate decay, Adam optimizer with β<sup>1</sup> = 0.9 and β<sup>2</sup> = 0.95, a weight decay of 0.1, and gradient clipping at 1.0. Computation Cost Each 24-layer model is trained on 8 NVIDIA-A100 GPUs for approximately 5 days.

Instruction Tuning Data The Alpaca [\(Taori et al.,](#page-13-4) [2023\)](#page-13-4) dataset is an open-source instructionfollowing dataset created by Stanford researchers, inspired by OpenAI's ChatGPT. The dataset consists of 52,000 instruction-response pairs generated using the text-davinci-003 model by providing diverse and comprehensive instructions and recording the corresponding responses. It is designed to facilitate the training and evaluation of models in understanding and generating human-like text responses to various instructions.

Instruction Tuning Setting We use the codebase[2](#page-15-1) and corresponding default configurations. More concretely, we use bfloat16 (bf16) precision to accelerate training while maintaining numerical stability. The model is trained for 3 epochs using AdamW optimizer with a global batch size 128. We set the learning rate to 2e-5 and do not apply weight decay. A warmup ratio of 0.03 is used to gradually increase the learning rate at the beginning of training, and we utilize a cosine learning rate scheduler to adjust it throughout the training process, promoting smoother convergence. Computation Cost Each is trained on 8 NVIDIA-A100 GPUs for approximately 2 hours.

Evaluation Tasks Here we shortly describe our used evaluation datasets:

ARC-Easy is a subset of the AI2 Reasoning Challenge (ARC) dataset [\(Clark et al.,](#page-10-12) [2018\)](#page-10-12). It consists of multiple-choice questions from elementary and middle school science exams that are relatively easier than the ARC-Challenge set. These questions require basic reasoning and knowledge application.

Hellaswag [\(Zellers et al.,](#page-13-15) [2019\)](#page-13-15) is a dataset designed for commonsense reasoning and narrative prediction. It involves choosing the most plausible continuation of a given scenario from multiple options. The task is challenging because it requires understanding and applying common sense knowledge.

PIQA [\(Bisk et al.,](#page-10-13) [2020\)](#page-10-13) dataset tests a model's ability to understand and reason about physical interactions and affordances. The task involves selecting the correct answer to questions about everyday physical activities.

SciQ [\(Welbl et al.,](#page-13-16) [2017\)](#page-13-16) is a dataset of science questions that includes multiple-choice and directanswer formats. It aims to test a model's ability to understand and reason with scientific concepts typically taught at the school level.

LAMBADA [\(Paperno et al.,](#page-12-15) [2016\)](#page-12-15) is a dataset designed for language modeling and comprehension. The task involves predicting the last word of a given passage, which requires a deep understanding of the context provided by the preceding text.

## <span id="page-15-0"></span>A.3 FURTHER PRETRAINING VALIDATION

To further validate the scalability of RMoE, we conduct experiments with larger model sizes and increased pre-training corpus. Both MoE models followed the design principles of DeepSeek-MoE [\(Dai et al.,](#page-10-3) [2024\)](#page-10-3), utilizing fine-grained experts and shared experts to maintain strong baselines. We evaluated the models on more challenging benchmarks, including Hellaswag, MMLU, GSM8K, and HumanEval, to assess their language capabilities, multi-domain knowledge, mathematical skills, and coding abilities. Additionally, we tested the models' perplexity on multiple domain test datasets and reported the average results.

<span id="page-15-1"></span>https://github.com/tatsu-lab/stanford alpaca

Tab. [9](#page-16-0) and Tab. [10](#page-16-1) present the performance of a 15-billion parameter model with 2.7 billion activated experts, trained on 120 billion and 400 billion tokens, respectively. The results show that RMoE consistently delivers improvements even with increased data volumes. The findings indicate that RMoE enhances performance in standard language modeling tasks, such as Hellaswag and PPL, and improves performance on more complex reasoning tasks.

<span id="page-16-0"></span>Table 9: Performance comparison of SMoE, SMoE-MLP and RMoE at the model scale of 15B activation 2.7B parameters, training 120B tokens.

|                     | Hellaswag            | MMLU                 | GSM8K | Avg PPL |  |  |  |  |  |
|---------------------|----------------------|----------------------|-------|---------|--|--|--|--|--|
| Pretrain 80B Tokens |                      |                      |       |         |  |  |  |  |  |
| SMoE                | 67.69                | 46.24                | 24.18 | 7.406   |  |  |  |  |  |
| SMoE-MLP            | 67.98                | 46.47                | 23.58 | 7.437   |  |  |  |  |  |
| RMoE                | 68.00                | 47.74                | 27.14 | 7.361   |  |  |  |  |  |
|                     | Pretrain 100B Tokens |                      |       |         |  |  |  |  |  |
| SMoE                | 70.98                | 50.61                | 30.78 | 6.754   |  |  |  |  |  |
| SMoE-MLP            | 70.8                 | 50.6                 | 30.17 | 6.786   |  |  |  |  |  |
| RMoE                | 71.02                | 51.74                | 32.98 | 6.732   |  |  |  |  |  |
|                     |                      | Pretrain 120B Tokens |       |         |  |  |  |  |  |
| SMoE                | 72.03                | 52.79                | 34.8  | 6.447   |  |  |  |  |  |
| SMoE-MLP            | 72.19                | 52.81                | 34.57 | 6.479   |  |  |  |  |  |
| RMoE                | 72.36                | 54.02                | 36.13 | 6.425   |  |  |  |  |  |

<span id="page-16-1"></span>Table 10: Performance comparison of SMoE, SMoE-MLP and RMoE at the model scale of 15B activation 2.7B parameters, training 400B tokens.

|                      | Hellaswag | MMLU                 | GSM8K | Avg PPL |  |  |  |  |  |
|----------------------|-----------|----------------------|-------|---------|--|--|--|--|--|
| Pretrain 200B Tokens |           |                      |       |         |  |  |  |  |  |
| SMoE                 | 69.48     | 49.96                | 33.21 | 7.718   |  |  |  |  |  |
| SMoE-MLP             | 69.76     | 50.27                | 31.77 | 7.736   |  |  |  |  |  |
| RMoE                 | 70.00     | 52.21                | 32.98 | 7.608   |  |  |  |  |  |
|                      |           | Pretrain 280B Tokens |       |         |  |  |  |  |  |
| SMoE                 | 72.40     | 54.66                | 42.61 | 6.477   |  |  |  |  |  |
| SMoE-MLP             | 72.62     | 55.33                | 38.51 | 6.502   |  |  |  |  |  |
| RMoE                 | 73.18     | 56.06                | 44.35 | 6.400   |  |  |  |  |  |
|                      |           | Pretrain 400B Tokens |       |         |  |  |  |  |  |
| SMoE                 | 76.39     | 59.54                | 52.16 | 5.685   |  |  |  |  |  |
| SMoE-MLP             | 76.09     | 59.96                | 51.71 | 5.709   |  |  |  |  |  |
| RMoE                 | 76.72     | 60.60                | 52.99 | 5.620   |  |  |  |  |  |

## A.4 ADDITIONAL OBSERVATIONS

## <span id="page-17-0"></span>A.4.1 ROUTER GRADIENT NORM AND DROP RATIO

<span id="page-17-1"></span>Table 11: Comparison of linear and RNN routers in terms of gradients and drop ratios at various training steps. We record the router gradient every 10k training steps (20B tokens). We compute the gradient with language modeling (LM) loss and load balance (LB) loss. Drop ratio is the ratio of dropped tokens and all tokens as we assign capacity factor 1.0 for each expert.

| Training steps (k step)                                                          | 0.1                             | 10                              | 20                               | 30                              | 40                              | 50                              | 60                              |  |  |  |
|----------------------------------------------------------------------------------|---------------------------------|---------------------------------|----------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|--|--|--|
|                                                                                  | Linear router                   |                                 |                                  |                                 |                                 |                                 |                                 |  |  |  |
| grad from the whole loss<br>grad from LM loss<br>grad from LB loss<br>drop ratio | 1.058<br>0.625<br>0.433<br>35.6 | 0.194<br>0.183<br>0.011<br>5.43 | 0.1911<br>0.184<br>0.008<br>5.34 | 0.198<br>0.192<br>0.006<br>5.17 | 0.208<br>0.204<br>0.004<br>4.89 | 0.217<br>0.215<br>0.002<br>4.64 | 0.221<br>0.220<br>0.001<br>4.50 |  |  |  |
|                                                                                  |                                 | RNN router                      |                                  |                                 |                                 |                                 |                                 |  |  |  |
| grad from the whole loss<br>grad from LM loss<br>grad from LB loss<br>drop ratio | 0.972<br>0.636<br>0.337<br>38.7 | 0.160<br>0.146<br>0.014<br>6.35 | 0.153<br>0.138<br>0.015<br>6.30  | 0.153<br>0.139<br>0.014<br>5.94 | 0.155<br>0.144<br>0.011<br>5.32 | 0.155<br>0.148<br>0.007<br>4.54 | 0.154<br>0.151<br>0.003<br>4.09 |  |  |  |

Based on the setting of training 15B models for 120B tokens, we investigate how the gradient norm of the router varies throughout the entire training process. When training an MoE-based model, the gradient of the router has two separate sources: due to (1) the language modeling (LM) loss, and (2) the load balancing (LB) loss that forces the router to assign tokens to different experts in a balanced manner. Therefore, for each router, we compare the gradient from the LM loss only and from the whole training loss. We calculate the average for 100 training steps to estimate the gradient norm.

Furthermore, to better investigate the relation between the router behavior and the router gradient, we calculate the drop ratio for the router. This is because during the large-scale MoE pre-training, to ensure the training efficiency, the expert is usually controlled by an hyper-parameter called capacity factor, which determines the total tokens that one expert can process. If the router assigns tokens to some expert that exceeds its capacity, the expert will drop tokens with the lowest scores. And we define the drop ratio as tokens dropped / total tokens. The LB loss mentioned before is critical to decreasing the drop ratio.

According to Tab. [11,](#page-17-1) we have the following observations: 1. The gradient norm of the RNN router is generally smaller than that of the linear router. And for both routers, the drop ratio decreases with the training. 2. According to the drop ratios, we observe the significant behavioral difference between the two routers: during the early training phase (10k steps -¿ 30k steps), the drop ratio of the linear router is noticeably lower than that of the RNN router; the drop ratio of the RNN router archives at the lower value in the end. 3. The trend observed in the drop ratio is consistent with the results of the gradient norm. The grad norm for LB loss is relatively higher in the RNN router until the final training stage (50k - 60k), whereas the gradient from LB loss in the linear router is high at the beginning and generally low during the later part of training (10k - 60k).

These phenomena indicate that the LB loss could dominate the training of the linear router: when the drop ratio is low and stays unchanged, the grad from LB loss will be low because the router is already well-optimized for LB loss. Such early convergence in the LB loss may reach a suboptimal solution in the trade-off between optimizing load balance and language modeling. On the contrary, the gradient of the RNN router from LB loss stabilizes in the early training steps (10k - 30k), and the gradient from the lm loss keeps decreasing, suggesting that the RNN router is more optimized towards the LM loss.

```
A.4.2 MUTUAL INFORMATION
i m p o r t numpy as np
from s k l e a r n . m e t r i c s i m p o r t m u t u a l i n f o s c o r e
d e f d i s c r e t i z e p r o b d i s t ( p r o b d i s t , b i n s = 1 0 0 ) :
      """
      D i s c r e t i z e t h e p r o b a b i l i t y d i s t r i b u t i o n i n t o d i s c r e t e b i n s .
      """
      d i s c r e t i z e d = np . d i g i t i z e ( p r o b d i s t , b i n s =np . l i n s p a c e ( 0 , 1 , b i n s ) )
      r e t u r n d i s c r e t i z e d
d e f c a l c u l a t e m u t u a l i n f o r m a t i o n ( x1 , x2 , b i n s = 1 0 0 ) :
      """
      C a l c u l a t e mutual i n f o r m a t i o n between each p a i r of d i s t r i b u t i o n s i n x1 and x2 .
      x1 , x2 : numpy a r r a y s of shape (N, 16)
      b i n s : number of b i n s t o use f o r d i s c r e t i z a t i o n
      R e t u r n s a numpy a r r a y of mutual i n f o r m a t i o n v a l u e s .
      """
      m i v a l u e s = [ ]
      f o r i i n r a n g e ( x1 . shape [ 0 ] ) :
             x 1 d i s c r e t i z e d = d i s c r e t i z e p r o b d i s t ( x1 [ i ] , b i n s )
             x 2 d i s c r e t i z e d = d i s c r e t i z e p r o b d i s t ( x2 [ i ] , b i n s )
            mi = m u t u a l i n f o s c o r e ( x 1 d i s c r e t i z e d , x 2 d i s c r e t i z e d )
            m i v a l u e s . append ( mi )
      r e t u r n np . a r r a y ( m i v a l u e s )
A.4.3 EXPERT SIMILARITIES
d e f g e t s i m i l a r i t i e s ( htoh4 0 , htoh4 1 , h4toh ) :
      a v g k e y 0 = h t o h 4 0 . mean ( dim =1) # ( num experts , 4h , h )
      a v g k e y 1 = h t o h 4 1 . mean ( dim =1) # ( num experts , 4h , h )
      a v g v a l u e = h4toh . mean ( dim =2) # ( num experts , h , 4h )
      normed key 0 = nn . f u n c t i o n a l . n o r m a l i z e ( avg key 0 , p =2 , dim =1)
      normed key 1 = nn . f u n c t i o n a l . n o r m a l i z e ( avg key 1 , p =2 , dim =1)
      n or me d v al ue = nn . f u n c t i o n a l . n o r m a l i z e ( a v g v a l u e , p =2 , dim =1)
      n o r m e d a v g e x p e r t = t o r c h . c a t ( [ normed key 0 , normed key 1 , n or me d v al ue ] , dim =1)
      # compute t h e a v e r a g e e x p e r t s i m i l a r i t y
      s i m i l a r i t y = t o r c h .mm( n o r m e d a v g e x p e r t , n o r m e d a v g e x p e r t . t ( ) )
      avg sim = n o r m e d s i m i l a r i t y . mean ( ) . it em ( )
      r e t u r n avg sim
```

