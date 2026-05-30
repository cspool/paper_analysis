# AQUILAMOE: EFFICIENT TRAINING FOR MOE MODELS WITH SCALE-UP AND SCALE-OUT STRATEGIES

Bo-Wen Zhang, Liangdong Wang, Ye Yuan, Jijie Li, Shuhao Gu, Mengdi Zhao, Xinya Wu, Guang Liu<sup>∗</sup> , Chengwei Wu, Hanyu Zhao, Li Du, Yiming Ju, Quanyue Ma, Yulong Ao, Yingli Zhao, Songhe Zhu, Zhou Cao, Dong Liang, Yonghua Lin, Ming Zhang, Shunfei Wang, Yanxin Zhou, Min Ye, Xuekai Chen, Xinyang Yu, Xiangjun Huang, Jian Yang†

> Beijing Academy of Artificial Intelligence (BAAI) School of Computer Science, Peking University MetaX-Tech

## ABSTRACT

In recent years, with the rapid application of large language models across various fields, the scale of these models has gradually increased, and the resources required for their pre-training have grown exponentially. Training an LLM from scratch will cost a lot of computation resources, while scaling up from a smaller model is a more efficient approach and has thus attracted significant attention. In this paper, we present AquilaMoE, a cutting-edge bilingual 8\*16B Mixture of Experts (MoE) language model that has 8 experts with 16 billion parameters each and is developed using an innovative training methodology called EfficientScale. This approach optimizes performance while minimizing data requirements through a two-stage process. The first stage, termed *Scale-Up*, initializes the larger model with weights from a pre-trained smaller model, enabling substantial knowledge transfer and continuous pretraining with significantly less data. The second stage, *Scale-Out*, uses a pre-trained dense model to initialize the MoE experts, further enhancing knowledge transfer and performance. Extensive validation experiments on 1.8B and 7B models compared various initialization schemes, achieving models that maintain and reduce loss during continuous pretraining. Utilizing the optimal scheme, we successfully trained a 16B model and subsequently the 8\*16B AquilaMoE model, demonstrating significant improvements in performance and training efficiency.

*K*eywords Mixture of Experts · Efficient Training · Model Initialization · Continuous Pretraining

## 1 Introduction

Language models have become a cornerstone of modern natural language processing (NLP) systems, driving applications such as machine translation, conversational agents, text summarization, and question answering [\[1,](#page-8-0) [2\]](#page-8-1). Recent advancements in large language models (LLMs) like GPT-3, BERT, and T5 have demonstrated remarkable proficiency across numerous tasks, highlighting the importance of pretraining on large-scale datasets to achieve state-of-the-art results [\[3,](#page-8-2) [4\]](#page-8-3). Despite their success, traditional dense models face significant challenges in scalability and efficiency, particularly as parameter sizes increase.

Mixture of Experts (MoE) models have emerged as a promising solution to these challenges. By dynamically selecting different subsets of model parameters (experts) for various inputs, MoE architectures can scale to a much larger number of parameters without a corresponding increase in computational cost [\[5\]](#page-8-4). This selective activation mechanism allows MoE models to achieve higher performance while maintaining computational efficiency. However, training such large-scale MoE models presents significant challenges, including the vast amounts of data and computational power required.

<sup>∗</sup> Project Lead, the corresponding author, contact <liuguang@baai.ac.cn>

<sup>†</sup> Full authorship contribution statements appear at the end of the document.

Training large-scale models, including MoE architectures, involves several critical challenges. Traditional training methods require enormous amounts of data, which can be resource-intensive and time-consuming to collect and process. The computational cost is substantial, requiring high-performance hardware such as GPUs or TPUs, and significant energy consumption, making it challenging for many institutions with limited resources to train and deploy such models. Additionally, training large models from scratch can take weeks or even months, delaying experimentation and iteration. Ensuring that the model efficiently learns and generalizes well is also challenging, as poor initialization and inefficient training strategies can lead to suboptimal performance and wasted resources.

Several strategies have been proposed to address these challenges. For instance, the Net2Net method accelerates learning via knowledge transfer, allowing the seamless transition of knowledge from smaller to larger networks, which shows significant acceleration in image classification task [\[6\]](#page-8-5). The StackBERT method improves training efficiency by progressively increasing model depth and capacity [\[7\]](#page-8-6). The bert2BERT approach focuses on reusing pre-trained language models to initialize new models, promoting efficiency and reusability [\[8\]](#page-8-7). It expands both the width and depth of the smaller model and finally saves nearly half of the pre-training consumption of language models. The primary motivation behind developing AquilaMoE is to introduce an efficient training framework, EfficientScale, which reduces data and computational requirements while enhancing overall model performance. Our approach leverages the strengths of MoE architectures and introduces innovative techniques to improve training efficiency and effectiveness.

In this paper, we introduce AquilaMoE, a bilingual 8\*16B Mixture of Experts language model that has 8 experts with 16 billion parameters each and is developed using the EfficientScale methodology. This approach optimizes performance and minimizes data needs through a two-stage process. The first stage, *Scale-Up*, leverages the weights of a pre-trained smaller model to initialize the larger model, enabling substantial knowledge transfer and continuous pretraining with significantly less data compared to traditional from-scratch training. The second stage, *Scale-Out*, uses a pre-trained dense model to initialize the MoE experts, further enhancing knowledge transfer and performance.

Through extensive validation experiments on 1.8B and 7B models, we compared various initialization schemes to achieve models that maintain and further reduce loss during continuous pretraining. Based on these findings, we utilized the optimal initialization scheme to successfully train a 16B model and subsequently the 8\*16B AquilaMoE model, demonstrating significant advancements in model performance and training efficiency.

## 2 Methodology

The EfficientScale pipeline is designed to efficiently train a large-scale Mixture of Experts (MoE) model by leveraging knowledge transfer from smaller models. The process involves three main phases: Preparation, Scale-Up, and Scale-Out. Each phase plays a crucial role in ensuring effective knowledge transfer and continuous learning, resulting in a highly optimized MoE model.

#### 2.1 Preparation Phase

The preparation phase involves training a small dense model and preparing the datasets required for subsequent phases. This phase ensures that the initial model has sufficient transferable knowledge and that the data is ready for effective training and validation.

- Model Training: Train a small dense model from scratch on a substantial amount of tokens or use an already pre-trained small model. This step ensures the model has accumulated sufficient transferable knowledge to serve as a robust starting point.
- Data Preparation: Collect, clean, and preprocess the training and validation datasets. This step involves managing large datasets to ensure they are suitable for training and validation purposes.
- Validation Setup: Develop both training and validation datasets to monitor the model's performance during subsequent phases. Continuous tracking of the language model's loss on the validation dataset is essential to ensure the initialized models retain transferred knowledge and can learn new information effectively.

#### 2.2 Scale-Up Phase

The Scale-Up phase involves two critical steps: initializing the weights of a larger dense model using the smaller model and performing continuous pretraining to ensure effective knowledge transfer and model enhancement. We use the bert2BERT[\[8\]](#page-8-7) method to initialize the large model and propose the AKI-Pro method, improving bert2BERT-AKI from depth expansion and group query attention.

![](_page_2_Figure_1.jpeg)

<span id="page-2-0"></span>Figure 1: An example of FPI on an MLP layer.

#### 2.2.1 Weight Initialization Strategies

The weights of the small dense model are used to initialize a larger dense model. There are two strategies proposed in bert2BERT[\[8\]](#page-8-7): Function Preserving Initialization(FPI) and Advanced Knowledge Initialization(AKI). Both the original and our experiments in Section [3.2.1](#page-4-0) show that AKI performs better. Besides, recent research[\[9\]](#page-9-0) shows that it is better to use interpolation instead of stacking when expanding the depth, which is more stable for continuous training. Moreover, the original AKI method is not suitable for Group Query Attention (GQA), so we modify the transformation of the weights in attention blocks to fit GQA. Finally, we have AKI-Pro as our initialization method. Below we will introduce the three initialization methods, starting with a review of the first two approaches in bert2BERT, followed by our improvements.

Function Preserving Initialization (FPI): This strategy is firstly proposed in Net2Net[\[6\]](#page-8-5) to expand the intermediate dim of an MLP layer. Bert2BERT[\[8\]](#page-8-7) enhances the Net2Net method to FPI, which enables it to expand the hidden dims(i.e. input and output dims). It is applied in training language models in bert2BERT and can expand the width of a smaller model to a larger model, getting the same output with the same input. With the FPI, the larger model can get the transferred knowledge from the smaller model. The basic idea behind FPI is that when expanding the dims, it makes both the input and output tensor concatenate a copy of the smaller tensor, as illustrated in Figure [1.](#page-2-0) For an MLP layer with two linear mappings in the example: y = U <sup>⊤</sup>W<sup>⊤</sup>x, the input and output dims are 2, and the intermediate dim is 3. Suppose we want to expand this block to that with 3 as input and output dims, and 4 as intermediate size, then there are three steps. (1) Input Dim Expansion FPI copies the input neurons from left to right and splits the corresponding weights to the new input neurons. (2) Output Dim Expansion For the expansion of the output in the upsampling linear weights, FPI also makes the new hidden neurons copy from the original ones. (3) MLP Expansion Expand the downsampling linear weights the same as the upsampling weights, and finally, the new output neurons of this MLP layers are also the copy from the smaller ones, which makes the block can be stacked as layers. The weights W′ = FPI(W) are transformed as follows:

$$w'_{1,*} = w'_{3,*} = \frac{w_{1,*}}{2}$$
 $w'_{*,4} = w'_{*,1}$ 
(1)

Most modules of a transformer block can be transformed the same as an MLP layer, including embedding layers and QKV projections. For the MHA module, each attention head should be seen as a neuron, and then the head number can be expanded as before. Notably, the output of the LN modules will not be the same when the new dimension is not an integer multiple of the old one, but this will not hurt a lot on the final loss.

Advanced Knowledge Initialization (AKI): As shown in both Net2Net[\[6\]](#page-8-5) and bert2BERT[\[8\]](#page-8-7), the symmetry from the FPI will hinder the model convergence. Specifically, if we have a linear layer y = w1x + w2x, where x, y ∈ R, and w<sup>1</sup> = w<sup>2</sup> when initializing the weights, the gradient and the value of these two weights will always be the same, which makes the effective number of parameters for this linear layer only 1. So AKI is proposed to break the symmetry with expanding width based on not only the weights of the same layer but also the upper layer in the smaller model. Take a model with two MLP blocks as an example:

$$y_{1} = U^{(1)\top} W^{(1)\top} x, y_{2} = U^{(2)\top} W^{(2)\top} y_{1}, x, y_{1}, y_{2} \in \mathbb{R}^{2}$$

$$W^{(1,2)} \in \mathbb{R}^{2\times 3}, U^{(1,2)} \in \mathbb{R}^{3\times 2}$$
(2)

FPI expands W<sup>1</sup> as FPI W(1) = h w ′(1) 1 ; w ′(1) 2 ; w ′(1) 3 ; w ′(1) 1 i , while AKI uses the output expansion of next layer: AKI W(1) = h w ′(1) 1 ; w ′(1) 2 ; w ′(1) 3 ; w ′(2) 1 i . Inspired by the observation that neighboring layers have similar functions, AKI breaks the symmetry and keep the knowledge from the smaller models. Moreover, FPI can't expand the depth, so bert2BERT uses the stacking method to expand the model depth proposed by StackBERT [\[7\]](#page-8-6).

AKI-Pro: Our proposed improvement on AKI further refines weight initialization from two aspects: depth growing method and GQA compatibility.

![](_page_3_Figure_1.jpeg)

<span id="page-3-0"></span>Figure 2: Comparison of different growing methods: stacking and interpolation.

• Depth Growing Method: We use interpolation in the depth growth to make the continuous training more stable, following the recent research [9]. The stacking method just copies the layers of the source model to the top. For the source model with  $L_1$  layers:  $\{W_l|l\in[0,L_1)\}$  and target model with  $L_2$  layers:  $\{W_l'|l\in[0,L_1)\}$ , stacking method can be formed as  $W_l'=W_{(l\mod L_1)}$ . However, the output space of the last layer does not match the input space of the first layer, which can make the continuous training unstable. Based on the observation of similar functionality in neighboring layers, recent research[9] improves this by using interpolation, which can be formulated as below:

$$W_l' = \lfloor \frac{l * L_2}{L_1} \rfloor \tag{3}$$

Figure 2 shows an example when  $L_1 = 3$ ,  $L_2 = 6$ . We show the comparison of validation losses and training curves after the depth growth with different methods in Section 3.2.1.

• **GQA Compatibility:** The original AKI method only supports MHA in transformer models. We adapt AKI for Group Query Attention models. To be specific, under the constraint that the number of groups in the GQA of the source model and the target model are consistent, we expand the output of the attention heads inside each group. Each group can be seen as a separate MHA block with common KV projection weights, and the expansion operator is the same as MHA.

#### 2.2.2 Continuous Pretraining Process

The scaled-up dense model undergoes continuous pretraining on a substantial amount of tokens. This phase ensures the successful transfer of knowledge and allows the model to acquire additional information from the data, enhancing its overall performance and capability.

#### 2.3 Scale-Out Phase

The scale-out phase involves transforming the large dense model into a Mixture of Experts (MoE) model. This phase includes initializing the MoE model's weights and performing continuous pretraining to refine the model's knowledge and performance.

- MoE Weight Initialization: Aquila-MoE is initialized using Sparse Upcycling [10, 11]. The dense model checkpoint obtained from the Aquila dense model undergoes a transformation where each MLP layer is replaced by an MoE layer. These new MoE layers are exact replicas of the original MLP layers from the dense checkpoint. The router parameters are randomly initialized following a normal distribution with a mean of 0 and a variance of 0.02.
- Continuous Pretraining of MoE: During both training and inference, two out of eight experts are activated for each token, resulting in approximately 30B activated parameters. To prevent training collapse, additional

load balancing loss [\[12\]](#page-9-3) and max z-loss [\[13,](#page-9-4) [14\]](#page-9-5) are applied to the final training objective. The auxiliary loss and max z-loss are multiplied by 0.001 and 0.01, respectively, to ensure a balanced distribution of tokens assigned to different experts and a stable training trajectory.

By following this structured approach, EfficientScale enables efficient training of large-scale models through systematic preparation, scaling up, and scaling out. This methodology leverages pre-trained smaller models to reduce data and computational requirements while ensuring efficient knowledge transfer and continuous learning. The result is a highly optimized MoE model capable of performing complex tasks with enhanced efficiency and performance.

## 3 Experiemnts

#### 3.1 Datasets Description

We constructed a bilingual pretraining dataset of 4TB tokens in both Chinese and English. This dataset includes webpages, arXiv papers, encyclopedic data, books, codes, and QA pairs. It covers a wide range of high-quality opensource pretraining data such as [RedPajama-Data-V2,](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) [falcon-refinedweb,](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) [C4,](https://huggingface.co/datasets/allenai/c4) [Pile,](https://huggingface.co/datasets/EleutherAI/pile) [WuDaoCorporaText,](https://data.baai.ac.cn/details/WuDaoCorporaText) [ChineseWebText,](https://huggingface.co/datasets/CASIA-LM/ChineseWebText) etc. The above open-source data underwent language filtering to retain only Chinese and English texts, heuristic refinement to remove low-quality content, deduplication to maintain uniqueness, domain-specific filtering for relevance, data quality checks, removal of toxic and explicit content, and finally, data mixing in specified proportions.

#### 3.2 Experimental Setups and Results

#### <span id="page-4-0"></span>3.2.1 Scale-up Validation

For the scale-up experiment, we used a 1.3B Aquila2 [3](#page-4-1) architecture model as the baseline. This model was scaled up to a 7B model using two different methods: FPI and AKI. Additionally, a 7B model was trained from scratch to serve as a control. All three 7B models were trained using the same hyperparameters and on the same dataset for a specified number of steps. We use M(24, 2048) to denote the 1.3B model with 24 layers and 2048 hidden dimensions and use M(32, 4096) to denote the 7B model. We first calculated the validation loss of models with different initializations. The results are shown in Table [1.](#page-4-2) We check the loss of an intermediate model M(24, 4096) without doing depth growth. We got exactly the same loss as the original model using FPI. Moreover, we found that with interpolation, both FPI and AKI have lower initial losses.

The loss convergence for the training process is shown in Figure [3.](#page-5-0) The experimental results indicate that the 7B models initialized using the FPI and AKI methods exhibited significantly lower loss values compared to the 7B model trained from scratch. Furthermore, these models converged at a notably faster rate. Consistent with findings in the paper [\[8\]](#page-8-7), our results also demonstrate that the AKI method surpasses FPI in performance after a certain number of steps.

| Method                  | M(24, 4096) | M(32, 4096) |
|-------------------------|-------------|-------------|
| M(24, 2048) (Original)  | 2 .97       |             |
| Random                  | -           | 12.22       |
| FPI (Stacking)          | 2 .97       | 4.30        |
| FPI (Interpolation)     | 2 .97       | 3.31        |
| AKI (Stacking)          | -           | 9.56        |
| AKI-Pro (Interpolation) | -           | 7.81        |

<span id="page-4-2"></span>Table 1: Validation losses of different initialization methods.

#### 3.2.2 Scale-out Validation

For the scale-out validation experiment, we trained a 1.8B model from scratch with a training data volume of 3.6T tokens. These models were then scaled out to 8\*1.8B configurations, followed by continuous pretraining with an additional 400B tokens. The respective model configurations and training hyperparameters are detailed in Table [3.](#page-6-0) We analyzed the loss convergence on the training set with the results depicted in Figure [4.](#page-5-1)

Based on the results of the aforementioned validation experiments, we verified the effectiveness of both scale-up and scale-out approaches on smaller-sized models. Specifically, we trained a model from scratch with a size of 7B, and

<span id="page-4-1"></span><sup>3</sup> https://github.com/FlagAI-Open/Aquila2

Table 2: Validation losses of the AquilaDense-16B initializations. M(32, 4096) is 7B. M(40, 5120) is 13B. M(32, 5120) and M(32, 8192) are for checking loss before depth growth.

| Method                 | M(32, 8192) | M(32, 5120) | M(40, 5120)  |
|------------------------|-------------|-------------|--------------|
| M(32, 4096) (Original) |             | 1 .85       |              |
| FPI<br>AKI-Pro         | 1 .85<br>-  | 1.96<br>-   | 2.24<br>7.90 |

![](_page_5_Figure_3.jpeg)

![](_page_5_Figure_4.jpeg)

<span id="page-5-0"></span>Figure 3: Comparison between the convergence of FPI and AKI methods.

<span id="page-5-1"></span>Figure 4: Training loss of AquilaMoE.

pre-trained it on 3.6T tokens, resulting in AquilaDense-7B. Subsequently, we scaled it up to a model with a size of 16B and further trained it on 1.2T tokens, yielding AquilaDense-16B. Finally, we scaled it out to 8\*16B and trained it on 545B tokens, ultimately obtaining AquilaMoE. The configurations and training parameters of the models are presented in Table [3.](#page-6-0)

# 4 Model Evaluation

#### 4.1 Evaluation of Foundation Models

Following OpenCompass[4](#page-5-2) , in the evaluation process, we use two types of evaluation methods: discriminant analysis evaluation and generative evaluation. Discriminant analysis evaluation means combining the question with candidate answers, calculating the perplexity of all combinations, and selecting the answer with the lowest perplexity as the model's final output. Generative evaluation uses the question as the model's original input and leaves the answer area blank for the model to complete subsequently.

The performance of AquilaDense-7B, AquilaDense-16B, and AquilaMoE(8\*16B) models are presented in Table [4.](#page-6-1) The indicators ending in "ppl" represent discriminant analysis evaluation, while those ending in "gen" represent generative evaluation.

Generally, as the model size increases, the scores tend to improve. For instance, AquilaDense-7B scores 7.81 on GSM8K-gen, while AquilaDense-16B scores 28.51. A similar trend also is observed in most other tasks. The AquilaMoE models show improved performance in most tasks over AquilaDense-16B. For example, in the ARC-c-ppl task, AquilaMoE scored 43.05 compared to 38.31 for AquilaDense-16B. These findings highlight the benefits of both scaling up model parameters and implementing MoE architectures in improving model performance.

#### 4.2 Evaluation of Fine-tuned Models

Table [5](#page-7-0) presents the overall results of AquilaMoE-8\*16B after fine-tuning across various benchmark datasets. The performance is measured using generative evaluation, and the results are expressed as percentages.

<span id="page-5-2"></span><sup>4</sup> https://github.com/open-compass

|                         | 1.8B    | 8*1.8B  | <b>7B</b> (AquilaDense-7B) | <b>16B</b> (AquilaDense-16B) | <b>8*16B</b> (AquilaMoE) |
|-------------------------|---------|---------|----------------------------|------------------------------|--------------------------|
| Context Length          | 2048    | 2048    | 4096                       | 4096                         | 4096                     |
| QKV Bias                | yes     | yes     | yes                        | yes                          | yes                      |
| Layers                  | 24      | 24      | 32                         | 40                           | 40                       |
| Hidden Dim              | 2048    | 2048    | 4096                       | 5120                         | 5120                     |
| <b>Intermediate Dim</b> | 5504    | 5504    | 14336                      | 20480                        | 20480                    |
| <b>Heads Num</b>        | 32      | 32      | 32                         | 40                           | 40                       |
| KV Group                | 32      | 32      | 32                         | 8                            | 8                        |
| Trained Tokens (B)      | 3600    | 400     | 3600                       | 1200                         | 545                      |
| LR                      | 1.20e-3 | 2.20e-4 | 1.20e-3                    | 4.00e-4                      | 1.50e-4                  |
| <b>Batch Size</b>       | 12M     | 12M     | 12M                        | 12M                          | 24M                      |

<span id="page-6-0"></span>Table 3: Model configurations and training parameters for different models.

Table 4: Overall evaluation results of AquilaDense and AquilaMoE(AquilaMoE-8\*16B)

<span id="page-6-1"></span>

| Model         | AquilaDense-7B | AquilaDense-16B | AquilaMoE |
|---------------|----------------|-----------------|-----------|
| ARC-c-ppl     | 37.63          | 38.31           | 43.05     |
| ARC-e-ppl     | 56.08          | 52.2            | 65.61     |
| Hellaswag-ppl | 67.49          | 71.62           | 73.94     |
| GSM8K-gen     | 7.81           | 28.51           | 54.51     |
| HumanEval-gen | 14.02          | 29.88           | 15.85     |
| MMLU-ppl      | 46.47          | 57.11           | 61        |
| Winograd-ppl  | 50.53          | 54.04           | 55.4      |
| MATH-gen      | 1.32           | 4.24            | 10.4      |
| MBPP-gen      | 15.6           | 36.4            | 37.2      |
| DROP-gen      | 4.35           | 33.35           | 37.62     |
| AGI Eval-gen  | 14.47          | 18.57           | 13.69     |
| BBH-gen       | 34.51          | 41.45           | 46.04     |
| NQ-gen        | 8.61           | 9.94            | 10.78     |
| PIQA-ppl      | 76.71          | 79.22           | 80.3      |

#### 4.3 Comparsion of Computational Efficiency

We present the details of the training process for both scale-up + scale-out and from-scratch approaches in Table 6. The table lists the number of devices in the cluster, the GFLOPS per device, the model parameters size, the number of training tokens, the actual running tokens per day, the actual training time, and the actual training GFLOPS for each phase.

The time savings factor is calculated by comparing the total training time of the from-scratch approach to the total training time of the scale-up and scale-out approach. The formula is:

$$\text{Time Savings Factor} = \frac{\frac{\sum_{i=1}^{n} N_{\text{tokens},i}}{R_{\text{tokens/day, from scratch}}}}{\sum_{i=1}^{n} \frac{N_{\text{tokens},i}}{R_{\text{tokens/day},i}}}$$

Given the data:

Time Savings Factor = 
$$\frac{\frac{3600+1200+545}{25}}{\frac{3600}{279}+\frac{1200}{70}+\frac{545}{25}} = \frac{213.80}{51.84} \approx 4.12$$

The computational power savings factor is calculated by comparing the total GFLOPS-days of the from-scratch approach to the total GFLOPS-days of the scale-up and scale-out approach. The formula is:

$$\text{Computational Power Savings Factor} = \frac{\frac{\sum_{i=1}^{n} N_{\text{tokens,}i} \times \text{GFLOPS}_{\text{from scratch}}}{R_{\text{tokens,}i} \times \text{GFLOPS}_{\text{from scratch}}}}{\sum_{i=1}^{n} \frac{N_{\text{tokens,}i} \times \text{GFLOPS}_{i}}{R_{\text{tokens,}i \times \text{MISM}}}}$$

<span id="page-7-0"></span>

| Model      | AquilaMoE-8*16B-SFT |
|------------|---------------------|
| ARC-c      | 82.03               |
| ARC-e      | 87.3                |
| Hellaswag  | 75.08               |
| GSM8K      | 71.27               |
| NQ         | 21.39               |
| TriviaQA   | 65.33               |
| AGI Eval   | 13.61               |
| Math       | 13.26               |
| HumanEval  | 44.51               |
| PIQA       | 81.72               |
| OBQA       | 75.2                |
| DROP       | 62.32               |
| BoolQ      | 85.02               |
| GPQA       | 25.76               |
| C-Eval     | 57.99               |
| MMLU       | 61.51               |
| CMMLU      | 57.63               |
| Winogrande | 57.54               |
|            |                     |

Table 5: Overall results of AquilaMoE after fine-tuning.

<span id="page-7-1"></span>Table 6: Training details for scale-up and scale-out and from-scratch approaches, note that for preparation phase different chip is used.

| Approach/Phase    | Devices | GFLOPS/Device | Model Size (B) | Trained Tokens (B) | Training Tokens/Day (B) |
|-------------------|---------|---------------|----------------|--------------------|-------------------------|
| Preparation Phase | 480     | 989.5         | 7              | 3600               | 279                     |
| Scale-Up Phase    | 1024    | 240           | 16             | 1200               | 70                      |
| Scale-Out Phase   | 1024    | 240           | 32             | 545                | 25                      |
| From Scratch      | 1024    | 240           | 32             | 5345               | 25                      |

Given the data:

$$\begin{aligned} & \text{GFLOPS}_{\text{preparation}} = 480 \times 989.5 = 475, 360 \\ & \text{GFLOPS}_{\text{scale-up}} = 1024 \times 240 = 245, 760 \\ & \text{GFLOPS}_{\text{scale-out}} = 1024 \times 240 = 245, 760 \\ & \text{GFLOPS}_{\text{from scratch}} = 1024 \times 240 = 245, 760 \end{aligned}$$

The computational power savings factor is:

Computational Power Savings Factor = 
$$\frac{\frac{5345 \times 245,760}{25}}{\frac{3600 \times 475,360}{279} + \frac{1200 \times 245,760}{70} + \frac{545 \times 245,760}{25}} = \frac{52,592,640}{15,705,343} \approx 3.35$$

The method proposed in this paper significantly reduces both the computational power and the time required for training. By employing a scale-up and scale-out approach, we achieved a computational power savings factor of approximately 3.35 and a time savings factor of approximately 4.12.

Additionally, if we start with a pre-trained smaller model, the computational power and time required for the preparation phase can be further reduced. This approach not only accelerates the training process but also lowers the overall computational costs.

In summary, the proposed training methodology offers substantial improvements in efficiency. The combined scale-up and scale-out approach, along with the potential use of pre-trained models, represents a significant advancement in the optimization of training large-scale models.

## 5 Conclusion and Future Work

We present AquilaMoE, a bilingual 8\*16B mixture of experts (MoE) language model developed using the EfficientScale training method. EfficientScale optimizes performance while significantly reducing data requirements through a two-stage approach: Scale-Up and Scale-Out. Our contributions are as follows: 1) An effective training methodology that achieves knowledge transfer and continuous pretraining with significantly reduced data and computational needs; 2) Innovative initialization strategies, such as Functional Progressive Initialization (FPI) and Approximate Knowledge Integration (AKI), which demonstrate substantial loss retention and reduction during continual pre-training; 3) Successful training of 16B and 8\*16B AquilaMoE models using these initialization strategies, enhancing performance and training efficiency. Future work involves exploring the scalability of larger MoE models, investigating cross-linguistic knowledge transfer, developing new optimization techniques to further reduce training time and costs, fine-tuning for specific application domains, and ensuring the robustness and generalization of MoE models across diverse datasets and real-world applications.

## Authorship

Language Foundation Model & Software Team, BAAI: Bo-Wen Zhang, Liangdong Wang, Jijie Li, Shuhao Gu, Mengdi Zhao, Xinya Wu, Guang Liu (Project lead) [5](#page-8-8)

Data Research Team, BAAI: Chengwei Wu, Hanyu Zhao, Li Du, Yiming Ju, Quanyue Ma

AI Framework Research and Development Team, BAAI: Yulong Ao (Infrastructure lead), Yingli Zhao, Songhe Zhu, Zhou Cao, Dong Liang, Yonghua Lin

School of Computer Science, Peking University: Ye Yuan[6](#page-8-9) , Ming Zhang

MetaX-Tech: Shunfei Wang, Yanxin Zhou, Min Ye, Xuekai Chen, Xinyang Yu, Xiangjun Huang, Jian Yang

# References

- <span id="page-8-0"></span>[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, 2017.
- <span id="page-8-1"></span>[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018.
- <span id="page-8-2"></span>[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *arXiv preprint arXiv:2005.14165*, 2020.
- <span id="page-8-3"></span>[4] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140):1–67, 2020.
- <span id="page-8-4"></span>[5] Dmitry Lepikhin, Yinhan Lee, Hao Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen, and Yonghui Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*, 2021.
- <span id="page-8-5"></span>[6] Tianqi Chen, Ian Goodfellow, and Jonathon Shlens. Net2net: Accelerating learning via knowledge transfer. In *International Conference on Learning Representations (ICLR)*, 2016.
- <span id="page-8-6"></span>[7] Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tieyan Liu. Efficient training of bert by progressively stacking. In *International conference on machine learning*, pages 2337–2346. PMLR, 2019.
- <span id="page-8-7"></span>[8] Cheng Chen, Yichun Yin, Lifeng Shang, Xin Jiang, Yujia Qin, Fengyu Wang, Zhi Wang, Xiao Chen, Zhiyuan Liu, and Qun Liu. bert2BERT: Towards reusable pretrained language models. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, Dublin, Ireland, May 2022. Association for Computational Linguistics.

<span id="page-8-8"></span><sup>5</sup>The correspinding author, contact <liuguang@baai.ac.cn>.

<span id="page-8-9"></span><sup>6</sup>Responsible for the full design and implementation of the Scale-Up strategy. Main work done during his internship at BAAI.

- <span id="page-9-0"></span>[9] Yu Pan, Ye Yuan, Yichun Yin, Jiaxin Shi, Zenglin Xu, Ming Zhang, Lifeng Shang, Xin Jiang, and Qun Liu. Preparing lessons for progressive training on language models. *arXiv preprint arXiv:2401.09192*, 2024.
- <span id="page-9-1"></span>[10] Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-of-experts from dense checkpoints. *arXiv preprint arXiv:2212.05055*, 2022.
- <span id="page-9-2"></span>[11] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. 2024. *URL https://doi. org/10.48550/arXiv*, 2404.
- <span id="page-9-3"></span>[12] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022.
- <span id="page-9-4"></span>[13] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. *Journal of Machine Learning Research*, 24(240):1–113, 2023.
- <span id="page-9-5"></span>[14] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. St-moe: Designing stable and transferable sparse expert models. *arXiv preprint arXiv:2202.08906*, 2022.