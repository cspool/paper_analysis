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

