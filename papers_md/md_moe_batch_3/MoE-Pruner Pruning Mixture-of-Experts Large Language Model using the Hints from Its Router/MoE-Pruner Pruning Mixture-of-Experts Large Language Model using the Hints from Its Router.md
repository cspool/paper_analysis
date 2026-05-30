# MOE-PRUNER: PRUNING MIXTURE-OF-EXPERTS LARGE LANGUAGE MODEL USING THE HINTS FROM ITS ROUTER

Yanyue Xie $^{1*}$  Zhi Zhang $^2$  Ding Zhou $^2$  Cong Xie $^2$  Ziang Song $^2$  Xin Liu $^2$  Yanzhi Wang $^1$  Xue Lin $^1$  An Xu $^2$ †  $^1$ Northeastern University  $^2$ ByteDance Inc. {xie.yany, yanz.wang, xue.lin}@northeastern.edu {zhangzhi.joshua, ding.zhou, cong.xie, ziang.song, liuxin.ai, an.xu}@bytedance.com

#### **ABSTRACT**

Mixture-of-Experts (MoE) architectures face challenges such as high memory consumption and redundancy in experts. Pruning MoE can reduce network weights while maintaining model performance. Motivated by the recent observation of emergent large magnitude features in Large Language Models (LLM) and MoE routing policy, we propose MoE-Pruner, a method that prunes weights with the smallest magnitudes multiplied by the corresponding input activations and router weights, on each output neuron. Our pruning method is one-shot, requiring no retraining or weight updates. We evaluate our method on Mixtral-8x7B and Mixtral-8x22B across multiple language benchmarks. Experimental results show that our pruning method significantly outperforms state-of-the-art LLM pruning methods. Furthermore, our pruned MoE models can benefit from a pretrained teacher model through expert-wise knowledge distillation, improving performance post-pruning. Experimental results demonstrate that the Mixtral-8x7B model with 50% sparsity maintains 99% of the performance of the original model after the expert-wise knowledge distillation.

## 1 Introduction

Scaling neural network models is one of the main drivers of better performance in deep learning. From BERT (Devlin et al., 2019) to GPT-3 (Brown et al., 2020) to Llama 3.1 405B (Dubey et al., 2024) in natural language processing, or from ResNet (He et al., 2016) to ViT (Dosovitskiy et al., 2021) in computer vision, breakthroughs in performance have been obtained from larger models, datasets, and computational resources for training (Kaplan et al., 2020). However, the cost of training state-of-the-art models grows exponentially. For instance, BERT-Large (345M parameters, proposed in 2018) requires an estimated  $5 \times 10^{20}$  FLOPs (Devlin et al., 2019) to train, GPT-3 (175B parameters, from 2020) requires  $3.14 \times 10^{23}$  FLOPs (Brown et al., 2020), while Llama 3.1 (405B, released in 2024) requires  $3.8 \times 10^{25}$  FLOPs (Dubey et al., 2024) to train. This exponential growth motivates researchers to seek more efficient and effective training approaches.

Mixture-of-Experts (MoE) architectures (Jacobs et al., 1991; Shazeer et al., 2017) have been proposed to reduce the computing cost while enabling efficient scaling of network capacity. It has been successfully employed to scale both vision (Ruiz et al., 2021; Shen et al., 2023) and language (Lepikhin et al., 2021; Fedus et al., 2022) models. In addition, these models provide other advantages, including sparsity that can mitigate catastrophic forgetting in continual learning and an inductive bias that can enhance performance in multitask learning. Overall, MoE has proven to be a promising strategy for scaling deep learning models across various domains.

However, several crucial limitations persist in MoE for expanding its capacity. First of all, the static parameters, particularly those required for constructing the MoE architecture, introduce substantial

<sup>\*</sup>Work done during an internship at ByteDance.

<sup>&</sup>lt;sup>†</sup>Corresponding author.

memory overheads and constraints for deployment. For example, Mixtral-8x7B [\(Jiang et al., 2024\)](#page-11-2) expert layers account for 96% of model parameters (45B out of 47B), which demands considerable memory and storage during inference. Moreover, MoE has a poor utilization of its experts. The conventional learning-based routing policy for MoE suffers from representation collapse issues since it encourages token embeddings to be clustered around expert centroids [\(Chi et al., 2022\)](#page-10-4) and results in redundant experts [\(Mittal et al., 2022;](#page-13-3) [Chen et al., 2022\)](#page-10-5).

One possible solution to address those drawbacks and fully unleash the power of MoE is consolidating information from insignificant experts, aiming to establish a more compact MoE without hurting performance. Another solution is pruning experts that yield the lowest token reconstruction loss. Nevertheless, naively combining existing model merging mechanisms or expert pruning leads to performance degradation in the MoE architectures. We raise the following pivotal questions for MoE LLM pruning: (i) How do we formulate and devise comprehensive pruning metrics that leverage existing methods? (ii) How do we find the optimal pruning metric tailored for MoE Large Language Models?

In this paper, we systematically explore MoE LLM pruning and target a high-quality compressed MoE model in downstream fine-tuning scenarios. Specifically, we first analyze the open-source MoE model's expert activation frequency and observe that different MoE expert initialization methods result in different expert activation frequencies and expert similarities. We leverage existing LLM pruning methods such as SparseGPT [\(Frantar & Alistarh, 2023b\)](#page-11-3) and Wanda [\(Sun et al., 2024\)](#page-13-4), and design a novel pruning metric that incorporates MoE router weights information to identify and remove unimportant weights in expert layers. An overview of MoE-Pruner is shown in Figure [1.](#page-1-0) Since the pruning process is one-shot and only requires a small set of calibration data, the MoE model suffers from performance degradation. To recover MoE model performance, we further propose an expert-wise knowledge distillation method that utilizes the pretrained model as a teacher model, facilitating the recovery of the pruned model's performance.

<span id="page-1-0"></span>![](_page_1_Picture_3.jpeg)

Figure 1: Overview of MoE-Pruner. For the MoE expert layer, the output is the weighted sum of the outputs from selected experts over inputs. <sup>G</sup><sup>i</sup> denoted the routing logits and <sup>G</sup>f<sup>i</sup> denotes the normalized router weight of each selected expert. Our pruning metric is the multiplication of weight magnitude and the norm of input activations by the router weights.

Our main contributions can be summarized as follows:

- We propose a novel framework, MoE-Pruner, that is efficient and effective for pruning MoE models with minimal performance degradation.
- We design an innovative expert-wise knowledge distillation method that leverages the pretrained MoE model as a teacher model to recover pruned MoE student model performance.
- Experimental results on Mixtral MoE models across nine zero-shot evaluation benchmarks demonstrate the effectiveness of our MoE-Pruner algorithm. MoE-Pruner achieves minimal performance drop even at 50% sparsity with only a small set of calibration data compared with existing pruning methods. The pruned model maintains 99% of the performance of the original model after the expert-wise knowledge distillation.

# 2 PRELIMINARIES

Mixture-of-Experts (MoE). Scaling model size increases learning capacity and enhances generalization [\(Kaplan et al., 2020;](#page-12-0) [Brown et al., 2020;](#page-9-0) [Hoffmann et al., 2022\)](#page-11-4). MoE [\(Jacobs et al., 1991;](#page-11-1) [Shazeer et al., 2017;](#page-13-0) [Lepikhin et al., 2021;](#page-12-1) [Fedus et al., 2022\)](#page-10-3) is an efficient approach that enables significantly more compute-efficient pretraining and inference. It replaces the feed-forward network (FFN) layers in Transformers [\(Vaswani et al., 2017\)](#page-13-5) with expert layers, where different experts are activated for different input tokens instead of utilizing the full network parameters. Sparse MoE architecture can dramatically scale the model with the same compute budget as a dense model.

Large Language Model Pruning. Magnitude pruning [\(Han et al., 2016\)](#page-11-5) is a standard approach to induce sparsity in neural networks. It removes individual weights with magnitudes below a certain threshold. However, magnitude pruning fails dramatically on LLMs even with relatively low levels of sparsity [\(Frantar & Alistarh, 2023b\)](#page-11-3). SparseGPT [\(Frantar & Alistarh, 2023b\)](#page-11-3) proposes a one-shot, post-training pruning method that prunes LLM weights and uses Hessian matrix and calibration data to update the remaining weights without any retraining. Wanda [\(Sun et al., 2024\)](#page-13-4) is a simple method that prunes LLM weights with the smallest magnitudes multiplied by the corresponding input activations without any additional weight update.

# 3 METHODOLOGY

## 3.1 THE MIXTURE-OF-EXPERTS ARCHITECTURE

Mixture-of-Experts (MoE) architecture. MoE architecture replaces the feed-forward networks (FFN) in Transformers with mixture-of-expert layers. A router or a gating network is trained to select a subset of experts for each input token based on its routing policy. Given n experts in a layer, the output of the expert layer is given by:

<span id="page-2-0"></span>
$$y = \sum_{i=0}^{n-1} Gate(x)_i \cdot E_i(x), \tag{1}$$

where the Gate(x)<sup>i</sup> is the router weights from the gating network assigned to the i-th expert, and Ei(x) is the output of i-th expert. The router weights can be formulated as softmax over the Top-K logits:

$$Gate(x) = Softmax(TopK(x \cdot W_g)),$$
 (2)

where W<sup>g</sup> is the weight of the router or gating network, and TopK(X)<sup>i</sup> = l<sup>i</sup> if i is in the top-K coordinates of logits l and TopK(X)<sup>i</sup> = −∞ otherwise.

Since current LLMs mostly adopt SwiGLU [\(Shazeer, 2020\)](#page-13-6) architecture for the FFN, and MoE LLM such as Mixtral-8x7B [\(Jiang et al., 2024\)](#page-11-2) uses a top-2 to select experts, we can derive the output of an expert layer as:

$$y = \sum_{i=0}^{n-1} \text{Softmax}(\text{Top2}(x \cdot W_g))_i \cdot \text{SwiGLU}_i(x).$$
 (3)

Some recent MoE LLMs, such as DeepSeekMoE [\(Dai et al., 2024\)](#page-10-6), adopt shared experts that are always activated, aiming at capturing and consolidating common knowledge across varying contexts.

MoE Expert Initialization. MoE expert initialization uses different strategies, which can be classified into two categories: sparse upcycling [\(Komatsuzaki et al., 2023\)](#page-12-2) and training from scratch. Some open-source MoE models such as Mixtral [\(Jiang et al., 2024\)](#page-11-2), Qwen1.5-MoE-A2.7B [\(Team,](#page-13-7) [2024\)](#page-13-7), and MiniCPM-MoE [\(Hu et al., 2024\)](#page-11-6) all employ the upcycling approach to reduce the total training costs. While some MoE models like DeepSeek-V2 [\(Liu et al., 2024\)](#page-12-3), OLMoE [\(Muennighoff](#page-13-8) [et al., 2024\)](#page-13-8), and Yuan2.0-M32 [\(Wu et al., 2024\)](#page-14-0) use the training from scratch approach to help expert diversification. We find that different MoE expert initialization methods result in different expert activation frequencies and expert similarities, which will impact the MoE pruning strategies. For instance, the MoE model initialized with upcycling can take advantage of the dense model and reduce training costs. The final MoE model exhibits higher expert similarity and more balanced expert activation frequency, which indicates that expert pruning will result in a performance drop, and weight pruning will be a better choice. MoE model trained from scratch might yield better performance as it avoids the limitations of starting with a group of identical experts, which can hinder diversification (Wei et al., 2024). It also shows imbalanced expert activation frequency, indicating that least-used expert pruning could help compress model size and not bring performance degradation.

**MoE Expert Activation Frequency.** We use a subset of the C4 (Raffel et al., 2020) dataset and collect the activation frequency of MoE experts. The expert activation frequency is task-agnostic since C4 pretraining datasets are comprehensive and not dominated by knowledge specific to any particular domain. Motivated by the load balancing loss (Shazeer et al., 2017; Lepikhin et al., 2021; Fedus et al., 2022), we propose to use the coefficient of variation of expert activation frequency in each layer to represent the load balancing score, where a lower score represents more balanced loads. Given n experts and l layers and a batch  $\mathcal B$  with T tokens, the load balancing score for one layer is:

$$s = \frac{\sigma}{\mu} = \frac{\sqrt{\frac{1}{n} \sum_{i=0}^{n-1} (f_i - \mu)^2}}{\mu},$$

$$\mu = \frac{1}{n} \sum_{i=0}^{n-1} f_i,$$
(4)

where  $f_i$  is the number of tokens dispatched to expert i,

$$f_i = \sum_{x \in \mathcal{B}} \mathbb{1}\{\operatorname{argmax} p(x) = i\}. \tag{5}$$

<span id="page-3-0"></span>We can derive the load balancing score by calculating the mean of scores across all *l* MoE layers, such that we can use this score to compare with various MoE models with different numbers of experts. Figure 2 shows the load balancing scores of Mixtral-8x7B (Jiang et al., 2024), Qwen-1.5-A2.7B (Team, 2024), DeepSeek-V2 and DeepSeek-V2-Lite (Liu et al., 2024), MiniCPM-MoE-8x2B (Hu et al., 2024), and OLMoE (Muennighoff et al., 2024).

![](_page_3_Figure_6.jpeg)

Figure 2: Load balancing score of MoE models. We collect the expert activation frequency of MoE models and calculate the load balancing score (lower is better). The circle area represents the model size. MoE model trained from scratch are marked with red, while MoE models that use upcycling are marked with blue. MoE models trained from scratch usually have more experts and imbalanced loads. MoE models initialized with upcycling tend to have more balanced loads and less number of experts. The only exception is Qwen-1.5-A2.7B, which is initialized with upcycling. But according to the report (Yang et al., 2024), its expert parameters are shuffled along the intermediate dimension to guarantee that each fine-grained expert exhibits unique characteristics and therefore exhibits more like trained from scratch MoE models.

#### 3.2 PRUNING METRIC

**Problem Formulation.** Post-training pruning for LLMs can be decomposed into layer-wise subproblems (Lu et al., 2022; Frantar & Alistarh, 2023b; Sun et al., 2024; Dong et al., 2024). Given a

sparsity ratio and a linear layer with weight W, the pruning algorithm tries to find a sparsity mask M that minimizes reconstruction loss:

$$\underset{\mathbf{M}}{\operatorname{argmin}} \|\mathbf{W}\mathbf{X} - (\mathbf{M} \odot \mathbf{W})\mathbf{X}\|. \tag{6}$$

Optimal Brain Damage (OBD) [\(LeCun et al., 1989\)](#page-12-5) first sets up a pioneering framework for neural network pruning. It uses second-order information without off-diagonal elements in the Hessian matrix for faster approximation. Optimal Brain Surgeon (OBS) [\(Hassibi et al., 1993\)](#page-11-7) develops upon OBD partly by taking into account the off-diagonal elements. SparseGPT [\(Frantar & Alistarh,](#page-11-3) [2023b\)](#page-11-3) revisits the OBS, computes the inverse Hessian only once, and reuses to update weight in the remaining rows that are also in the mask to mitigate reconstruction loss. The pruning metric in SparseGPT is:

$$S_{ij} = [|\mathbf{W}|^2 / \operatorname{diag}(\mathbf{H}^{-1})]_{ij}. \tag{7}$$

Wanda [\(Sun et al., 2024\)](#page-13-4) further simplifies the pruning metric to the following form without the need to compute the inverse of the Hessian matrix H:

$$S_{ij} = [|\mathbf{W}|^2 / \operatorname{diag}((\mathbf{X}^T \mathbf{X})^{-1})]_{ij} \approx [|\mathbf{W}|^2 / (\operatorname{diag}(\mathbf{X}^T \mathbf{X})^{-1})]_{ij} = (|\mathbf{W}_{ij}| \cdot ||\mathbf{X}_j||)^2.$$
(8)

When it comes to pruning MoE, the expert layers constitute the majority of model parameters. For example, the Mixtral-8x7B [\(Jiang et al., 2024\)](#page-11-2) has a total of 47B parameters where 1.3B belongs to attention modules and 45B is used for expert layers (2 out of 8 experts are activated, 12.5B active parameters during inference). Only a subset of experts are activated for different input tokens, so there is a large space of expert redundancy.

Router Tells It All. As shown in Equation [1,](#page-2-0) the router weights are assigned to each expert output. Motivated by the pruning metric in Wanda and the MoE routing policy, our approach, MoE-Pruner, prunes weights with the smallest magnitudes multiplied by the corresponding input activations and router weights, on each output neuron:

$$S = |\mathbf{W}_{ij}| \cdot ||\mathbf{X}_j \cdot \mathbf{Gate}_j||. \tag{9}$$

<span id="page-4-0"></span>Table 1: Comparison of different pruning methods including magnitude pruning, SparseGPT, Wanda, and MoE-Pruner.

| Method     | Weight update | Calibration data | Pruning metric S           |                     |
|------------|---------------|------------------|----------------------------|---------------------|
| Magnitude  | ✗             | ✗                | W                          | O(1)                |
| SparseGPT  | ✔             | ✔                | 2/diag(H−1<br>[ W <br>)]ij | 3<br>O(d<br>hidden) |
| Wanda      | ✗             | ✔                | Wij<br>  · ∥Xj∥            | 2<br>O(d<br>hidden) |
| MoE-Pruner | ✗             | ✔                | Wij<br>  · ∥Xj<br>· Gatej∥ | 2<br>O(d<br>hidden) |

Table [1](#page-4-0) summarizes pruning methods, including magnitude pruning, SparseGPT, Wanda, and MoE-Pruner and their corresponding pruning metric and complexity. Algorithm [1](#page-5-0) presents the unstructured sparsity version of our MoE-Pruner algorithm. Our method is simple and efficient for MoE models and does not require a sophisticated weight update procedure.

Structured N:M Sparsity. Structured N:M sparsity can leverage NVIDIA's sparse tensor cores to accelerate matrix multiplication. While MoE-Pruner so far has been developed for unstructured sparsity, it can be easily extended to structured N:M sparsity [\(Mishra et al., 2021\)](#page-13-10), where we compare weights using the same metric among every M consecutive weights, for all weights connected to an output.

Comparison Group. Generally, for a pruning method, each weight is first assigned an importance score, calculated by the pruning metric. These weights are then grouped into comparison groups where weights within each group are compared against one another, and weights with lower importance scores are pruned. Most previous pruning methods default to comparing weights locally within each layer or globally across the whole network. Our method compares weights using the comparison groups on each output neuron, which aligns with Wanda and can be easily extended to <span id="page-5-0"></span>**Algorithm 1** The MoE-Pruner algorithm. We prune each expert layer weight matrix **W** to p% sparsity.

```
1: Initialize: A MoE model \mathcal{M} with l MoE layers, where each MoE layer has n experts. Let \mathbf{X} \in \mathbb{R}^{b \times d_{\mathrm{col}}}
     and \mathbf{Gate} \in \mathbb{R}^{b \times n} denote the calibration samples and router weights respectively.
 2: for layer t = 1, \ldots, l do
           X', G \leftarrow forward(layer_t, X)

    ▶ get router weights

           for expert e = 1, \dots, n do
 5:
                 \mathbf{M} \leftarrow \mathbf{1}_{d_{\mathrm{row}} \times d_{\mathrm{col}}}
                                                                                                                                   ▶ binary pruning mask
                 \mathcal{S} \leftarrow |\mathbf{W}_{ij}| \cdot \|\mathbf{X}_j \cdot \mathbf{Gate}_j\|
                                                                                                                              > compute pruning metric
 6:
 7:
                 idx \leftarrow \text{sort}(\mathcal{S}, dim = 1)
                                                                                                        > prund weights indices based on metric
                 \mathbf{M} \leftarrow \mathtt{scatter}(0, idx_{:,d_{\mathtt{col}}*p\%})
 8:
 9:
                 \mathbf{W} \leftarrow \mathbf{M} \odot \mathbf{W}
                                                                                                                               > set pruned weights to 0
10:
           end for
           \mathbf{X} \leftarrow \mathbf{X}'
11:
12: end for
13: Return: A pruned MoE model \mathcal{M}'.
```

N:M semi-structured sparsity. Moreover, our method is not limited to pruning individual weights but can also group structured weights into a unit and compare those units among a larger comparison group, such that we can extend our method to structured sparsity.

#### 3.3 EXPERT-WISE KNOWLEDGE DISTILLATION

**Expert-Wise Knowledge Distillation.** MoE models can preserve most of their capacity after pruning but still suffer from performance degradation. To recover MoE LLM performance, we fine-tune the model by leveraging the unpruned pretrained model as a teacher model in an expert-wise knowledge distillation (KD) manner. The pretrained model is a natural teacher model for the pruned model since they share exactly the same number of layers, experts, and dimensions (Kurtic et al., 2023). The loss function for expert-wise knowledge distillation is formulated as follows:

$$\mathcal{L}_{KD} = \mathcal{L}_{CE} + \lambda \times \mathcal{L}_{expert} = \mathcal{L}_{CE} + \lambda \times \sum_{j=0}^{l-1} \sum_{i=0}^{n-1} \text{MSE}(E_{it}^j, E_{is}^j), \tag{10}$$

where  $\mathcal{L}_{CE}$  is the cross entropy loss, MSE is the mean squared error calculated as MSE $(X,Y) = \frac{1}{N} \sum_{i=0}^{N-1} (x_i - y_i)^2$  for N-dimensional vectors X and Y.  $\lambda$  is a weighting coefficient and initialized based on the strength of cross entropy loss and expert-wise knowledge distillation loss:  $\frac{\mathcal{L}_{CE}}{\mathcal{L}_{expert}}$ . We sum up all the differences between teacher experts and student experts. Figure 3 illustrates the expert-wise knowledge distillation for pruned models. The corresponding expert in the pretrained teacher model will be used to distill the expert in the pruned student model.

#### 4 EXPERIMENTS

Models, Datasets, and Evaluation. We conduct pruning experiments on widely adopted open-source MoE models: the base and instruct version of Mixtral-8x7B and Mixtral-8x22B (Jiang et al., 2024). We use samples from the pretraining dataset C4 (Raffel et al., 2020) as calibration data for one-shot pruning since pretraining datasets are often more comprehensive and not dominated by knowledge specific to any particular domain. We use the exact same 128 sequences of calibration data for all one-shot pruning experiments to control this variable factor. We evaluate the perplexity on the WikiText (Merity et al., 2017) validation set. Our expert-wise knowledge distillation method uses a subset of the C4 (Raffel et al., 2020) as the training set. We measure the performance of pruned models on zero-shot tasks and language modeling. For zero-shot evaluation, we use nine popular tasks from EleutherAI LM Harness (Gao et al., 2023). The nine evaluated zero-shot tasks are: ARC-easy, ARC-challenge (Clark et al., 2018), Boolq (Clark et al., 2019), HellaSwag (Zellers et al., 2019), MMLU (Hendrycks et al., 2021), OpenBookQA (OBQA) (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), RTE (Wang et al., 2018), and WinoGrande (Sakaguchi et al., 2021).

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

Figure 3: Expert-wise knowledge distillation for the pruned MoE model using the pretrained MoE model as the teacher to recover the performance of the pruned model.

Baselines and Experiments Setup. We compare MoE-Pruner with prior pruning approaches, including SparseGPT [\(Frantar & Alistarh, 2023b\)](#page-11-3) and Wanda [\(Sun et al., 2024\)](#page-13-4). Similarly, our pruning algorithm is implemented in a layer-wise reconstruction manner. All pruning experiments are conducted on a single NVIDIA H100-80GB GPU. The fine-tuning experiments use the pruned model as a starting point and perform full-parameter fine-tuning to preserve the sparsity mask. We implement the expert-wise knowledge distillation method in Llama-Factory [\(Zheng et al., 2024\)](#page-14-4) and conduct experiments on 2 servers, each with 8 NVIDIA H100-80GB GPUs. We fine-tune the pruned student model for three epochs, using a learning rate of 2e-5 with the cosine learning rate scheduler.

# 4.1 ONE-SHOT PRUNING

Table [2](#page-7-0) shows the one-shot pruning model perplexity on WikiText with 50% sparsity. There is a clear difference between MoE-Pruner and other pruning methods, including SparseGPT [\(Frantar &](#page-11-3) [Alistarh, 2023b\)](#page-11-3) and Wanda [\(Sun et al., 2024\)](#page-13-4). For Mixtral-8x7B [\(Jiang et al., 2024\)](#page-11-2) models, MoE-Pruner achieves 0.22-0.31 better perplexity over SparseGPT and Wanda. This improvement expands when the MoE model scales to the Mixtral-8x22B model. For the larger Mixtral-8x22B model, MoE-Pruner achieves 0.55 better perplexity over SparseGPT and 0.31-0.34 better perplexity over Wanda. MoE-Pruner further expands the improvement to 1.21 better perplexity over SparseGPT and 1.10 better perplexity over Wanda when we prune the MoE models with the 2:4 semi-structured sparsity.

Table [3](#page-7-1) shows the average zero-shot accuracies on nine zero-shot tasks of the pruned Mixtral-8x7B MoE models with 50% unstructured sparsity. The average performance of pretrained models, SparseGPT, Wanda, and our pruned models are 69.16, 66.27, 65.90, and 67.23, respectively. MoE-Pruner outperforms the state-of-the-art pruning approaches, SparseGPT and Wanda, by a large margin. Given that no fine-tuning takes place at this time, there is a noticeable gap between the sparse pruned MoE model and the original pretrained MoE model.

### 4.2 EXPERT-WISE KNOWLEDGE DISTILLATION PERFORMANCE

The gap between the pruned MoE model and the pretrained MoE model can be largely mitigated via expert-wise knowledge distillation. We only need 1000 training samples from C4 [\(Raffel et al.,](#page-13-9) [2020\)](#page-13-9), and training can be done in 1 hour. Table [4](#page-7-2) shows the average zero-shot accuracy of the pruned and fine-tuned Mixtral-8x7B MoE models with 50% unstructured sparsity. The fine-tuned model could achieve a 68.40 average performance on nine zero-shot tasks. The performance is very close to the pretrained Mixtral-8x7B MoE model, which demonstrates a 69.16 average performance.

<span id="page-7-0"></span>Table 2: Perplexity against other one-shot pruning methods, including SparseGPT and Wanda with 50% sparsity.

| Model                  | Method            | Sparsity | WikiText Perplexity ↓ |  |  |
|------------------------|-------------------|----------|-----------------------|--|--|
|                        | Pretrained        | 0%       | 3.84                  |  |  |
|                        | SparseGPT         | 50%      | 4.99                  |  |  |
| Mixtral-8x7B           | Wanda             | 50%      | 4.97                  |  |  |
|                        | MoE-Pruner (Ours) | 50%      | 4.68                  |  |  |
|                        | Pretrained        | 0%       | 3.84                  |  |  |
| Mixtral-8x7B           | SparseGPT         | 2:4      | 7.09                  |  |  |
|                        | Wanda             | 2:4      | 6.98                  |  |  |
|                        | MoE-Pruner (Ours) | 2:4      | 5.88                  |  |  |
|                        | Pretrained        | 0%       | 4.14                  |  |  |
|                        | SparseGPT         | 50%      | 5.20                  |  |  |
| Mixtral-8x7B-Instruct  | Wanda             | 50%      | 5.16                  |  |  |
|                        | MoE-Pruner (Ours) | 50%      | 4.94                  |  |  |
|                        | Pretrained        | 0%       | 4.14                  |  |  |
|                        | SparseGPT         | 2:4      | 7.19                  |  |  |
| Mixtral-8x7B-Instruct  | Wanda             | 2:4      | 6.92                  |  |  |
|                        | MoE-Pruner (Ours) | 2:4      | 6.11                  |  |  |
|                        | Pretrained        | 0%       | 2.83                  |  |  |
|                        | SparseGPT         | 50%      | 4.19                  |  |  |
| Mixtral-8x22B          | Wanda             | 50%      | 3.97                  |  |  |
|                        | MoE-Pruner (Ours) | 50%      | 3.64                  |  |  |
|                        | Pretrained        | 0%       | 2.89                  |  |  |
|                        | SparseGPT         | 50%      | 4.27                  |  |  |
| Mixtral-8x22B-Instruct | Wanda             | 50%      | 4.06                  |  |  |
|                        | MoE-Pruner (Ours) | 50%      | 3.72                  |  |  |

<span id="page-7-1"></span>Table 3: Average zero-shot performance on 9 evaluation tasks of pruned models using SparseGPT, Wanda, and MoE-Pruner.

| Model            | Method     | ARC-c | ARC-e | Boolq | HellaSwag | MMLU  | OBQA | PIQA  | RTE   | WinoGrande | Average |
|------------------|------------|-------|-------|-------|-----------|-------|------|-------|-------|------------|---------|
|                  | Pretrained | 56.91 | 84.47 | 85.29 | 64.78     | 67.03 | 35.0 | 82.43 | 70.4  | 76.16      | 69.16   |
| Mixtral<br>-8x7B | SparseGPT  | 50.43 | 80.68 | 84.62 | 60.20     | 61.79 | 32.8 | 81.12 | 68.59 | 76.16      | 66.27   |
|                  | Wanda      | 51.02 | 80.89 | 85.08 | 60.45     | 62.73 | 32.6 | 80.90 | 64.64 | 74.82      | 65.90   |
|                  | MoE-Pruner | 53.33 | 81.86 | 86.02 | 62.29     | 64.76 | 33.6 | 81.61 | 66.06 | 75.53      | 67.23   |

<span id="page-7-2"></span>Table 4: Average zero-shot performance after pruning and expert-wise knowledge distillation.

| Model            | Method        | ARC-c | ARC-e | Boolq | HellaSwag | MMLU  | OBQA | PIQA  | RTE   | WinoGrande | Average |
|------------------|---------------|-------|-------|-------|-----------|-------|------|-------|-------|------------|---------|
| Mixtral<br>-8x7B | Pretrained    | 56.91 | 84.47 | 85.29 | 64.78     | 67.03 | 35.0 | 82.43 | 70.4  | 76.16      | 69.16   |
|                  | MoE-Pruned    | 53.33 | 81.86 | 86.02 | 62.29     | 64.76 | 33.6 | 81.61 | 66.06 | 75.53      | 67.23   |
|                  | MoE-Distilled | 54.35 | 81.19 | 85.26 | 68.77     | 65.59 | 36.0 | 82.48 | 68.23 | 75.72      | 68.40   |

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

![](_page_8_Figure_1.jpeg)

Figure 4: Perplexity with different number of calibration samples at 50% sparsity.

Figure 5: Perplexity over different pruning ratios with 128 calibration samples.

#### 4.3 ABLATION STUDIES

Ablation on Different Number of Calibration Samples. We change the number of calibration samples by selecting different sample sizes ranging from 2 to 256. Results are summarized in Figure 4. We see a clear difference in trend as the number of calibration samples changes. MoE-Pruner is much more robust than SparseGPT when there are few calibration samples and performs the same trend but better perplexity over Wanda. Notably, even with just two calibration samples, pruned networks obtained by MoE-Pruner have a perplexity of just 4.95. This may be because input norm statistics could be much easier to estimate than the full inverse Hessian of the local layer-wise reconstruction problem.

**Ablation on Different Sparsity Ratio.** We also change the pruning ratio using the same 128 calibration samples. Figure 5 shows that at lower pruning ratios, such as 10% to 40%, all pruning methods result in almost the same perplexity. When the pruning ratio increases, the Wanda pruned model perplexity changes dramatically and fails at 70%. MoE-Pruner shows better and more stable pruning results than SparseGPT and Wanda, especially at higher pruning ratios. This demonstrates that router weights preserve important information when selecting experts and provide a clear hint for pruning unimportant weights.

#### 5 RELATED WORKS

**Pruning and Sparsity.** Pruning (LeCun et al., 1989; Hassibi et al., 1993; Han et al., 2015) is an important approach for compressing neural networks through eliminating weights (Han et al., 2016) or activations (Rao et al., 2021), yielding sparse networks. It can be mainly classified into two categories based on the granularity: *unstructured* and *structured* pruning.

Unstructured pruning such as magnitude pruning (Han et al., 2015; 2016) removes individual weights to introduce sparsity while preserving accuracy even at high sparsity. Existing methods either require retraining or fine-tuning the pruned models (Liu et al., 2019) or the whole iterative retraining process (Frankle & Carbin, 2019). However, in the era of LLMs, these methods fail as retraining LLMs demands substantial computational resources. SparseGPT (Frantar & Alistarh, 2023b) and Wanda (Sun et al., 2024) propose efficient post-training pruning method that prunes LLM weights in a layer-wise manner without retraining the model.

Structured pruning eliminates weights as a group, such as channel pruning (He et al., 2017), kernel pruning (Zhong et al., 2022), attention head pruning (Wang et al., 2021), token pruning (Rao et al., 2021), and layer pruning (Elhoushi et al., 2024). Unlike unstructured pruning, it leads to more hardware-friendly, dense blocks of computation, which facilitates acceleration on modern hardware platforms. Some methods explore structured pruning based on sparsity on the structural components of LLMs, such as attention heads (Wang et al., 2021) and FFN channels (Ma et al., 2023). Muralid-haran et al. (2024) uses both structured pruning and knowledge distillation to compress LLM models and shows improvement over models trained from scratch. Due to the constraint of removing regular components, structured pruning usually has low sparsity ratios and high accuracy loss. NVIDIA

proposes N:M semi-structured sparsity [\(Mishra et al., 2021\)](#page-13-10), which can preserve model performance by retraining and leverage GPU tensor core acceleration.

Pruning for MoE Models. Most of the works for MoE pruning focus on structured expert pruning. [Chen et al.](#page-10-5) [\(2022\)](#page-10-5) and [Koishekenov et al.](#page-12-11) [\(2023\)](#page-12-11) prune experts based on their utilization to save memory. However, this usually leads to degraded performance. [Lu et al.](#page-12-12) [\(2024\)](#page-12-12) enumerates expert combinations based on the required expert number and uses calibration data to find a set of remaining experts that has the minimum reconstruction loss. [Chowdhury et al.](#page-10-12) [\(2024\)](#page-10-12) prunes experts based on the change in the router's norm and proves that the generalization accuracy can be preserved. However, expert pruning sometimes removes experts with certain knowledge and results in the loss of model performance. Therefore, [Li et al.](#page-12-13) [\(2024\)](#page-12-13) and [Zhang et al.](#page-14-7) [\(2024\)](#page-14-7) both leverage expert merging techniques to compress the expert layer while also preserving expert knowledge. [He et al.](#page-11-12) [\(2024\)](#page-11-12) proposes a unified framework to compress MoE models. The framework consists of two perspectives: (i) expert slimming that compresses individual experts by weight pruning and quantization, and (ii) expert trimming that removes whole structured modules by layer drop and block drop.

Efficiency for MoE and Existing Solutions. MoE models require huge memory to host expert layers, while many experts have low utilization during inference. To address this, [Gao et al.](#page-11-13) [\(2022\)](#page-11-13) uses a tensor decomposition method to share the central tensor's parameters across experts and keep different auxiliary tensors for each expert. MoQE [\(Kim et al., 2023\)](#page-12-14) and QMoE [\(Frantar & Alistarh,](#page-11-14) [2023a\)](#page-11-14) both study extreme low-bit quantization for compressing MoE model size. Moreover, some works employ knowledge distillation [\(Fedus et al., 2022;](#page-10-3) [Artetxe et al., 2021\)](#page-9-2) to create either a smaller dense model or a MoE model with fewer layers. However, they also overlook the existing redundancy within MoE expert layers. [Yadav et al.](#page-14-8) [\(2023\)](#page-14-8) shows that experts can be compressed to a huge degree without any performance loss.

# 6 CONCLUSION

We propose a simple and effective pruning method for MoE models, MoE-Pruner. We prune weights with the smallest magnitudes multiplied by the corresponding input activations and router weights, on each output neuron. Our pruning method is one-shot and fast, without the need for any retraining or weight update procedures. Pruning MoE LLM with high sparsity will incur performance degradation, so we also propose a fine-tuning method that leverages the unpruned pretrained MoE model as a teacher to guide the pruned student model through expert-wise knowledge distillation. The fine-tuned MoE models could maintain 99% of the performance of the original model after the expert-wise knowledge distillation, using only a small set of training data and low GPU hours. In the future, MoE-Pruner could also be extended to structured pruning of MoE LLMs, such as channel pruning and expert pruning, for better hardware acceleration.

# REFERENCES

<span id="page-9-2"></span>Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, et al. Efficient large scale language modeling with mixtures of experts. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, 2021.

<span id="page-9-1"></span>Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, pp. 7432–7439, 2020.

<span id="page-9-0"></span>Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. *arXiv preprint arXiv:2005.14165*, 2020.

- <span id="page-10-5"></span>Tianyu Chen, Shaohan Huang, Yuan Xie, Binxing Jiao, Daxin Jiang, Haoyi Zhou, Jianxin Li, and Furu Wei. Task-specific expert pruning for sparse mixture-of-experts. *arXiv preprint arXiv:2206.00277*, 2022.
- <span id="page-10-4"></span>Zewen Chi, Li Dong, Shaohan Huang, Damai Dai, Shuming Ma, Barun Patra, Saksham Singhal, Payal Bajaj, Xia Song, Xian-Ling Mao, Heyan Huang, and Furu Wei. On the representation collapse of sparse mixture of experts. In *Advances in Neural Information Processing Systems*, 2022. URL <https://openreview.net/forum?id=mWaYC6CZf5>.
- <span id="page-10-12"></span>Mohammed Nowaz Rabbani Chowdhury, Meng Wang, Kaoutar El Maghraoui, Naigang Wang, Pin-Yu Chen, and Christopher Carothers. A provably effective method for pruning experts in finetuned sparse mixture-of-experts. In *International Conference on Machine Learning*, 2024. URL <https://openreview.net/forum?id=1oU4FKpVx5>.
- <span id="page-10-9"></span>Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, 2019.
- <span id="page-10-8"></span>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*, 2018.
- <span id="page-10-6"></span>Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, Zhenda Xie, Y.K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. Deepseekmoe: Towards ultimate expert specialization in mixture-ofexperts language models. *arXiv preprint arXiv:2401.06066*, 2024.
- <span id="page-10-0"></span>Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 4171–4186, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL <https://aclanthology.org/N19-1423>.
- <span id="page-10-7"></span>Peijie Dong, Lujun Li, Zhenheng Tang, Xiang Liu, Xinglin Pan, Qiang Wang, and Xiaowen Chu. Pruner-zero: Evolving symbolic pruning metric from scratch for large language models. In *Proceedings of the 41st International Conference on Machine Learning*, 2024.
- <span id="page-10-2"></span>Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations*, 2021. URL [https:](https://openreview.net/forum?id=YicbFdNTTy) [//openreview.net/forum?id=YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy).
- <span id="page-10-1"></span>Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.
- <span id="page-10-11"></span>Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed A Aly, Beidi Chen, and Carole-Jean Wu. Layer skip: Enabling early exit inference and self-speculative decoding. *arXiv preprint arXiv:2404.16710*, 2024.
- <span id="page-10-3"></span>William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022.
- <span id="page-10-10"></span>Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International Conference on Learning Representations*, 2019. URL [https://](https://openreview.net/forum?id=rJl-b3RcF7) [openreview.net/forum?id=rJl-b3RcF7](https://openreview.net/forum?id=rJl-b3RcF7).

- <span id="page-11-14"></span>Elias Frantar and Dan Alistarh. Qmoe: Practical sub-1-bit compression of trillion-parameter models. *arXiv preprint arXiv:2310.16795*, 2023a.
- <span id="page-11-3"></span>Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot. In *International Conference on Machine Learning*, pp. 10323–10337. PMLR, 2023b.
- <span id="page-11-8"></span>Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL [https://zenodo.org/records/](https://zenodo.org/records/10256836) [10256836](https://zenodo.org/records/10256836).
- <span id="page-11-13"></span>Ze-Feng Gao, Peiyu Liu, Wayne Xin Zhao, Zhong-Yi Lu, and Ji-Rong Wen. Parameter-efficient mixture-of-experts architecture for pre-trained language models. In *Proceedings of the 29th International Conference on Computational Linguistics*, 2022.
- <span id="page-11-10"></span>Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In *Advances in neural information processing systems*, 2015.
- <span id="page-11-5"></span>Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In *International Conference on Learning Representations*, 2016.
- <span id="page-11-7"></span>Babak Hassibi, David G Stork, and Gregory J Wolff. Optimal brain surgeon and general network pruning. In *IEEE international conference on neural networks*, 1993.
- <span id="page-11-0"></span>Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.
- <span id="page-11-12"></span>Shwai He, Daize Dong, Liang Ding, and Ang Li. Demystifying the compression of mixture-ofexperts through a unified framework. *arXiv preprint arXiv:2406.02500*, 2024.
- <span id="page-11-11"></span>Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In *Proceedings of the IEEE international conference on computer vision*, pp. 1389–1397, 2017.
- <span id="page-11-9"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In *International Conference on Learning Representations*, 2021. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=d7KBjmI3GmQ) [d7KBjmI3GmQ](https://openreview.net/forum?id=d7KBjmI3GmQ).
- <span id="page-11-4"></span>Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.
- <span id="page-11-6"></span>Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. *arXiv preprint arXiv:2404.06395*, 2024.
- <span id="page-11-1"></span>Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. *Neural computation*, 3(1):79–87, 1991.
- <span id="page-11-2"></span>Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lelio Renard Lavaud, Lucile Saulnier, Marie-Anne ´ Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Theophile Gervet, Thibaut Lavril, Thomas Wang, Timoth ´ ee Lacroix, and William El Sayed. Mix- ´ tral of experts. *arXiv preprint arXiv:2401.04088*, 2024.

- <span id="page-12-0"></span>Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.
- <span id="page-12-14"></span>Young Jin Kim, Raffy Fahim, and Hany Hassan Awadalla. Mixture of quantized experts (moqe): Complementary effect of low-bit quantization and robustness. *arXiv preprint arXiv:2310.02410*, 2023.
- <span id="page-12-11"></span>Yeskendir Koishekenov, Alexandre Berard, and Vassilina Nikoulina. Memory-efficient nllb-200: Language-specific expert pruning of a massively multilingual machine translation model. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2023. URL <https://aclanthology.org/2023.acl-long.198>.
- <span id="page-12-2"></span>Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-of-experts from dense checkpoints. In *International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=T5nUQDrM4u>.
- <span id="page-12-6"></span>Eldar Kurtic, Denis Kuznedelev, Elias Frantar, Michael Goin, and Dan Alistarh. Sparse finetuning for inference acceleration of large language models. *arXiv preprint arXiv:2310.06927*, 2023.
- <span id="page-12-5"></span>Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. In *Advances in Neural Information Processing Systems*, 1989.
- <span id="page-12-1"></span>Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. {GS}hard: Scaling giant models with conditional computation and automatic sharding. In *International Conference on Learning Representations*, 2021. URL <https://openreview.net/forum?id=qrwe7XHTmYb>.
- <span id="page-12-13"></span>Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, and Tianlong Chen. Merge, then compress: Demystify efficient SMoe with hints from its routing policy. In *International Conference on Learning Representations*, 2024. URL [https://openreview.](https://openreview.net/forum?id=eFWG9Cy3WK) [net/forum?id=eFWG9Cy3WK](https://openreview.net/forum?id=eFWG9Cy3WK).
- <span id="page-12-3"></span>Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixtureof-experts language model. *arXiv preprint arXiv:2405.04434*, 2024.
- <span id="page-12-9"></span>Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. Rethinking the value of network pruning. In *International Conference on Learning Representations*, 2019. URL [https:](https://openreview.net/forum?id=rJlnB3C5Ym) [//openreview.net/forum?id=rJlnB3C5Ym](https://openreview.net/forum?id=rJlnB3C5Ym).
- <span id="page-12-4"></span>Miao Lu, Xiaolong Luo, Tianlong Chen, Wuyang Chen, Dong Liu, and Zhangyang Wang. Learning pruning-friendly networks via frank-wolfe: One-shot, any-sparsity, and no retraining. In *International Conference on Learning Representations*, 2022. URL [https://openreview.net/](https://openreview.net/forum?id=O1DEtITim__) [forum?id=O1DEtITim\\_\\_](https://openreview.net/forum?id=O1DEtITim__).
- <span id="page-12-12"></span>Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 6159–6172, 2024.
- <span id="page-12-10"></span>Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. In *Advances in Neural Information Processing Systems*, 2023.
- <span id="page-12-7"></span>Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In *International Conference on Learning Representations*, 2017. URL [https:](https://openreview.net/forum?id=Byj72udxe) [//openreview.net/forum?id=Byj72udxe](https://openreview.net/forum?id=Byj72udxe).
- <span id="page-12-8"></span>Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, 2018.

- <span id="page-13-10"></span>Asit Mishra, Jorge Albericio Latorre, Jeff Pool, Darko Stosic, Dusan Stosic, Ganesh Venkatesh, Chong Yu, and Paulius Micikevicius. Accelerating sparse deep neural networks. *arXiv preprint arXiv:2104.08378*, 2021.
- <span id="page-13-3"></span>Sarthak Mittal, Yoshua Bengio, and Guillaume Lajoie. Is a modular architecture enough? In *Advances in Neural Information Processing Systems*, 2022. URL [https://openreview.](https://openreview.net/forum?id=3-3XMModtrx) [net/forum?id=3-3XMModtrx](https://openreview.net/forum?id=3-3XMModtrx).
- <span id="page-13-8"></span>Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. Olmoe: Open mixture-of-experts language models. *arXiv preprint arXiv:2409.02060*, 2024.
- <span id="page-13-14"></span>Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. Compact language models via pruning and knowledge distillation. In *Advances in Neural Information Processing Systems*, 2024.
- <span id="page-13-9"></span>Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-totext transformer. *Journal of Machine Learning Research*, 21(140):1–67, 2020. URL [http:](http://jmlr.org/papers/v21/20-074.html) [//jmlr.org/papers/v21/20-074.html](http://jmlr.org/papers/v21/20-074.html).
- <span id="page-13-13"></span>Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. In *Advances in neural information processing systems*, 2021.
- <span id="page-13-1"></span>Carlos Riquelme Ruiz, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andre Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture ´ of experts. In *Advances in Neural Information Processing Systems*, 2021. URL [https:](https://openreview.net/forum?id=NGPmH3vbAA_) [//openreview.net/forum?id=NGPmH3vbAA\\_](https://openreview.net/forum?id=NGPmH3vbAA_).
- <span id="page-13-12"></span>Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021.
- <span id="page-13-6"></span>Noam Shazeer. Glu variants improve transformer. *arXiv preprint arXiv:2002.05202*, 2020.
- <span id="page-13-0"></span>Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In *International Conference on Learning Representations*, 2017. URL [https://openreview.](https://openreview.net/forum?id=B1ckMDqlg) [net/forum?id=B1ckMDqlg](https://openreview.net/forum?id=B1ckMDqlg).
- <span id="page-13-2"></span>Sheng Shen, Zhewei Yao, Chunyuan Li, Trevor Darrell, Kurt Keutzer, and Yuxiong He. Scaling vision-language models with sparse mixture of experts. In *The 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. URL [https://openreview.net/](https://openreview.net/forum?id=IpJ5rAFLv7) [forum?id=IpJ5rAFLv7](https://openreview.net/forum?id=IpJ5rAFLv7).
- <span id="page-13-4"></span>Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning approach for large language models. In *International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=PxoFut3dWW>.
- <span id="page-13-7"></span>Qwen Team. Qwen1.5-moe: Matching 7b model performance with 1/3 activated parameters", February 2024. URL <https://qwenlm.github.io/blog/qwen-moe/>.
- <span id="page-13-5"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, 2017.
- <span id="page-13-11"></span>Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: a multi-task benchmark and analysis platform for natural language understanding. In *Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP*, 2018.

- <span id="page-14-6"></span>Hanrui Wang, Zhekai Zhang, and Song Han. Spatten: Efficient sparse attention architecture with cascade token and head pruning. In *International Symposium on High-Performance Computer Architecture (HPCA)*, pp. 97–110. IEEE, 2021.
- <span id="page-14-1"></span>Tianwen Wei, Bo Zhu, Liang Zhao, Cheng Cheng, Biye Li, Weiwei Lu, Peng Cheng, Jianhao Zhang, ¨ Xiaoyu Zhang, Liang Zeng, et al. Skywork-moe: A deep dive into training techniques for mixtureof-experts language models. *arXiv preprint arXiv:2406.06563*, 2024.
- <span id="page-14-0"></span>Shaohua Wu, Jiangang Luo, Xi Chen, Lingjun Li, Xudong Zhao, Tong Yu, Chao Wang, Yue Wang, Fei Wang, Weixu Qiao, et al. Yuan 2.0-m32: Mixture of experts with attention router. *arXiv preprint arXiv:2405.17976*, 2024.
- <span id="page-14-8"></span>Prateek Yadav, Leshem Choshen, Colin Raffel, and Mohit Bansal. Compeft: Compression for communicating parameter efficient updates via sparsification and quantization. *arXiv preprint arXiv:2311.13171*, 2023.
- <span id="page-14-2"></span>An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. *arXiv preprint arXiv:2407.10671*, 2024.
- <span id="page-14-3"></span>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 2019.
- <span id="page-14-7"></span>Zeliang Zhang, Xiaodong Liu, Hao Cheng, Chenliang Xu, and Jianfeng Gao. Diversifying the expert knowledge for task-agnostic pruning in sparse mixture-of-experts. *arXiv preprint arXiv:2407.09590*, 2024.
- <span id="page-14-4"></span>Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)*, Bangkok, Thailand, 2024. Association for Computational Linguistics. URL <http://arxiv.org/abs/2403.13372>.
- <span id="page-14-5"></span>Shaochen Zhong, Guanqun Zhang, Ningjia Huang, and Shuai Xu. Revisit kernel pruning with lottery regulated grouped convolutions. In *International Conference on Learning Representations*, 2022. URL <https://openreview.net/forum?id=LdEhiMG9WLO>.

# **APPENDIX**

# A MOE EXPERT ACTIVATION FREQUENCY RESULTS

## A.1 MIXTRAL-8X7B

Experts Activation Frequency of Mixtral-8x7B on C4 Datasets

![](_page_15_Figure_4.jpeg)

Figure 6: Mixtral-8x7B expert activation frequency on C4 datasets.

# A.2 QWEN-1.5-A2.7B

![](_page_16_Figure_1.jpeg)

Figure 7: Qwen-1.5-A2.7B expert activation frequency on C4 datasets.

# B OPEN-SOURCE MOE MODELS

Table 5: Open-Source MoE Models List (Released after Jan. 2024).

| Name                      | Active<br>Parameters | Total<br>Parameters | # Experts     | Routing<br>Policy | Initialized<br>Method | MMLU* |
|---------------------------|----------------------|---------------------|---------------|-------------------|-----------------------|-------|
| OLMoE                     | 1B                   | 7B                  | 64            | top-8             | train from scratch    | 54.1  |
| MiniCPM-MoE-8x2B          | 4B                   | 13.6B               | 8             | top-2             | upcycling             | 58.9  |
| Qwen1.5-MoE-A2.7B         | 2.7B                 | 14.3B               | 4(shared)+60  | 4+top-4           | upcycling             | 62.5  |
| Deepseek-V2-Lite          | 2.4B                 | 16B                 | 2(shared)+64  | 2+top-6           | train from scratch    | 58.3  |
| Yuan2.0-M32               | 32 3.7B 40B          |                     | 32            | top-2             | train from scratch    | 72.2  |
| GRIN-MoE                  | 6.6B                 | 41.9B               | 16            | top-2             | upcycling             | 79.4  |
| Mixtral-8x7B              | 12.5B                | 47B                 | 8             | top-2             | upcycling             | 70.4  |
| Jamba                     | 12B                  | 52B                 | 16            | top-2             | unknown               | 67.4  |
| Qwen2-57B-A14B            | 14B 57.4B            |                     | 8(shared)+64  | 8+top-8           | upcycling             | 76.5  |
| DBRX                      | 36B                  | 132B                | 16            | top-4             | unknown               | 73.7  |
| Mixtral-8x22B             | 8x22B 39B 141B       |                     | 8             | top-2             | upcycling             | 77.8  |
| Skywork-MoE               | MoE 22B 146B         |                     | 16            | top-2             | upcycling             | 77.4  |
| Deepseek-V2               | 21B                  | 236B                | 2(shared)+160 | 2+top-6           | train from scratch    | 78.5  |
| grok-1                    | 80B                  | 314B                | 8             | top-2             | unknown               | 73.0  |
| Snowflake Arctic 17B 480B |                      | 480B                | 128           | top-2             | unknown               | 67.3  |

<sup>\*</sup>Note: This table presents a subset of open-source MoE models and is not exhaustive. The list is sorted by total parameters. MMLU scores are extracted from original papers or reports and may not reflect model real performance.