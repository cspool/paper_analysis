# Abstract

Training large-scale Mixture-of-Experts (MoE) models typically requires high-memory, highbandwidth GPUs (e.g., A100), and their high cost has become a major barrier to large-model training. In contrast, affordable hardware is low-cost but constrained by by memory capacity and bandwidth, making it unsuitable for direct LLM training. To address this, we propose MoE-DisCo (Mixture-of-Experts with Disentangled Clustering and Coordination)—a staged training framework. MoE-DisCo decomposes the MoE model into multiple dense submodels, each consisting of a shared backbone and a single expert, and partitions the training data into subsets using unsupervised clustering. Each submodel is trained independently and in parallel on its assigned data subset using low-cost devices, without any inter-device communication. Subsequently, all experts are integrated into a complete MoE model and fine-tuned globally for a short period on highmemory, high-bandwidth GPUs. Experiments show that our method matches or even surpasses full-parameter training in performance across multiple downstream tasks, loss function and perplexity (PPL) with a cost reduction of 47.6% to 69.5% on Qwen1.5-MoE-2.7B and Llama-MoE-3.5B across different datasets.

## 1 Introduction

Large Language Models (LLMs) and Mixture of Experts (MoE) have significantly advanced the field of natural language processing, yet their training economy cost has become a major barrier to broad participation. Current training paradigms heavily rely on high-memory GPUs such as NVIDIA A100/H800, which can cost over \$2.28 per GPU-hour in cloud environments. In contrast, affordable computing devices, such as DCUs, Ascend 910A, or consumer-grade GPUs, cost less. For example, the cost of DCU is less than \$0.03 per

hour. But, their limited memory capacity (typically ≤ 24 GB) and bandwidth make them seemingly unsuitable for training models at the billion- or trillion-parameter scale.

<span id="page-0-0"></span>![](_page_0_Figure_8.jpeg)

Figure 1: Model FLOPs Utilization (MFU) per GPU under 3D parallelism for language models of different scales as the number of GPUs increases. As the cluster scales from tens to thousands of GPUs, MFU per GPU consistently declines across all model sizes.[\(Narayanan](#page-9-0) [et al.,](#page-9-0) [2021\)](#page-9-0)

Furthermore, in large-scale model training, as the scale of GPU deployment expands, the per-GPU computational utilization almost inevitably declines. This is because larger clusters introduce more pronounced communication overhead—such as gradient synchronization and activation transmission—along with pipeline bubbles in pipeline parallelism, inter-node bandwidth bottlenecks, and load imbalance. These factors cause each GPU to spend more time waiting rather than computing, shown in Figure [1.](#page-0-0)

Therefore, using low-cost hardware as much as possible and minimizing multi-GPU coordination is key to reducing the cost of large model training.

To address high-cost challenge, we propose MoE-DisCo (Mixture-of-Experts with Disentangled Clustering and Coordination), a staged, hardware-aware training framework for MoE models. Specifically, MoE-DisCo decomposes a full MoE model with K-experts for each MoE layers into K independent dense submodels, each consisting of the shared Transformer backbone plus a single expert. Concurrently, we partition the original training data into K subsets using unsupervised clustering, establishing a disentangled alignment between experts and data clusters. Each submodel is trained exclusively on its assigned data subset, without any inter-model communication. Since each submodel is equivalent to a standard dense model (with a parameter scale much smaller than that of the full MoE model), it can be efficiently trained on a single low cost device. Crucially, this parallel training phase eliminates inter-device communication overhead. After all submodels are trained, MoE-DisCo reintegrates the individual expert modules into a unified MoE architecture and performs a brief global fine-tune phase on the full dataset. Although this final stage requires highmemory GPUs, it takes significantly less time than the total training time of conventional approaches, shown in Figure 2.

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 2: Comparison of training cost profiles between MoE-DisCo (left) and traditional MoE training (right). MoE-DisCo first performs the majority of training on low-cost GPUs (cost  $c_0$ ), which can be highly parallelized across multiple devices, followed by a short finetune phase on high-cost GPUs (cost  $c_1$ ). In contrast, traditional methods rely solely on high-cost hardware throughout the entire training process. The total cost (shaded area under the curve) is significantly reduced in MoE-DisCo, demonstrating its efficiency in lowering the monetary burden of large-scale MoE training.

Our main contributions are as follows: (1) We propose MoE-DisCo—a low-cost, staged MoE training framework designed for resource-constrained hardware; (2) We show that two stages of MoE-DisCo can be executed on hardware of different cost hardware, significantly reducing the overall training economy cost of large-scale MoE systems; (3) We evaluated MoE-DisCo across multiple datasets and model architectures. Results show that MoE-DisCo effectively reduces the

usage of high-cost GPUs while producing MoE models whose accuracy matches or even exceeds that of models trained with the original MoE approach with a cost reduction of 47.6% to 69.5% on Qwen1.5-MoE-2.7B and Llama-MoE-3.5B across different datasets. The code address is shown in Appendix.

#### 2 Related Work

The Mixture of Experts (MoE) is an architectural paradigm that enhances model capacity and computational efficiency (Jacobs et al., 2014) in different areas (Shao et al., 2021; Zhou et al., 2024; Zhang et al., 2024). Subsequently, the work (Shazeer et al., 2017) proposed the sparsely-gated MoE mechanism, which enables significant model scaling under a fixed computational budget—by activating only the top-K experts during each forward pass. GShard (Lepikhin et al., 2020) was the first to successfully apply MoE to the Transformer architecture and introduced the notion of "expert capacity". Subsequent works have improved MoE from multiple perspectives, Switch Transformer (Fedus et al., 2021), Hash FFN, proposed by Facebook AI (Roller et al., 2021), StableMoE (Dai et al., 2022).

The past decade, the researchers propose algorithms and high-performance software to make MoE practically useful (Gray et al., 2017; Narang et al., 2017; Artetxe et al., 2021; Du et al., 2021). Most of them are working on high performance framework (Rajbhandari et al., 2020; Kim et al., 2021; He et al., 2021, 2022; Liu et al., 2025; Jin et al., 2025a; Yu et al., 2024b; Jin et al., 2025a,b; Yuan et al., 2025). There are a few works working on how to train MoE model with low economy cost in algorithm view, which works well with system.

## 3 Methodology

In standard MoE training paradigms, although only a small number of experts are activated during inference, all expert parameters must be loaded and updated simultaneously during training to support backpropagation and gradient updates. This causes memory and computational overhead to grow linearly with the number of experts, severely limiting the trainability of MoE models on low-cost GPUs.

The core question addressed in this paper is: without significantly sacrificing the final performance of the model, can we alter the parameter update strategy to minimize the need for simultaneous loading of all experts during MoE training,

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 3: Overview of the MoE-DisCo training framework. The original MoE model is decomposed into multiple dense submodels, each containing a single expert and shared parameters. Training data is clustered via k-means to assign semantically distinct subsets to individual experts. Submodels are trained independently on low-cost GPUs, enabling high parallelism and reduced hardware cost. Finally, the experts are integrated into a unified MoE architecture and fine-tuned on a high-cost GPU using the full dataset.

thereby drastically reducing reliance on high-cost GPUs? To this end, we introduce Block Coordinate Descent (BCD) and SimulParallel SGD (Zinkevich et al., 2011) as the core optimization idea for MoE training and design an expert-level block training framework based on the structural characteristics of MoE, enabling the main training stages of large-scale MoE models to be completed on low-memory, low-cost GPUs.

#### 3.1 Start Point and Theoretical Foundation

Our approach is inspired by the theoretical foundations of two classical optimization frameworks: Block Coordinate Descent (BCD) and SimulParallel SGD (Zinkevich et al., 2011). These provide critical insights into low-cost MoE training from the perspectives of parameter update strategies and distributed training architectures, respectively.

BCD is an iterative algorithm widely used for large-scale non-convex optimization. Its core idea is to update only one block of parameters per iteration while keeping all others fixed, thereby significantly reducing memory and computational overhead. The MoE architecture exhibits similar sparsity during inference—the gating network activates only the Top-K experts, avoiding computation over

the entire model. However, in standard training, backpropagation must traverse all expert paths, making it impossible to exploit this sparsity for memory savings.

We propose integrating BCD into MoE training: in each step, only one expert and the shared backbone are updated, while all other experts remain frozen. In this setting, each training unit maintains only a dense sub-model equivalent to a single-expert branch. This enables training on low-memory devices. Moreover, different experts can be trained fully in parallel without any communication or synchronization.

However, merely decoupling parameters is insufficient to guarantee performance. The key challenge lies in assigning appropriate data to each expert to foster complementary representation learning. Inspired by SimulParallel SGD—which independently trains multiple model replicas on disjoint data subsets and aggregates them via parameter averaging—we view this method as an extreme case of MoE with uniform gating and All-*K* averaged outputs. Accordingly, in MoE-DisCo, maximizing the distributional divergence among the data subsets assigned to individual experts enhances expert specialization and system diversity, accelerating

convergence and improving ensemble effectiveness. To this end, we employ unsupervised clustering to partition the training corpus: first, semantic embeddings of samples are extracted using a pre-trained model; then, these embeddings are clustered into K groups (where K equals the number of experts) via K-Means, with each cluster assigned to a distinct expert. Clustering naturally groups semantically similar samples, ensuring separation between clusters in the embedding space and satisfying the requirement of "maximizing distributional divergence." This strategy draws inspiration from LRP [\(Yang,](#page-9-13) [2025;](#page-9-13) [Gururangan et al.,](#page-8-11) [2023\)](#page-8-11) and the theoretical foundations of SimulParallel SGD.

Furthermore, SimulParallel SGD also informs the fusion of the shared backbone: if the subdatasets are balanced in size, a simple average of backbone parameters suffices to approximate the global optimum; if they are severely imbalanced, a sample-count-weighted average (as in WP-SGD [\(Cheng et al.,](#page-8-12) [2020\)](#page-8-12)) is required to maintain unbiased gradients.

In summary, the complete MoE-DisCo training pipeline consists of four key stages: 1. Model Decoupling: The original MoE is decomposed into several dense submodels, each comprising the full shared backbone plus a single expert; 2. Data Decoupling: Semantic clustering generates divergent subdatasets, establishing a disentangled alignment between experts and data; 3. Independent Parallel Training: Each submodel is trained independently on its assigned subdataset in a fully decentralized manner, with zero communication overhead, and this process can be solved by low-cost hardware; 4. Model Reintegration and Fine-Tune: Experts and the shared backbone are fused, followed by a short global fine-tune phase on the full dataset to restore coordinated gating behavior. The whole process is shown in Figure [3.](#page-2-0)

## 3.1.1 Construction of Single-Expert Submodels

Let the MoE model contain E experts. Its parameters split into: (1) shared parameters θshared (embedding, attention, LayerNorm, etc.), and (2) expert parameters θexp = (θ1, . . . , θE), where θ<sup>k</sup> is the k-th expert's parameters. Thus, Θ = (θshared, θ1, . . . , θE).

MoE-DisCo constructs lightweight submodels to

drastically cut GPU memory during training. Each submodel includes the full θshared and only one expert θk. In every MoE layer, the gating mechanism is removed, retaining just a single expert—yielding a compact, dense submodel.

This design sharply reduces model size. For large MoEs, submodels become small enough to train efficiently on low-cost GPUs, shifting workloads from expensive hardware to affordable devices.

Critically, submodels are fully independent during training: no gradient/parameter exchange or synchronization is needed. This eliminates inter-GPU communication overhead and complex multi-GPU coordination, greatly simplifying parallel training. System scalability and deployment flexibility improve significantly. Moreover, since each submodel fits on low-cost hardware, individual device utilization rises, lowering overall training costs.

## 3.1.2 Dataset Partitioning

In MoE-Disco, we propose an unsupervised, clustering-based partitioning method to allocate semantically similar subsets of the training data to individual expert submodels. The approach begins by deriving a fixed-dimensional representation for each input sentence: given a sentence x = (x1, . . . , xn), we encode all its tokens using a pre-trained embedding layer and compute the sentence vector h<sup>x</sup> via mean pooling over the token embeddings,

$$h_x = \frac{1}{n} \sum_{i=1}^{n} \text{Embedding}(x_i)$$
 (1)

where n is the sentence length. These sentence vectors are then clustered using K-means with K = E clusters, where E denotes the number of experts.

## 3.1.3 Submodel Integrate and Global Training

After completing independent training of all singleexpert submodels, the complete MoE model is constructed and globally optimized through the following steps: 1. Expert layer merging: adopt a "direct integration" strategy to concatenate the trained expert parameters θ<sup>1</sup> ∼ θ<sup>E</sup> into a complete expert layer; 2. Shared parameter fusion: weighted average the shared parameters θ (k) shared (k = 1 ∼ E) of

<span id="page-4-0"></span>**Algorithm 1** MoE-DisCo:Mixture-of-Experts with Disentangled Clustering and Coordination

**Require:** Original dataset  $\mathcal{D}$ ; MoE shared parameters  $\theta_{\text{shared}}$ ; Number of experts E and ith expert's parameters  $\theta_i$ ; Model  $M(\theta, \mathcal{D})$ 

**Ensure:** Trained global MoE model  $M(\Theta, \mathcal{D})$ 

```
1: h_x \leftarrow \text{MeanPool}(\text{Embed}(x)) for all x \in \mathcal{D}

2: \{\mathcal{D}_1, \dots, \mathcal{D}_E\} \leftarrow \text{K-means}(\{h_x\}, K = E)

3: for k \leftarrow 1 to E do

4: \theta_{\text{shared}}^{(k)} \leftarrow \theta_{\text{shared}}

5: \Theta_k \leftarrow (\theta_{\text{shared}}^{(k)}, \theta_k)

6: Train M(\Theta_k, \mathcal{D}_k)

7: end for

8: \theta_{\text{exp}}^* \leftarrow \text{Concat}(\theta_1, \dots, \theta_E)

9: \theta_{\text{shared}}^* \leftarrow \frac{1}{E} \sum_{k=1}^{E} \theta_{\text{shared}}^{(k)}

10: \Theta \leftarrow (\theta_{\text{shared}}^*, \theta_{\text{exp}}^*)

11: Fine-tune M(\Theta, \mathcal{D})

return M(\Theta, \mathcal{D})
```

all submodels to obtain global shared parameters:

$$\theta_{\text{shared}}^* = \sum_{k=1}^{E} \gamma_k \theta_{\text{shared}}^{(k)} \tag{2}$$

 $\gamma_k$  is the weight gain from WP-SGD (Cheng et al., 2020). When subset of dataset size is almost the same,  $\gamma_k = \frac{1}{E}$ . The gating part is initialized by tensor concatenation. 3. Global fine-tune: assemble the merged parameters  $(\theta^*_{\text{shared}}, \theta^*_{\text{exp}})$  into a complete MoE model, and perform joint fine-tune on the original full dataset to alleviate distribution bias between experts and improve overall model performance.

#### 3.2 MoE-DisCo Algorithm

Based on above introduction, we give a MoE-DisCo algorithm description, which shown in Algorithm 1.

## 4 Experiments

In this work, we conduct a systematic comparison between MoE-DisCo and a full-parameter trained MoE baseline, evaluating both model performance and training cost. For performance evaluation, we compare the resulting models in terms of language modeling quality (measured by training loss, PPL and downstream task) demonstrating that MoE-DisCo preserves—or even enhances—algorithmic effectiveness relative to standard MoE training. For economy cost analysis, we focus specifically on resource consumption during the fine-tune phase, as

this is the only stage requiring high-cost GPUs (e.g., A100). The main training phase of MoE-DisCo, by contrast, runs entirely on low-cost hardware, making fine-tune the primary determinant of its overall infrastructure expense. For ablation study, we will show the influence of k-means cluster and the number of experts.

#### 4.1 Experimental Setup

#### **4.1.1** Models

**Qwen1.5-MoE-2.7B.** The Qwen1.5-MoE-2.7B model activates approximately 2.7 billion parameters during inference, while achieving performance comparable to dense models with around 7 billion parameters, such as Mistral-7B. In our experiments, we set the number of experts to 4 while keeping all other configurations identical to the original model. We use abbr. Owen in following parts.

**LLaMA-MoE-3.5B.** The LLaMA-MoE-3.5B model is built upon the LLaMA architecture and incorporates a Mixture-of-Experts design to improve parameter efficiency and scalability. In our experiments, we set the number of experts to 4 while keeping all other configurations identical to the original model. We use abbr. Llama in following parts.

## 4.1.2 Datasets

Experiments use three public standard datasets, as follows: 1. C4: A large-scale English corpus with strict filtering and cleaning; 2. WikiText-2: A high-quality English dataset compiled from Wikipedia, commonly used for language model benchmark evaluation; 3. OpenWebText: An open web text collection constructed following the GPT-2 training data.

## **4.1.3** Evaluation Metrics

We evaluate MoE-DisCo against the full-parameter trained MoE baseline using three key criteria: (1) language modeling capability, measured by training loss, PPL and downstream tasks; (2) training efficiency, assessed via the number of training steps required to reach a target loss and the total usage time of high-cost GPUs. (3) The economy cost of training a MoE model in dollar.

#### 4.2 Performance Comparison

Table 1 presents a comparison of training efficiency and convergence performance between MoE-DisCo and full-parameter training across two MoE architectures (Qwen and LLaMA) and three

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Figure 4: Loss trends between MoE-Disco on fine-tune stage and Full-Parameter across different datasets

datasets (C4, WikiText-2, and OpenWebText). The results show that MoE-DisCo substantially reduces the number of training steps required to reach the same training loss. For instance, on the Qwen architecture, the steps to achieve convergence decrease from 21150, 12500, and 28600 to 4100, 3150, and 6650 on the C4, WikiText-2, and Open-WebText datasets, respectively, representing more than a fourfold reduction. A similar trend is observed for the LLaMA architecture. Moreover, at the same training loss, the PPL of models trained by MoE-DisCo metgod is comparable to, and in some cases lower than that of models trained by fullparameter method, indicating that the method accelerates training without compromising language modeling performance.

Figures [4](#page-5-1) and [5](#page-6-0) show the training loss and ppl curves of the Full-Parameter and MoE-DisCo finetune stage methods on three datasets, respectively. We use fine-tune stage of MoE-DisCo because this part is the most expensive stage (Cost information is shown in Economic Cost Analysis section). It can be observed that, given the same training duration, models trained with the MoE-DisCo method achieve significantly better performance than those trained with full-parameter training. If the target is to reach the same model performance, the MoE-DisCo method substantially reduces the required training time. This indicates that the proposed method can markedly lower training economy costs without sacrificing model performance. The whole

<span id="page-5-0"></span>

| Model | Data                                  |              | Full-Param |                           | MoE-DisCo |                    |        |  |
|-------|---------------------------------------|--------------|------------|---------------------------|-----------|--------------------|--------|--|
|       |                                       | Step         | Loss       | PPL                       | Step      | Loss               | PPL    |  |
| Qwen  | C4                                    |              |            | 21,150 4.954 230.32 4,100 |           | 4.925              | 165.86 |  |
|       | WikiText-2                            | 12,500 2.303 |            |                           |           | 71.89 3,150 2.2964 | 57.55  |  |
|       | OpenWebText 28,600 4.606 162.51 6,650 |              |            |                           |           | 4.588              | 134.31 |  |
| Llama | C4                                    |              |            | 11,750 5.274 356.28 4,150 |           | 5.276              | 296.35 |  |
|       | WikiText-2                            |              |            | 15,000 3.805 160.91 5,300 |           | 3.794              | 163.27 |  |
|       | OpenWebText 21,000 4.536 114.93 7,800 |              |            |                           |           | 4.541              | 106.82 |  |

Table 1: Performance comparison between MoE-DisCo and full-parameter training across two MoE architectures. The table show that the number of steps required for MoE-DisCo to achieve full-parameter convergence result.

training process including submodel training is shown in Cost Analysis part and Appendix.

