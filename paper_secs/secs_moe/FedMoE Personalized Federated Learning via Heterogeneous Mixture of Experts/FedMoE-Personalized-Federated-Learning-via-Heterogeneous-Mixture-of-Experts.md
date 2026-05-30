# FedMoE: Personalized Federated Learning via Heterogeneous Mixture of Experts

Hanzi Mei<sup>1</sup> , Dongqi Cai<sup>1</sup> , Ao Zhou<sup>1</sup> , Shangguang Wang<sup>1</sup> , Mengwei Xu<sup>1</sup>

> <sup>1</sup>Beijing University of Posts and Telecommunications {meihanzi, cdq, aozhou, sgwang, mwx}@bupt.edu.cn

## Abstract

As Large Language Models (LLMs) push the boundaries of AI capabilities, their demand for data is growing. Much of this data is private and distributed across edge devices, making Federated Learning (FL) a de-facto alternative for finetuning (i.e., FedLLM). However, it faces significant challenges due to the inherent heterogeneity among clients, including varying data distributions and diverse task types. Towards a versatile FedLLM, we replace traditional dense model with a sparsely-activated Mixture-of-Experts (MoE) architecture, whose parallel feed-forward networks enable greater flexibility. To make it more practical in resourceconstrained environments, we present FedMoE, the efficient personalized FL framework to address data heterogeneity, constructing an optimal sub-MoE for each client and bringing the knowledge back to global MoE. FedMoE is composed of two fine-tuning stages. In the first stage, FedMoE simplifies the problem by conducting a heuristic search based on observed activation patterns, which identifies a suboptimal submodel for each client. In the second stage, these submodels are distributed to clients for further training and returned for server aggregating through a novel modular aggregation strategy. Meanwhile, FedMoE progressively adjusts the submodels to optimal through global expert recommendation. Experimental results demonstrate the superiority of our method over previous personalized FL methods.

## Introduction

Emerging applications of Large Language Models (LLMs) like ChatGPT and Sora are transforming the AI landscape (Wu et al. 2023; Liu et al. 2024). However, such models require increasing amounts of downstream data for finetuning (Kaplan et al. 2020). Currently, the public cloud data is running out (Villalobos et al. 2022). A huge amount of private data is still unexplored on distributed edge devices such as smartphones and autonomous vehicles. To leverage such distributed and private data, Federated Learning (FL) has become a de-facto approach (Bonawitz et al. 2019), allowing edge devices to process data on-device and collaboratively train stronger LLMs while preserving privacy, noted as FedLLM (Xu et al. 2024).

Despite the promising vision, FedLLM remains challenging in heterogeneous reality. Generally, participants possess data that varies in domain and quality due to varying environments, typically noted as non-independent and identically distributed (non-IID) issue (Zhu et al. 2021). More critically, when extended to cross-task scenarios with different optimization objectives, the heterogeneity could be further exacerbated. This difficult but more realistic setting has been researched as task level (Cai et al. 2023; Yao et al. 2022). Therefore, personalized FL has emerged, with some approaches aiming to train proprietary and heterogeneous local models for each client. Prior studies mainly focus on model distillation or model pruning derived from dense models (Zhu et al. 2022; Ilhan, Su, and Liu 2023).

Recently, Mixture-of-Experts (MoE) (Rajbhandari et al. 2022; Jiang et al. 2024) has been widely explored to scale up the model efficiently, with experts pretrained to be sparsely activated for various tokens. We thereby prototype FedMoE, the personalized FL framework that integrates transformerbased MoE into FL to address heterogeneity challenges. The impetus behind FedMoE stems from the unique and inherent features of MoE model. The expert-parallel structure along with the sparsely-activated mechanism has the potential to facilitate personalization for different clients, as each client flexibly activates the most relevant parameters. Furthermore, the MoE model supports on-demand scaling of model capacity (Krajewski et al. 2024), enabling it to adapt to the increasing complexity of data and tasks in FL.

While the idea is intriguing, deploying MoE models into edge clients is not trivial. For example, training a Switch Transformers model (Fedus, Zoph, and Shazeer 2022) with 32 experts per layer demands at least 37GB of memory, overly exceeding the capacity of many client devices. To further facilitate the practical deployment of FedMoE, we propose an efficient FedMoE fine-tuning system to tackle data heterogeneity while considering resource constraints. The main idea is to construct a specific, optimal sub-MoE model for each client, and then aggregate the knowledge back to the global MoE model. The holistic system is composed of two stages. In the first stage, FedMoE efficiently reduces the problem complexity by identifying an initial submodel close to optimal for each client. Specifically, it conducts a heuristic search among all experts based on activation patterns observed during preliminary fine-tuning. In the second stage, the initial submodels are sub-sampled from the global model and distributed to clients for edge training. After edge training, FedMoE adopts a modular aggregation strategy in cloud server, sharing the relevant knowledge across clients while

![](_page_1_Figure_0.jpeg)

Figure 1: Heatmap of expert activation frequencies across different training stages and datasets.

excluding negative interference. Additionally, FedMoE progressively adjusts the edge model structures through expert recommendation from the global perspective, striking the balance between efficiency and performance.

In general, our contributions are highlighted as follows:

- We conduct preliminary experiments to demonstrate the characteristics of expert activation during heterogeneous FedLLM fine-tuning.
- We present FedMoE, an efficient FL system integrating transformer-based MoE models to address heterogeneity issues. FedMoE dynamically searches and distributes personalized experts for different clients and absorbs the knowledge back into a generic global model.
- We conduct extensive experiments to demonstrate the empirical effectiveness of FedMoE. Owing to meticulously designed personalization, FedMoE achieves superior performance across all tasks, while reducing the memory footprint and network traffic over existing baselines.

## **Background and Motivation**

#### **Federated Learning**

The FL paradigm enables multiple edge devices to collaboratively train a shared model through rounds of edge-cloud communications, without sharing the local data (Kairouz et al. 2021). In each round, the cloud server randomly selects a set of edge devices (or *clients*) and distributes the latest global model for edge fine-tuning. The server then collects those edge models to update the global model through aggregation. FL serves as a privacy-friendly training paradigm to utilize decentralized data under legal regulation such as GDPR (Wikipedia 2024) and CCPA (Wikipedia contributors 2024).

Despite FL's effectiveness in building privacy-preserving LLMs, training a generic large model across multiple tasks is still in its infancy. Given the divergent or even conflicting update directions produced by different tasks, it is challenging for global model to achieve optimal convergence (Tan et al. 2022). Furthermore, the fine-tuning process in FL is bottlenecked by on-device resources and network transmission (Lim et al. 2020). The global model must be affordable for the weakest device, which restricts the model scale and overall performance. Meanwhile, the resources of other devices are underutilized.

### **Mixture of Experts**

Inspired by conditional computation, the MoE architecture breaks the principle of using the same set of parameters for all inputs (Shazeer et al. 2017). An MoE layer consists of a trainable gating network and a pool of experts, each a feed-forward network (FFN). The gating networks within the MoE determine the pathway of sequentially activated experts for each token. In the case of Switch Transformers, one of the most popular MoE-based models, the gating network is simplified to a top-1 router. Each token activates one expert per layer based on the probabilities computed by router, leaving the other experts unupdated.

MoE architecture scales up the model capacity in a computationally efficient manner, achieving significant performance improvements across many downstream tasks (Fedus, Dean, and Zoph 2022). The effectiveness also extends from individual tasks to multi-task learning scenarios, where multiple objectives are learned simultaneously (Gupta et al. 2022; Kim et al. 2021). The advantage of MoE can be attributed to the unique expert-parallel structure and sparsely-activated mechanism, rendering each expert capable of handling specific tasks or data subsets. Therefore, MoE model shows significant potential for adapting to multi-tasking and heterogeneous data environments.

### **Motivation and Preliminary Experiments**

Sharing all parameters across clients is not a silver-bullet approach in FL, especially when dealing with heterogeneous data. Unlike previous FL studies, we aim to train an MoE-architectured backbone. The parameter spaces in sparse layers are disentangled, allowing for a more flexible and ondemand parameter-sharing scheme. Specifically, a subset of optimal experts can be sub-sampled from the global model, constructing a client-specific heterogeneous model for personalized learning, and bringing the knowledge back to the global model during aggregation. This novel scheme allows clients to share the most relevant parameters, encouraging cooperation while preventing interference.

However, the main obstacle lies in identifying the optimal experts for specific data under the resource constraints. To gain insights for the design of FedMoE, we conduct a series of preliminary experiments based on Switch Transformers to explore the characteristics of expert activation.

**Observation-1: Expert activation characteristics are skewed and dynamic during fine-tuning.** As illustrated in Figure 1, experts exhibit a distinct imbalance in activation

![](_page_2_Figure_0.jpeg)

Figure 2: Performance of full model (8 experts per layer) and submodels (4 optimal experts per layer, 4 random experts per layer).

frequencies, indicating that only certain experts (those frequently activated) are of critical importance (Yi et al. 2023; Lu et al. 2024). The comparison between Figure 1a and Figure 1c further reveals that fine-tuning significantly alters the activation patterns, enabling the model to identify the optimal experts progressively. This observation suggests that the importance of experts should be assessed dynamically during training, rather than statically beforehand.

Observation-2: Optimal experts converge quickly and perform comparably. We monitor the activation characteristics at every epoch during fine-tuning on a text summarization task, whose evolution is shown in Figure 1a, Figure 1b and Figure 1c. The optimal experts quickly stabilize after a certain epoch, exhibiting only minor fluctuations in their activation frequencies, which facilitates model pruning. As demonstrated in Figure 2, the submodel constructed by optimal experts shows a comparable performance to full model, while the submodel built with randomly chosen experts falls short.

Observation-3: Optimal experts differ across heterogeneous data, especially when heterogeneity reaches the task level. As illustrated in Figure 1d and Figure 1e, within the same task, data of diverse distributions favor different subsets of experts, albeit with some overlap. The difference is exacerbated when it comes to different tasks that require distinct knowledge, as demonstrated by Figure 1c and Figure 1e.

## Method

#### **Problem Formulation**

In personalized FL, there are n clients collaborating to learn m types of downstream tasks under varying memory constraints. Each client  $k=1,2,\ldots,n$  executes edge training based on a local dataset  $\mathcal{D}_k:=(\mathcal{X},\mathcal{Y}_k,p_k(x))$ . The input space  $\mathcal{X}$  is globally shared, while the input distribution  $p_k(x)$  can be either IID or non-IID across different clients. The output space  $\mathcal{Y}_k$  corresponds to a target task  $\mathcal{T}_k\in\{1,2,\ldots,m\}$ . The union  $\mathcal{D}=\bigcup_{k=1}^n\mathcal{D}_k$  represents the overall dataset used for FL.

The local objective of client k is to minimize the sum of label-smoothed cross-entropy loss and the weighted load balance loss for MoE, while keeping the memory usage of the edge model  $w_k$  within the memory limit  $M_k$ , formulated

as:

$$\min_{w_k} \mathcal{L}_k = \mathbb{E}_{(x,y) \sim \mathcal{D}_k} \left[ \mathcal{L}_{CE}(y, \hat{y}(x; w_k)) + \alpha \mathcal{L}_{LB} \right]$$
s.t.  $\text{mem}(w_k) \le M_k$  (1)

The clients in FL collaboratively train a global model w to minimize the global objective function, formulated as:

$$\min_{w} \mathcal{L} = \sum_{k=1}^{n} \frac{|\mathcal{D}_{k}|}{|\mathcal{D}|} \mathcal{L}_{k} (w_{k})$$
 (2)

#### Overview

**Model Structure** The cloud hosts a large MoE model with an equal number of experts at every layer, initialized by a pre-trained MoE model. Global model's extensive capacity enables it to provide a broad range of knowledge and store new information effectively. The clients host heterogeneous sub-MoEs, differing in both model architecture (i.e., different expert numbers per layer) and parameter space (e.g., different pieces of the global model). These client-specific submodels are sub-sampled from the global model, retaining the most relevant experts to flexibly adapt to data characteristics under memory constraints.

Workflow FedMoE is structured into two stages, as illustrated in Figure 3. In the first stage, the server collects activation information from clients after memory-efficient finetuning at the edge (steps ①—②). Based on this information, the server conducts coarse-grained submodel searches (step ③) to determine an approximate model architecture for each client (i.e., an initial mapping of each client to its preferred expert subset). In the second stage, federated learning begins with the submodels initialized in stage one (step ④). In each round, fine-tuned submodels from different clients are integrated into the global model through modular aggregation (step ⑤—⑦), followed by fine-grained structural adjustments guided by real-time feedback (step ⑥).

#### **Stage One: Coarse-grained Submodel Initialization**

To begin with, we employ expert activation probability as a criterion to assess expert importance under specific data distributions, represented as  $p_{i,j} = \frac{n_{i,j}}{N}$ , where  $n_{i,j}$  is the activation times of the j-th expert in i-th layer and N is the total token count. Experts with higher activation probabilities are more crucial for the downstream task. Stage one involves a large-scale problem, as it requires fairly evaluating the importance of all experts across a vast number of clients. Conducting such evaluation within the memory limits of edge devices exacerbates the complexity.

Activation Probability Collection The cloud sends a full-scale MoE to clients with all experts working in parallel. Clients with sufficient memory further equip each expert with LoRA structure and then perform Parameter-Efficient Fine-Tuning (PEFT). PEFT is a well-established method for memory-efficient fine-tuning (Hu et al. 2021; Sun et al. 2022). Since expert preferences converge quickly, fine-tuning only lasts a few rounds (e.g., 5 rounds) to obtain expert activation probabilities on validation datasets.

![](_page_3_Figure_0.jpeg)

Figure 3: FedMoE workflow.

After gathering activation information, the cloud server performs a static prediction of expert activation probabilities for memory-insufficient clients, based on the intuition that homogeneous data typically favor a similar set of experts. Specifically, the cloud calculates the estimated probabilities for a given client by averaging the probabilities of other clients with the same task, weighted by their data volume. Ultimately, the cloud collects activation probabilities for all experts across all clients in a rough yet efficient manner, laying the foundation for constructing personalized models.

Heuristic Submodel Search Based on the activation probabilities, the cloud strives to initialize a suboptimal submodel for each client under its memory constraint. The issue is further modeled as an optimization problem. For each client, the cloud aims to find a maximum threshold θ, ensuring the combined activation probabilities of the experts retained in each layer are at least θ. The uniform threshold across each layer guarantees the optimal performance of this model, as some layers only require a few skilled experts, whereas others demand more to collaborate. Additionally, the memory usage of the retained experts and dense layers must not exceed α(0 < α ⩽ 1) of total available memory. The reserved memory allows for fine-grained model structural adjustments in the second stage, which potentially requires more experts. Solving for the binary variable xi,j ∈ {0, 1} that indicates whether to retain the j-th expert in the i-th layer, the problem is formulated as:

$$\max_{x_{i,j}} \theta$$
s.t. 
$$\begin{cases}
\sum_{j=1}^{E_i} x_{i,j} p_{i,j} \geqslant \theta, \forall i = 1, \dots, L \\
\min\left(\sum_{i=1}^{L} \sum_{j=1}^{E_i} x_{i,j} w_{i,j} + w_d\right) \leqslant \alpha \cdot M
\end{cases}$$
(3)

Since solving the NP-hard multi-dimensional knapsack problem directly is computationally prohibitive, we instead employ a heuristic algorithm based on binary search. In practice, the cloud searches for the optimal threshold θ within the range of [0, 1] by gradually adjusting the bounds of feasible region. For a given threshold, FedMoE attempts to construct a smallest submodel with respect to the threshold and verify whether it exceeds the memory limit. If not exceed, the lower bound of θ is adjusted upward to find a more effective submodel; otherwise, the upper bound of θ is adjusted downward to alleviate memory consumption. The efficient search initializes the client-expert map in the cloud, achieving an ideal balance between memory usage and performance.

## Stage Two: Federated Training and Fine-grained Submodel Adjustment

In the second stage shown in Algorithm 1, federated training begins with the submodels initialized by the first stage. After fine-tuning on clients' private data, the submodel is filled with personalized knowledge. The challenge lies in how to encourage cooperation while reducing interference among heterogeneous submodels during aggregation. Additionally, initial submodels may prove suboptimal, either failing to capture personalized knowledge (*underfitting*) or being excessively redundant (*overfitting*). The cloud serves as a central node, crucial not only for managing model parameters but also for structural model adjustments. Specifically, FedMoE incorporates clients with diverse data distributions and various task types through modular model sub-sampling and aggregation. Afterwards, the cloud progressively adjusts the submodels based on the expert recommendation mechanism from a global perspective, continually optimizing the overall system.

Algorithm 1: Federated training and submodel adjustments in the second stage.

```
1: for each round r = 1, \ldots, R do
          S \leftarrow \text{sample subset of clients } \mathcal{U} = \{u_1, \dots, u_n\}
 3:
          Construct and send w_k to each u_k \in \mathcal{S}
 4:
          for client u_k \in \mathcal{S} in parallel do
              w_k^* \leftarrow \text{TRAIN}(w_k, \mathcal{D}_k^{\text{train}})
  5:
              p_{\text{all\_experts}}, acc \leftarrow \text{VALIDATE}(w_k^*, \mathcal{D}_k^{\text{val}})
 6:
 7:
              send w_k^*, p_{\text{all\_experts}}, acc to server
 8:
          w_{global} \leftarrow \text{modular aggregate } w_k^* \text{ for } u_k \text{ in } \mathcal{S}
 9:
10:
          for client u_k \in \mathcal{S} do
11:
              if acc not improved then
                  for client u_a \in \mathcal{U} \setminus u_k do
12:
                      Caculate sim(u_k, u_a) // refer to equation (4)
13:
14:
                  \mathcal{S}' \leftarrow \text{top } K \text{ similar clients}
15:
                  n = AVG(n_{expert}(S')) - n_{expert}(u_k)
16:
                  if n > 0 then
17:
                      for expert out of w_k do
18:
                          caculate \hat{p}_{\mathsf{expert}}
                                                          // refer to equation (6)
19:
20:
                      \mathcal{E} \leftarrow \text{top } n \text{ experts ranked by highest } \hat{p}_{\text{expert}}
21:
                      add experts \mathcal{E} to submodel w_k
22:
23:
24:
                      for expert within w_k do
                          caculate \hat{p}_{\text{expert}}
                                                         // refer to equation (6)
25:
26:
                      \mathcal{E} \leftarrow \mathsf{top}\; n \; \mathsf{experts} \; \mathsf{ranked} \; \mathsf{by} \; \mathsf{lowest} \; \hat{p}_{\mathsf{expert}}
27:
                      remove experts \mathcal{E} out of submodel w_k
28:
29:
                  end if
              end if
30:
          end for
31:
32: end for
```

**Submodel Deployment** At the beginning of each round, a random selection of clients is chosen to participate in the training. The cloud sub-samples the dense layers and optimal expert subset from the global MoE based on the latest client-expert map, constructing a personalized submodel for each client. Deployed on edge devices, these submodels are fine-tuned on local training dataset. Subsequently, clients perform inference on their validation dataset, collecting the up-to-date expert activation probabilities and validation scores which are sent to cloud along with the edge model parameters.

Modular Aggregation After receiving all edge models, the cloud integrates newly-learned knowledge back into the global model based on the modular aggregation strategy. For parameters of dense layers, the cloud conventionally employs the Federated Averaging (FedAvg) strategy. For parameters of sparse layers, aggregation becomes more complicated due to the heterogeneous expert subsets across clients. In detail, unactivated experts in the global model remain unchanged, experts used by a single client are updated directly, and experts shared by multiple clients are aggregated based on the FedAvg strategy. The expert-

corresponding dimensions of routers are updated in the same pattern. Module-granular updates allow for independent optimization of different parts within the global model, fostering collaboration and mutual enhancement while preventing conflicts across different clients.

**Expert Recommendation** Besides updating the global model, the cloud precisely optimizes the submodel structure to ensure learning efficiency throughout the training cycle. If a client shows no performance gains after several rounds, its model is considered to have reached a bottleneck. The cloud leverages insights provided by other clients from a global perspective, conducting expert-granular structural adjustments. Specifically, the cloud calculates the cosine similarity between the current client and other clients based on their expert activation probabilities, formulated as:

$$\nsim (\mathbf{u}_{k}, \mathbf{u}_{a}) = \frac{\sum_{i=1}^{L} \sum_{j=1}^{E_{i}} \left( (p_{i,j})_{\mathbf{u}_{k}} \times (p_{i,j})_{\mathbf{u}_{a}} \right)}{\sqrt{\sum_{i=1}^{L} \sum_{j=1}^{E_{i}} \left( p_{i,j} \right)_{\mathbf{u}_{k}}^{2}} \times \sqrt{\sum_{i=1}^{L} \sum_{j=1}^{E_{i}} \left( p_{i,j} \right)_{\mathbf{u}_{a}}^{2}}}$$
(4)

The current client takes the top K clients ranked by similarity as reference, as indicated by:

$$\mathbb{S}(\mathbf{u}_k) = \{ \mathbf{u}_a | rank \operatorname{sim}(\mathbf{u}_k, \mathbf{u}_a) \leqslant K, a \neq k \} \quad (5)$$

If their number of experts exceeds that of the current client, the cloud recommends incorporating more effective experts into its submodel; otherwise, it suggests pruning underperforming experts. To fairly evaluate the effectiveness of all experts (including those within and outside the submodel), the cloud estimates their activation probabilities based on those of top-k most similar clients, formulated as:

$$(\hat{p}_{i,j})_{\mathbf{u}_k} = \frac{\sum_{\mathbf{u}_a \in \mathbb{S}(\mathbf{u}_k)} \sin(\mathbf{u}_k, \mathbf{u}_a) \times (p_{i,j})_{\mathbf{u}_a}}{\sum_{\mathbf{u}_a \in \mathbb{S}(\mathbf{u}_k)} \sin(\mathbf{u}_k, \mathbf{u}_a)}$$
(6)

Note that this is an exploratory adjustment. If the performance of the adjusted submodel does not improve, its structure will revert to the previous version and remain fixed thereafter.

## **Experiments**

### **Experimental Setups**

FL Simulations We create four FL simulations based on three classic NLP downstream datasets, including AG News (Zhang, Zhao, and LeCun 2015) for text classification (task-TC), SQuAD (Rajpurkar et al. 2016) for reading comprehension (task-RC) and XSum (Narayan, Cohen, and Lapata 2018) for text summarization (task-TS). The evaluation metrics are accuracy, F1 score and Rouge-2 score, respectively. Each simulation setting mimics a complex realworld scenario. (1) Standard-Hetero-T involves 30 clients with heterogeneous tasks, and 5 clients are randomly selected for training each round. (2) Standard-Hetero-TD

Table 1: End-to-end performance of different personalized methods in various FL settings.

(a) Setting: Standard-Hetero-T

(b) Setting: Standard-Hetero-TD

| Method                         | Performa<br>task- task-<br>TC RC | nce<br>task-<br>TS | Comm.<br>Vol. (GB) | Mem.<br>Usage (GB)              | Method        |       | forman<br>task-<br>RC | ce<br>task-<br>TS | Comm.<br>Vol. (GB) | Mem.<br>Usage (GB) |
|--------------------------------|----------------------------------|--------------------|--------------------|---------------------------------|---------------|-------|-----------------------|-------------------|--------------------|--------------------|
| randomMoE                      | 91.63 84.2                       | 3 14.51            | 2.30               | 15.63                           | randomMoE     | 34.19 | 82.93                 | 13.51             | 2.30               | 15.56              |
| FedProx                        | 92.92 <b>87.9</b>                | 9 11.94            | 2.30               | 24.71                           | FedProx       | 85.09 | 87.57                 | 11.76             | 2.30               | 24.63              |
| SCAFFOLD                       | 85.98 69.4                       | 4 5.86             | 4.61               | 17.29                           | SCAFFOLD      | 61.72 | 67.17                 | 5.83              | 4.61               | 17.23              |
| FedMoE (Ours)                  | <b>94.76</b> 86.6                | 4 16.92            | 1.76               | 13.44                           | FedMoE (Ours) | 88.44 | 86.55                 | 16.63             | 1.85               | 13.89              |
| (c) Setting: Enforced-Hetero-T |                                  |                    |                    | (d) Setting: Enforced-Hetero-TD |               |       |                       |                   |                    |                    |
| Method                         | Performa<br>task- task-<br>TC RC | nce<br>task-<br>TS | Comm.<br>Vol. (GB) | Mem.<br>Usage (GB)              | Method        |       | forman<br>task-<br>RC | ce<br>task-<br>TS | Comm.<br>Vol. (GB) | Mem.<br>Usage (GB) |
| randomMoE                      | 88.86 82.6                       | 3 14.37            | 2.30               | 15.44                           | randomMoE     | 37.81 | 82.61                 | 13.80             | 2.30               | 15.45              |
| FedProx                        | 92.51 <b>86.6</b>                | 9 11.88            | 2.30               | 24.55                           | FedProx       | 73.85 | 86.59                 | 12.03             | 2.30               | 24.51              |
| SCAFFOLD                       | 36.17 72.5                       | 1 6.78             | 4.61               | 17.21                           | SCAFFOLD      | 69.34 | 70.69                 | 6.96              | 4.61               | 17.18              |
| FedMoE (Ours)                  | <b>94.85</b> 85.7                |                    | 1.89               | 13.78                           | FedMoE (Ours) | 70 56 | 85.84                 | 1 ( =0            | 1.95               | 14.04              |

further introduces data heterogeneity by assigning label-skewed non-IID datasets to clients with identical tasks, following the methodology in FedNLP (Lin et al. 2021). (3) **Enforced-Hetero-T** simulates a more conflicting scenario by forcibly selecting 3 clients with different task types in each training round, using the same 30-client setup as in Standard-Hetero-T. (4) **Enforced-Hetero-TD** adopts conflicting client selection scheme and uses the same 30-client setup as in Standard-Hetero-TD. The clients in the above settings are resource-heterogeneous, with memory capacities ranging from 18GB to 24GB, typical of high-performance smartphones and edge computing platforms.

Baselines We compare FedMoE with three personalized FL methods: (1) randomMoE constructs personalized edge models by randomly selecting a subset of experts from the global MoE, ensuring a degree of information isolation among clients. (2) FedProx (Li et al. 2020) is a federated optimization algorithm that incorporates a regularization term during local updates to mitigate the impacts of heterogeneity. (3) SCAFFOLD (Karimireddy et al. 2020) employs control variates to correct the local update directions, overcoming client-drift caused by heterogeneous data.

**Models** The following experiments are conducted based on Switch Transformers architecture whose pre-trained weights are downloaded directly from Hugging Face (Wolf et al. 2019). The global models for FedMoE and random-MoE are configured with 32 experts per layer, whereas others are configured with 8 experts per layer due to edge device memory constraints.

#### **End-to-End Performance**

Table 1 summarizes end-to-end performance of four methods in various FL settings, including task performance, communication volume and peak memory usage. Figure 4 shows the end-to-end performance throughout the training process

in one of the settings. From the results, we draw four key observations as follows:

FedMoE outperforms baselines in overall task performance. FedMoE typically demonstrates comparable or superior performance across various tasks, achieving average improvements of 7.63%, 15.50%, 10.00%, and 12.74% over the best-performing baseline in the four settings. Notably, for complex tasks such as task-TS, FedMoE beats the best baseline non-trivially with improvements of up to 23.09%, effectively avoiding the knowledge interference between different tasks that occurs in FedProx and SCAF-FOLD.

**FedMoE** achieves faster convergence. Figures 4a–4c illustrate the convergence process across various task types in Enforced-Hetero-T Setting. To achieve 99% of the relative target performance, FedMoE demonstrates a speedup of 1.65×, 2.13×, and 1.64× compared to the three baselines, and 1.35×, 2.06×, and 2.92× for 90% of the target performance.

FedMoE reduces communication overhead and memory usage significantly, making personalized FL resource-efficient. The communication overhead of FedMoE decreases by 19.11%, 19.11%, and 59.56% on average compared to the three baselines, and memory consumption decreases by 11.16%, 43.95%, and 19.97%, respectively. The optimization methods in FedProx and SCAFFOLD introduce additional memory and communication overheads, which are accumulated during long-term FL training. In contrast, the cost of FedMoE is negligible, with only a one-time roundtrip communication of about 7.46GB and a memory overhead of approximately 13.06GB that lasts for only a few rounds in the first stage.

FedMoE demonstrates strong robustness and stable performance in complex scenarios. We calculate the Coefficient of Variation (CV), denoted as  $c_v = \frac{\sigma}{\mu}$ , across various settings for each task and use the average value as the

![](_page_6_Figure_0.jpeg)

Figure 4: End-to-end performance of different personalized methods during training under Enforced-Hetero-T setting.

Table 2: Ablation study under Standard-Hetero-T setting.

|                          | Perf        | orman       | ice            | Expert Num                                              |                  |  |
|--------------------------|-------------|-------------|----------------|---------------------------------------------------------|------------------|--|
| Method                   | task-<br>TC | task-<br>RC | task-<br>TS    | Avg.                                                    | Min/Max          |  |
| FedMoE                   | 94.76       | 86.64       | 16.92          | $78 \rightarrow 65$                                     | 52/92            |  |
| w/o stage1<br>w/o stage2 |             |             | 14.50<br>16.69 | $\begin{array}{c} 96 \rightarrow 104 \\ 78 \end{array}$ | 94/118<br>50/114 |  |

Composite Variation Index (CVI) to assess the robustness of each method. FedMoE with a notably small CVI of 0.0445 and FedProx with a CVI of 0.0569 exhibit relatively stable performance, whereas randomMoE with a CVI of 0.1827 and SCAFFOLD with a CVI of 0.1430 exhibit erratic performance. FedMoE benefits significantly from selecting the most relevant subset of experts for different settings.

#### **Ablation Study**

Table 2 compares FedMoE to variants without either the first or second stage, showing task performance, the evolution of average expert numbers throughout training, and the maximum and minimum expert numbers of clients after training. The results indicate that both stages are vital for achieving strong performance, particularly the first stage where expertactivated information is collected. The second stage plays a pivotal role in adjusting the number of experts. When built upon the solid starting point provided by the first stage, it effectively prunes the redundant experts, thereby improving resource efficiency while maintaining strong performance. The two-stage training paradigm achieves an optimal balance between performance and resource consumption.

#### **Related Work**

**Personalized Federated Learning (PFL).** With the growing concern for data privacy and model efficiency, PFL has recently garnered extensive attention (Vanhaesebrouck, Bellet, and Tommasi 2017; Tan et al. 2022; Sabah et al. 2023). A common strategy is regularization, which adds penalty terms to the loss function or update direction to guide the training process (Li et al. 2020; Yao and Sun 2020; Karimireddy et al. 2020). However, it is sensitive to regularization parameters and difficult to generalize to complex scenarios. Another approach is to build heterogeneous submodels for clients, utilizing techniques such as model dis-

tillation (Ni, Shen, and Zhao 2022; Lin et al. 2020; Zhu et al. 2022), pruning (Ilhan, Su, and Liu 2023), or quantization (Ozkara et al. 2021), allowing knowledge transfer between global model and submodels. However, this approach depends on proxy data and may prolong the training process. Parameter decoupling makes heterogeneous submodels more structured by separating the model into shared and personalized layers (Arivazhagan et al. 2019; Wu et al. 2021; Ma et al. 2022; Mei et al. 2021). However, designing the optimal layer combination for each client is challenging. Some studies allow clients to maintain their original heterogeneous models. The server then decomposes, groups, and aggregates these models based on module similarity (Wang et al. 2024, 2023). However, calculating modular similarity for large-scale models causes significant computational complexity.

FedMoE is one of the first PFL methods that supports LLMs with billion-scale parameters and can efficiently scale up, while previous methods are impractical for LLMs due to their complexity.

Mixture of Experts (MoE) Optimizations. MoE has demonstrated effectiveness in handling complex centralized learning issues, including multi-task (Chen et al. 2023a,b), multi-domain (Zhang et al. 2024), and multi-scenario (Zou et al. 2022). For example, AdaMV-MoE (Chen et al. 2023a) exhibits enhanced performance in multi-task vision recognition by automatically adjusting the number of experts for each task. Although some FL studies draw inspiration from the MoE concept, they arbitrarily treat the global and local models as individual experts and simply combine their outputs (Zec et al. 2020; Guo et al. 2021; Bai et al. 2022; Zhang et al. 2023). Experts at model granularity leads to limited flexibility and inadequate personalization.

FedMoE is the first to integrate MoE into FL for a generic and versatile FedLLM, while previous works involve managing a collection of models.

## **Conclusions**

FedMoE addresses the challenges of data heterogeneity in FL by employing models with MoE architecture. Through personalized sub-MoE construction, modular aggregation, and dynamic model adjustments, FedMoE enhances overall performance across diverse tasks, while significantly reducing memory footprint and network traffic. Empirical experiments demonstrate its effectiveness in cross-task scenarios.

## References

- Arivazhagan, M. G.; Aggarwal, V.; Singh, A. K.; and Choudhary, S. 2019. Federated learning with personalization layers. *arXiv preprint arXiv:1912.00818*.
- Bai, T.; Zhang, Y.; Wang, Y.; Qin, Y.; and Zhang, F. 2022. Multi-site MRI classification using Weighted federated learning based on Mixture of Experts domain adaptation. In *2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, 916–921. IEEE.
- Bonawitz, K.; Eichner, H.; Grieskamp, W.; Huba, D.; Ingerman, A.; Ivanov, V.; Kiddon, C.; Konecnˇ y, J.; Mazzocchi, ` S.; McMahan, B.; et al. 2019. Towards federated learning at scale: System design. *Proceedings of machine learning and systems*, 1: 374–388.
- Cai, R.; Chen, X.; Liu, S.; Srinivasa, J.; Lee, M.; Kompella, R.; and Wang, Z. 2023. Many-task federated learning: A new problem setting and a simple baseline. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 5037–5045.
- Chen, T.; Chen, X.; Du, X.; Rashwan, A.; Yang, F.; Chen, H.; Wang, Z.; and Li, Y. 2023a. Adamv-moe: Adaptive multi-task vision mixture-of-experts. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 17346–17357.
- Chen, Z.; Shen, Y.; Ding, M.; Chen, Z.; Zhao, H.; Learned-Miller, E. G.; and Gan, C. 2023b. Mod-squad: Designing mixtures of experts as modular multi-task learners. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 11828–11837.
- Fedus, W.; Dean, J.; and Zoph, B. 2022. A review of sparse expert models in deep learning. *arXiv preprint arXiv:2209.01667*.
- Fedus, W.; Zoph, B.; and Shazeer, N. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120): 1–39.
- Guo, B.; Mei, Y.; Xiao, D.; and Wu, W. 2021. PFL-MoE: personalized federated learning based on mixture of experts. In *Web and Big Data: 5th International Joint Conference, APWeb-WAIM 2021, Guangzhou, China, August 23– 25, 2021, Proceedings, Part I 5*, 480–486. Springer.
- Gupta, S.; Mukherjee, S.; Subudhi, K.; Gonzalez, E.; Jose, D.; Awadallah, A. H.; and Gao, J. 2022. Sparsely activated mixture-of-experts are robust multi-task learners. *arXiv preprint arXiv:2204.07689*.
- Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2021. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*.
- Ilhan, F.; Su, G.; and Liu, L. 2023. Scalefl: Resourceadaptive federated learning with heterogeneous clients. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 24532–24541.
- Jiang, A. Q.; Sablayrolles, A.; Roux, A.; Mensch, A.; Savary, B.; Bamford, C.; Chaplot, D. S.; Casas, D. d. l.; Hanna, E. B.; Bressand, F.; et al. 2024. Mixtral of experts. *arXiv preprint arXiv:2401.04088*.

- Kairouz, P.; McMahan, H. B.; Avent, B.; Bellet, A.; Bennis, M.; Bhagoji, A. N.; Bonawitz, K.; Charles, Z.; Cormode, G.; Cummings, R.; et al. 2021. Advances and open problems in federated learning. *Foundations and trends® in machine learning*, 14(1–2): 1–210.
- Kaplan, J.; McCandlish, S.; Henighan, T.; Brown, T. B.; Chess, B.; Child, R.; Gray, S.; Radford, A.; Wu, J.; and Amodei, D. 2020. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*.
- Karimireddy, S. P.; Kale, S.; Mohri, M.; Reddi, S.; Stich, S.; and Suresh, A. T. 2020. Scaffold: Stochastic controlled averaging for federated learning. In *International conference on machine learning*, 5132–5143. PMLR.
- Kim, Y. J.; Awan, A. A.; Muzio, A.; Salinas, A. F. C.; Lu, L.; Hendy, A.; Rajbhandari, S.; He, Y.; and Awadalla, H. H. 2021. Scalable and efficient moe training for multitask multilingual models. *arXiv preprint arXiv:2109.10465*.
- Krajewski, J.; Ludziejewski, J.; Adamczewski, K.; Pioro, ´ M.; Krutul, M.; Antoniak, S.; Ciebiera, K.; Krol, K.; ´ Odrzygo´zd´ z, T.; Sankowski, P.; et al. 2024. Scaling ´ laws for fine-grained mixture of experts. *arXiv preprint arXiv:2402.07871*.
- Li, T.; Sahu, A. K.; Zaheer, M.; Sanjabi, M.; Talwalkar, A.; and Smith, V. 2020. Federated optimization in heterogeneous networks. *Proceedings of Machine learning and systems*, 2: 429–450.
- Lim, W. Y. B.; Luong, N. C.; Hoang, D. T.; Jiao, Y.; Liang, Y.-C.; Yang, Q.; Niyato, D.; and Miao, C. 2020. Federated learning in mobile edge networks: A comprehensive survey. *IEEE communications surveys & tutorials*, 22(3): 2031–2063.
- Lin, B. Y.; He, C.; Zeng, Z.; Wang, H.; Huang, Y.; Dupuy, C.; Gupta, R.; Soltanolkotabi, M.; Ren, X.; and Avestimehr, S. 2021. Fednlp: Benchmarking federated learning methods for natural language processing tasks. *arXiv preprint arXiv:2104.08815*.
- Lin, T.; Kong, L.; Stich, S. U.; and Jaggi, M. 2020. Ensemble distillation for robust model fusion in federated learning. *Advances in neural information processing systems*, 33: 2351–2363.
- Liu, Y.; Zhang, K.; Li, Y.; Yan, Z.; Gao, C.; Chen, R.; Yuan, Z.; Huang, Y.; Sun, H.; Gao, J.; et al. 2024. Sora: A review on background, technology, limitations, and opportunities of large vision models. *arXiv preprint arXiv:2402.17177*.
- Lu, X.; Liu, Q.; Xu, Y.; Zhou, A.; Huang, S.; Zhang, B.; Yan, J.; and Li, H. 2024. Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models. *arXiv preprint arXiv:2402.14800*.
- Ma, X.; Zhang, J.; Guo, S.; and Xu, W. 2022. Layer-wised model aggregation for personalized federated learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 10092–10101.
- Mei, Y.; Guo, B.; Xiao, D.; and Wu, W. 2021. Fedvf: Personalized federated learning based on layer-wise parameter updates with variable frequency. In *2021 IEEE International Performance, Computing, and Communications Conference (IPCCC)*, 1–9. IEEE.

- Narayan, S.; Cohen, S. B.; and Lapata, M. 2018. Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. *arXiv preprint arXiv:1808.08745*.
- Ni, X.; Shen, X.; and Zhao, H. 2022. Federated optimization via knowledge codistillation. *Expert Systems with Applications*, 191: 116310.
- Ozkara, K.; Singh, N.; Data, D.; and Diggavi, S. 2021. Quped: Quantized personalization via distillation with applications to federated learning. *Advances in Neural Information Processing Systems*, 34: 3622–3634.
- Rajbhandari, S.; Li, C.; Yao, Z.; Zhang, M.; Aminabadi, R. Y.; Awan, A. A.; Rasley, J.; and He, Y. 2022. Deepspeedmoe: Advancing mixture-of-experts inference and training to power next-generation ai scale. In *International conference on machine learning*, 18332–18346. PMLR.
- Rajpurkar, P.; Zhang, J.; Lopyrev, K.; and Liang, P. 2016. Squad: 100,000+ questions for machine comprehension of text. *arXiv preprint arXiv:1606.05250*.
- Sabah, F.; Chen, Y.; Yang, Z.; Azam, M.; Ahmad, N.; and Sarwar, R. 2023. Model optimization techniques in personalized federated learning: A survey. *Expert Systems with Applications*, 122874.
- Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; Le, Q.; Hinton, G.; and Dean, J. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*.
- Sun, Z.; Yang, H.; Liu, K.; Yin, Z.; Li, Z.; and Xu, W. 2022. Recent advances in LoRa: A comprehensive survey. *ACM Transactions on Sensor Networks*, 18(4): 1–44.
- Tan, A. Z.; Yu, H.; Cui, L.; and Yang, Q. 2022. Towards personalized federated learning. *IEEE transactions on neural networks and learning systems*, 34(12): 9587–9603.
- Vanhaesebrouck, P.; Bellet, A.; and Tommasi, M. 2017. Decentralized collaborative learning of personalized models over networks. In *Artificial Intelligence and Statistics*, 509– 517. PMLR.
- Villalobos, P.; Sevilla, J.; Heim, L.; Besiroglu, T.; Hobbhahn, M.; and Ho, A. 2022. Will we run out of data? an analysis of the limits of scaling datasets in machine learning. *arXiv preprint arXiv:2211.04325*.
- Wang, J.; Yang, X.; Cui, S.; Che, L.; Lyu, L.; Xu, D. D.; and Ma, F. 2024. Towards personalized federated learning via heterogeneous model reassembly. *Advances in Neural Information Processing Systems*, 36.
- Wang, K.; He, Q.; Chen, F.; Chen, C.; Huang, F.; Jin, H.; and Yang, Y. 2023. Flexifed: Personalized federated learning for edge clients with heterogeneous model architectures. In *Proceedings of the ACM Web Conference 2023*, 2979–2990.
- Wikipedia. 2024. General Data Protection Regulation Wikipedia, The Free Encyclopedia. [Online; accessed 27- June-2024].
- Wikipedia contributors. 2024. California Consumer Privacy Act — Wikipedia, The Free Encyclopedia. [Online; accessed 27-June-2024].

- Wolf, T.; Debut, L.; Sanh, V.; Chaumond, J.; Delangue, C.; Moi, A.; Cistac, P.; Rault, T.; Louf, R.; Funtowicz, M.; et al. 2019. Huggingface's transformers: State-of-the-art natural language processing. *arXiv preprint arXiv:1910.03771*.
- Wu, J.; Liu, Q.; Huang, Z.; Ning, Y.; Wang, H.; Chen, E.; Yi, J.; and Zhou, B. 2021. Hierarchical personalized federated learning for user modeling. In *Proceedings of the Web Conference 2021*, 957–968.
- Wu, T.; He, S.; Liu, J.; Sun, S.; Liu, K.; Han, Q.-L.; and Tang, Y. 2023. A brief overview of ChatGPT: The history, status quo and potential future development. *IEEE/CAA Journal of Automatica Sinica*, 10(5): 1122–1136.
- Xu, M.; Cai, D.; Wu, Y.; Li, X.; and Wang, S. 2024. {FwdLLM}: Efficient Federated Finetuning of Large Language Models with Perturbed Inferences. In *2024 USENIX Annual Technical Conference (USENIX ATC 24)*, 579–596.
- Yao, L.; Gao, D.; Wang, Z.; Xie, Y.; Kuang, W.; Chen, D.; Wang, H.; Dong, C.; Ding, B.; and Li, Y. 2022. A benchmark for federated hetero-task learning. *arXiv preprint arXiv:2206.03436*.
- Yao, X.; and Sun, L. 2020. Continual local training for better initialization of federated models. In *2020 IEEE International Conference on Image Processing (ICIP)*, 1736–1740. IEEE.
- Yi, R.; Guo, L.; Wei, S.; Zhou, A.; Wang, S.; and Xu, M. 2023. Edgemoe: Fast on-device inference of moe-based large language models. *arXiv preprint arXiv:2308.14352*.
- Zec, E. L.; Mogren, O.; Martinsson, J.; Sutfeld, L. R.; and ¨ Gillblad, D. 2020. Specialized federated learning using a mixture of experts. *arXiv preprint arXiv:2010.02056*.
- Zhang, C.; Meng, X.; Liu, Q.; Wu, S.; Wang, L.; and Ning, H. 2023. FedBrain: A robust multi-site brain network analysis framework based on federated learning for brain disease diagnosis. *Neurocomputing*, 559: 126791.
- Zhang, X.; Zhao, J.; and LeCun, Y. 2015. Character-level convolutional networks for text classification. *Advances in neural information processing systems*, 28.
- Zhang, Z.; Liu, S.; Yu, J.; Cai, Q.; Zhao, X.; Zhang, C.; Liu, Z.; Liu, Q.; Zhao, H.; Hu, L.; et al. 2024. M3oE: Multi-Domain Multi-Task Mixture-of Experts Recommendation Framework. *arXiv preprint arXiv:2404.18465*.
- Zhu, H.; Xu, J.; Liu, S.; and Jin, Y. 2021. Federated learning on non-IID data: A survey. *Neurocomputing*, 465: 371–390. Zhu, Z.; Hong, J.; Drew, S.; and Zhou, J. 2022. Resilient and communication efficient learning for heterogeneous federated systems. *Proceedings of machine learning research*, 162: 27504.
- Zou, X.; Hu, Z.; Zhao, Y.; Ding, X.; Liu, Z.; Li, C.; and Sun, A. 2022. Automatic expert selection for multi-scenario and multi-task search. In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*, 1535–1544.