# <span id="page-0-1"></span>DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models

Yongqi Huang<sup>1</sup><sup>∗</sup> Peng Ye<sup>2</sup>,3<sup>∗</sup> Chenyu Huang<sup>1</sup> Jianjian Cao<sup>1</sup> Lin Zhang<sup>1</sup> Baopu Li<sup>4</sup> Gang Yu<sup>5</sup> Tao Chen<sup>1</sup>† <sup>1</sup> School of Information Science and Technology, Fudan University <sup>2</sup> The Chinese University of Hong Kong <sup>3</sup> Shanghai AI Laboratory <sup>4</sup> Baidu USA <sup>5</sup> StepFun

yqhuang23@m.fudan.edu.cn, eetchen@fudan.edu.cn

# Abstract

*Upcycled Mixture-of-Experts (MoE) models have shown great potential in various tasks by converting the original Feed-Forward Network (FFN) layers in pre-trained dense models into MoE layers. However, these models still suffer from significant parameter inefficiency due to the introduction of multiple experts. In this work, we propose a novel DeRS (Decompose, Replace, and Synthesis) paradigm to overcome this shortcoming, which is motivated by our observations about the unique redundancy mechanisms of upcycled MoE experts. Specifically, DeRS decomposes the experts into one expert-shared base weight and multiple expert-specific delta weights, and subsequently represents these delta weights in lightweight forms. Our proposed DeRS paradigm can be applied to enhance parameter efficiency in two different scenarios, including: 1) DeRS Compression for inference stage, using sparsification or quantization to compress vanilla upcycled MoE models; and 2) DeRS Upcycling for training stage, employing lightweight sparse or low-rank matrixes to efficiently upcycle dense models into MoE models. Extensive experiments across three different tasks show that the proposed methods can achieve extreme parameter efficiency while maintaining the performance for both training and compression of upcycled MoE models.*

# 1. Introduction

Recently, sparse Mixture-of-Experts [\[44\]](#page-9-0) (MoE) models have made remarkable progress [\[1,](#page-8-0) [8,](#page-8-1) [19,](#page-8-2) [23,](#page-9-1) [31,](#page-9-2) [49\]](#page-10-0). These models introduce the MoE layer, which consists of multiple Feed-Forward Network (FFN) experts and a learnable router, to replace a single FFN layer. By dynamically activating only a subset of experts for each input, these MoE models achieve superior performance while maintaining computational efficiency. However, training MoE models from scratch requires substantial computational resources [\[8,](#page-8-1) [9,](#page-8-3) [43\]](#page-9-3) and often en-

<span id="page-0-0"></span>![](_page_0_Figure_9.jpeg)

Figure 1. Visualization of cosine similarity in the first and last MoE layers of the MoE-LLaVA-Phi [\[30\]](#page-9-4) model. *FFN* denotes the initial weight while E<sup>i</sup> denotes the trained weight of the i-th expert.

counters training instabilities [\[41,](#page-9-5) [53\]](#page-10-1).

To address these challenges, upcycling pre-trained dense models into MoE models (referred to as upcycling) [\[21\]](#page-8-4) has emerged as an effective alternative. This approach converts the original FFN layers in a pre-trained dense model into MoE layers with N experts, initializing each expert from the original FFN weight while introducing a randomly initialized router for expert selection. By leveraging the knowledge embedded in pre-trained dense models, upcycled MoE models facilitate more efficient optimization and demonstrate competitive performance under constrained training budgets [\[14,](#page-8-5) [30,](#page-9-4) [47\]](#page-10-2). Due to its simplicity and effectiveness, upcycling has been widely applied across diverse domains, including natural language processing [\[6,](#page-8-6) [14,](#page-8-5) [21\]](#page-8-4), computer vision [\[13,](#page-8-7) [21,](#page-8-4) [24\]](#page-9-6), and multi-modal learning [\[20,](#page-8-8) [27,](#page-9-7) [30\]](#page-9-4).

While upcycled MoE models have shown significant success, they still suffer from parameter inefficiency due to the large number of extra parameters introduced by MoE experts. For instance, the upcycled MoE-LLaVA-Phi [\[30\]](#page-9-4) model comprises a total of 5 billion parameters, of which 3.4 billion are occupied by MoE experts. In this paper, we focus on enhancing the parameter efficiency of upcycled MoE models. We first revisit the characteristics of the N experts in upcycled MoE models: 1) We note that N experts share the same initial weight Wbase, and the weights of trained experts can be uniformly expressed as W<sup>i</sup> = Wbase + ∆<sup>i</sup> , where i = 1, .., N; 2) We observe that, there is an extremely high

<sup>†</sup>Corresponding author. <sup>∗</sup>Equal contribution.

<span id="page-1-0"></span>cosine similarity (higher than 0.999) between the expert's weight and the initial weight, as well as among the weights of different experts, as shown in Fig. [1.](#page-0-0) This extremely high cosine similarity suggests that ∆<sup>i</sup> is a minor adjustment to Wbase and may exhibit considerable redundancy.

Motivated by these observations, we propose a novel *Decompose, Replace and Synthesis* (DeRS) paradigm that remodels the weights of upcycled MoE experts {W1, . . . , W<sup>N</sup> } into one expert-shared base weight Wbase and multiple expert-specific delta weights {∆1, . . . , ∆<sup>N</sup> }, enhancing the expert parameter efficiency by applying lightweight representations to these delta weights. Building on the DeRS paradigm, we further design two distinct approaches for upcycled MoE models at different phases: DeRS compression for inference and DeRS upcycling for training. Specifically, DeRS compression employs sparsification or quantization techniques to compress the delta weights of already-trained experts, thereby achieving inference-time parameter efficiency for vanilla upcycled MoE models. On the other hand, DeRS upcycling enables the training of experts in a parameter-efficient way by training only one expertshared FFN weight and multiple sparse or low-rank weights, thus greatly reducing trainable parameters.

We conduct comprehensive experiments to verify the effectiveness of the proposed DeRS compression and DeRS upcycling methods across three upcycled MoE baselines encompassing general multi-modal tasks, medical multi-modal tasks and code generation tasks, as well as six different MoE model architectures. Experimental results demonstrate that our proposed methods bring extreme efficiency to upcycled MoE models. For example, on the general multi-modal task, our DeRS compression effectively reduces the parameter count of a MoE layer by 65% without a performance drop, and our DeRS upcycling method achieves a reduction of up to 2270 times in additional parameters while delivering better performance compared to vanilla upcycling.

Our contributions are as follows:

- We are the first to explore the unique redundancy mechanisms of experts in upcycled MoE models, and propose a novel DeRS paradigm that decomposes multiple experts into one expert-shared weight and multiple expert-specific weights to reduce parameter redundancy.
- Based on our DeRS paradigm, we further propose two application methods to achieve extremely efficient upcycled MoE: 1) DeRS compression to efficiently compress vanilla upcycled MoE models for inference. 2) DeRS upcycling to efficiently upcycle a pre-trained dense model into a MoE model for training and deployment.
- Comprehensive experiments across three tasks and six MoE model architectures consistently verify the efficiency, effectiveness, and generalizability of the proposed DeRS compression and DeRS upcycling techniques.

# 2. Related Work

### 2.1. Training of MoE Models

For MoE models, the straightforward training strategy [\[7,](#page-8-9) [8,](#page-8-1) [23,](#page-9-1) [48\]](#page-10-3) generally involves designing a MoE architecture, randomly initializing the MoE model weights, and training the model with extensive computational resources and data. Recently, upcycling [\[21\]](#page-8-4) has been proposed to reduce training costs by initializing MoE models using pre-trained dense models. Due to its simplicity and effectiveness, upcycling has found widespread application across various domains [\[6,](#page-8-6) [20,](#page-8-8) [24,](#page-9-6) [30\]](#page-9-4). For instance, XFT [\[6\]](#page-8-6) employs upcycling in the instruction tuning process of large language models, achieving performance improvement on code generation tasks. In the field of multi-modal learning, MoE-LLaVA [\[30\]](#page-9-4) successfully trains a vision-language MoE model with only 3B active parameters through upcycling, achieving performance comparable to LLaVA-1.5-7B [\[35\]](#page-9-8) on various visual understanding benchmarks. In this paper, we first find the unique redundancy of expert parameters in vanilla upcycling, and further propose the DeRS upcycling method, which significantly reduces the parameter count of experts in upcycled MoE models during both training and inference.

### 2.2. Compression of MoE Models

To date, many studies have explored the compression of experts in MoE models to improve deployment efficiency [\[15,](#page-8-10) [28,](#page-9-9) [33,](#page-9-10) [39\]](#page-9-11). Specifically, [\[39\]](#page-9-11) proposes a plugand-play expert-level sparsification technique by pruning unimportant experts and dynamically skipping specific experts during inference. EEP [\[33\]](#page-9-10) employs an evolutionary strategy to search the pattern for expert removal and consolidates the knowledge of removed experts into retained experts through weight merging. MC-SMoE [\[28\]](#page-9-9) first merges infrequently activated experts into those often activated experts, and then compresses the merged experts. In this paper, we also propose a novel expert compression method called DeRS compression, which focuses on compressing MoE models upcycled from pre-trained dense models. Leveraging the characteristics of upcycled MoE models, our DeRS compression innovatively decomposes the weights of multiple experts into an expert-shared important weight and multiple expert-specific redundant weights, and applies sparsification or quantization to those redundant weights.

# 3. Method

### 3.1. Preliminary

Mixture-of-Experts. A standard MoE layer consists of a set of N experts (i.e., N FFN layers) {E1, E2, . . . , E<sup>N</sup> } and a router network R with trainable weight WR. Given an input x, the output y of the MoE layer can be written as:

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 2. Overall procedure of (a) upcycling a dense model into a MoE model through vanilla upcycling and (b) compressing the vanilla upcycled MoE model using the proposed DeRS compression, which first decomposes the trained experts and then applies sparsification or quantization techniques to the expert-specific delta weights. During inference, when an expert is needed, we synthesize its weight online.

Replace (b) DeRS Compression

$$y = \sum_{i=1}^{N} R(x)_i \cdot E_i(x) \tag{1}$$

$$R(x) = TopK(softmax(x \cdot W_R), k)$$

where  $R(x)_i$  denotes the routing score to the *i*-th expert,  $E_i(x)$  stands for the output of i-th expert,  $TopK(\cdot, k)$  means selecting only the top-k experts by setting R(x) of other experts to 0. Usually,  $k \ll N$ , which means R(x) a sparse N-dimensional vector. When  $R(x)_i = 0$ ,  $E_i(x)$  does not need to be computed.

**Upcycled Mixture-of-Experts**. Let  $W_{base} \in \mathbb{R}^{d \times d_h}$  denote the weight of original FFN layer in the pre-trained dense model. For the corresponding upcycled MoE layer with Nexperts, the weight of each expert is uniformly initialized to  $W_{base}$  instead of random initialization. After training, the weights of these N experts are updated to  $\{W_1, \ldots, W_N\}$ .

# 3.2. Decompose, Replace, and Synthesis

In this paper, we revisit the Upcycled Mixture-of-Experts and remodel it according to the following three aspects: Decompose, Replace, and Synthesize.

**Decompose.** Since all N experts share the same initial weight  $W_{base}$ , the trained weight of a specific expert  $E_i$  can be expressed as:

$$W_i = W_{base} + \Delta_i \tag{2}$$

This implies that  $\{W_1, \dots, W_N\}$  can be decomposed into an expert-shared base weight  $W_{base}$  and N expert-specific delta weights  $\{\Delta_1, \ldots, \Delta_N\}$ .

**Replace.** Since  $W_{base}$  is pre-trained knowledge learned

from extensive data, the expert-specific delta weight  $\Delta_i$  is actually minor adjustment to  $W_{base}$ . As shown in Fig. 1, the extremely high cosine similarity among the trained experts and the original FFN further supports this opinion, suggesting that these N expert-specific delta weights  $\{\Delta_1, \ldots, \Delta_N\}$ may exhibit some degree of redundancy. Based on this hypothesis, it is feasible to adopt a lightweight representation  $\mathcal{F}(\Delta_i)$  to reformulate the original redundant  $\Delta_i$  without a performance drop. In other words, we can replace the original heavyweight set  $\{W_1, \dots, W_N\}$  with a more parameterefficient set  $\{\mathcal{F}(\Delta_1), \ldots, \mathcal{F}(\Delta_N)\} \cup \{W_{base}\}.$ 

**Synthesis** 

**Synthesis.** Based on  $\{\mathcal{F}(\Delta_1), \ldots, \mathcal{F}(\Delta_N)\}$  and  $W_{base}$ , we synthesize the weight of a specific expert  $E_i$  by fusing the lightweight expert-specific delta weight  $\mathcal{F}(\Delta_i)$  with the expert-shared base weight  $W_{base}$  as follows:

$$\hat{W}_i = W_{base} + \mathcal{F}(\Delta_i) \tag{3}$$

Based on the above framework of *Decompose*, *Replace* and Synthesis (DeRS), we propose two application methods: DeRS compression and DeRS upcycling. DeRS compression focuses on compressing already-trained vanilla upcycled MoE models, while DeRS upcycling is designed to efficiently upcycle existing dense models into MoE architectures for subsequent training and deployment.

### 3.3. DeRS Compression

As shown in Fig. 2, applying DeRS compression to compress an already-trained vanilla upcycled MoE model follows three steps: 1) decomposing the weights of N trained experts into  $W_{base}$  and  $\{\Delta_i\}_{i=1}^N$ ; 2) utilizing a post-training lightweight

<span id="page-3-1"></span><span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 3. Comparisons between vanilla upcycling and the proposed DeRS upcycling in the construction of experts. Instead of making N copies of the original FFN, our DeRS upcycling synthesizes experts by combining the shared FFN with expert-specific lightweight parameters (i.e., in the form of sparse matrixes or low-rank matrixes).

technique  $\mathcal{F}_{post}$  to compress  $\{\Delta_i\}_{i=1}^N$  into more efficient  $\{\mathcal{F}_{post}(\Delta_i)\}_{i=1}^N$ ; and 3) when a specific expert  $E_i$  is needed, synthesizing its weight by summing  $\mathcal{F}_{post}(\Delta_i)$  and  $W_{base}$ . Two different compression designs are proposed as follows:

**Sparsification.** A straightforward implementation is to remove unnecessary elements from the weight matrix. Inspired by [50], we randomly drop most elements of the expert-specific delta weight  $\Delta_i$  and compactly store the obtained sparse matrix  $\mathcal{F}_{post}(\Delta_i)$  as vectors. Given a drop rate p,  $\mathcal{F}_{post}(\Delta_i)$  can be written as:

$$M_i \sim \mathrm{Bernoulli}(p)$$
 
$$\mathcal{F}_{post}(\Delta_i) = (\mathbf{1} - M_i) \odot \Delta_i \ / \ (1-p) \eqno(4)$$

Although the theoretical shape of a sparse weight matrix remains  $\mathbb{R}^{d \times d_h}$ , the actual parameter count of  $\mathcal{F}_{post}(\Delta_i)$  is only  $d \cdot d_h \cdot (1-p)$  due to its compact storage format. In other words, by applying DeRS-Sparsification with the given drop rate p, the parameter count of N trained experts is reduced from the original  $N \cdot d \cdot d_h$  to  $(1+N \cdot (1-p)) \cdot d \cdot d_h$ .

**Quantization.** Additionally, we provide another approach to reduce redundancy by quantizing  $\Delta_i$  to a lower k-bit representation:

$$\mathcal{F}_{nost}(\Delta_i) = Quant(\Delta_i, k) \tag{5}$$

Assuming the original expert weights  $\{W_i\}_{i=1}^N$  are represented with K bits, applying DeRS-Quantization can reduce the parameter storage cost from  $N \cdot K$  to  $K + N \cdot k$ .

### 3.4. DeRS Upcycling

The difference between our proposed DeRS upcycling and vanilla upcycling is presented in Fig. 3. Instead of replicating the original FFN layer N times to form N experts of a MoE layer, we decompose the construction of N experts before training. Specifically, we decompose N expert weights into one trainable expert-shared weight  $W_{shared}$  and N trainable expert-specific incremental weights  $\{\mathcal{F}_{pre}(\Delta_i)\}_{i=1}^N$ , using a parameter-efficient design  $\mathcal{F}_{pre}$ . During both training and inference, the weight  $W_i$  of expert  $E_i$  is synthesized by combining  $W_{shared}$  and  $\mathcal{F}_{pre}(\Delta_i)$  via addition. Here,  $W_{shared}$  is initialized from the weight of the original FFN

layer and  $\mathcal{F}_{pre}(\Delta_i)$  is zero-initialized. We provide two types of parameter-efficient designs  $\mathcal{F}_{pre}$  as follows:

**Sparse Matrix**. Since sparsification can be used to reduce redundancy in our DeRS compression, it's feasible to adopt sparse matrixes as a parameter-efficient form of the trainable expert-specific incremental weights. However, simply employing a vanilla sparse matrix implementation, where the matrix maintains the shape of  $\mathbb{R}^{d \times d_h}$  and most of the element values are zero, can't effectively reduce the number of parameters in practice.

To address this, we reformulate the sparse matrix with a shape of  $\mathbb{R}^{d \times d_h}$  and a sparse rate p into two compact row vectors of length  $d \cdot d_h \cdot (1-p)$ , specifically an index vector I and a value vector V. As shown in Fig. 3(b), the value vector V stores the values of all nonzero elements, while the index vector I stores the positions of these nonzero elements in the original-shaped sparse matrix. These two vectors I and V are mapped back to the sparse matrix using the torch.scatter function. Based on the above efficient implementation of the sparse matrix, we propose the first design of expert-specific incremental weights:

$$\mathcal{F}_{pre}(\Delta_i) = \text{torch.scatter}(I_i, V_i)$$
 (6)

Given a sparse rate p, the index vector  $I_i$  is generated by randomly selecting  $d \cdot d_h \cdot (1-p)$  unique values from the range  $[0, d \cdot d_h)$  and is fixed thereafter, while  $V_i$  serves as the trainable parameter and is initialized to zero.

In terms of the number of trainable expert parameters, vanilla upcycling requires training N expert weights of shape  $\mathbb{R}^{d \times d_h}$ , while the proposed Sparse-Matrix-based DeRS (DeRS-SM) upcycling only necessitates training one  $W_{shared}$  of shape  $\mathbb{R}^{d \times d_h}$  along with N row vectors of length  $d \cdot d_h \cdot (1-p)$ . In other words, our DeRS-SM upcycling decreases the trainable parameter count of experts from  $N \cdot d \cdot d_h$  to  $(1+N \cdot (1-p)) \cdot d \cdot d_h$ .

**Low-rank Matrix**. Since representing the expert-specific incremental weight using the high-dimensional space  $\mathbb{R}^{d\times d_h}$  is redundant, it's sufficient to utilize two low-rank matrixes  $A\in\mathbb{R}^{d\times r}$  and  $B\in\mathbb{R}^{r\times d_h}$  for an efficient representation based on the low-rank matrix factorization. Thus, we develop

<span id="page-4-2"></span><span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 4. Performance of applying different DeRS compression methods to compress three vanilla upcycled MoE-LLaVA models respectively.

<span id="page-4-1"></span>Table 1. Performance comparison between vanilla upcycling and our DeRS upcycling on three MoE-LLaVA models on the general multi-modal task. DeRS-SM and DeRS-LM denote the Sparse-Matrix-based and Low-rank-Matrix-based DeRS upcycling respectively. **Added Params** represents the number of additional parameters of the upcycled MoE model compared to its corresponding dense model.

| MoE Model               | Upcycling      | Added   | I                 | mage Q | uestion A | nswering         | g       | Ben  | chmark ' | Toolkit | Overall |
|-------------------------|----------------|---------|-------------------|--------|-----------|------------------|---------|------|----------|---------|---------|
| WIOE WIOGEI             | Method         | Params. | VQA <sup>v2</sup> | GQA    | VisWiz    | SQA <sup>I</sup> | $VQA^T$ | POPE | MMB      | MM-Vet  | Overall |
|                         | Vanilla        | 1.24B   | 76.3              | 60.6   | 34.6      | 62.7             | 50.2    | 86.9 | 60.5     | 27.1    | 57.4    |
| MoE-LLaVA-StableLM [30] | DeRS-SM (ours) | 0.26M   | 76.5              | 60.7   | 34.9      | 62.9             | 50.2    | 87.3 | 60.4     | 28.4    | 57.7    |
|                         | DeRS-LM (ours) | 1.20M   | 76.6              | 60.6   | 34.4      | 62.3             | 50.0    | 87.1 | 60.1     | 28.6    | 57.5    |
|                         | Vanilla        | 1.22B   | 76.2              | 61.2   | 31.8      | 62.4             | 48.1    | 87.5 | 60.4     | 24.4    | 56.5    |
| MoE-LLaVA-Qwen [30]     | DeRS-SM (ours) | 0.26M   | 76.2              | 61.2   | 31.0      | 62.2             | 47.8    | 87.8 | 60.0     | 23.7    | 56.2    |
|                         | DeRS-LM (ours) | 1.19M   | 76.2              | 61.1   | 31.2      | 62.1             | 47.9    | 87.8 | 60.0     | 24.2    | 56.3    |
| MoE-LLaVA-Phi [30]      | Vanilla        | 2.52B   | 77.5              | 61.4   | 42.7      | 68.6             | 50.8    | 86.9 | 66.0     | 32.4    | 60.8    |
|                         | DeRS-SM (ours) | 1.11M   | 77.7              | 61.3   | 42.4      | 69.0             | 51.7    | 87.3 | 65.5     | 33.8    | 61.1    |
|                         | DeRS-LM (ours) | 2.42M   | 77.6              | 61.3   | 42.0      | 69.1             | 51.8    | 87.4 | 66.3     | 32.6    | 61.0    |

the Low-rank Matrix-based DeRS (DeRS-LM) upcycling . As shown in Fig. 3(c), the expert-specific incremental weight can be expressed as  $\mathcal{F}_{pre}(\Delta_i) = A_i \cdot B_i$ , where  $A_i$  is randomly initialized and  $B_i$  is zero-initialized.

Given the rank  $r, r \ll min(d, d_h)$ , our DeRS-LM upcycling reduces the number of trainable expert parameters from  $N \cdot d \cdot d_h$  to  $d \cdot d_h + N \cdot r \cdot (d + d_h)$ .

#### 4. Experiments

In this section, we conduct a series of experiments to evaluate our DeRS compression and DeRS upcycling methods across a range of tasks, including general multi-modal tasks, medical multi-modal tasks, and code generation tasks.

#### 4.1. General Multi-modal Task

Model architecture. We adopt the MoE-LLaVA [30] framework for the experiments of the general multi-modal task. We employ CLIP-Large [42] as the visual encoder. Three upcycled MoE models, respectively initialized from StableLM-

2-1.6B [4], Qwen-1.8B [3] and Phi-2-2.7B [18], serve as the language backbone. Before upcycling, all the dense models have been previously fine-tuned on datasets collected from MIMIC-IT [25], LRV [34], SVIT [52] and LVIS [46]. In each dense model, every other block's FFN layer is upcycled into a MoE layer with 4 experts, where the top-2 experts are dynamically activated when processing each input.

**Datasets.** The upcycled MoE models undergo fine-tuning on the LLaVA-mix-665k [35] dataset, followed by evaluation across five image question-answering benchmarks: VQA-v2 [10], GQA [17], VisWiz [12], ScienceQA-IMG [38] and TextVQA [45]. Besides, three benchmark toolkits, including POPE [29], MMBench [37] and MM-Vet [51], are adopted to evaluate the multi-modal understanding capabilities.

**Results of DeRS Compression.** The experimental results of DeRS compression are shown in Fig. 4. As we can see, the proposed two compression techniques, sparsification and quantization, can effectively reduce the parameter redundancy while not affecting the model's performance. Specifically, as shown in Fig. 4a, applying DeRS-Sparsification

<span id="page-5-2"></span><span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 5. Performance of applying different DeRS compression methods to compress two vanilla upcycled Med-MoE models respectively.

<span id="page-5-1"></span>Table 2. Performance comparison between vanilla upcycling and our DeRS upcycling on two Med-MoE models on the medical multi-modal task. DeRS-SM and DeRS-LM denote the Sparse-Matrix-based and Low-rank-Matrix-based DeRS upcycling respectively. The light-gray Added Params denotes the additional parameters introduced by the universal FFN layers that are not considered as experts of MoE layers.

| MoE Model             | Upcycling              | Added               | VQA  | -RAD   | SL   | AKE    | Path | VQA    | Overall |
|-----------------------|------------------------|---------------------|------|--------|------|--------|------|--------|---------|
| MOE Model             | Method                 | Params.             | Open | Closed | Open | Closed | Open | Closed | Overall |
| Med-MoE-StableLM [20] | Vanilla                | 0.42B <b>+1.24B</b> | 51.0 | 82.3   | 82.4 | 85.3   | 33.4 | 91.4   | 71.0    |
|                       | DeRS-SM (ours)         | 0.42B + 0.29M       | 49.5 | 84.2   | 83.8 | 84.9   | 33.1 | 91.6   | 71.2    |
| (EMNLP 24)            | DeRS-LM (ours) 0.42B+1 | 0.42B <b>+1.20M</b> | 53.5 | 81.6   | 83.7 | 84.1   | 33.6 | 91.5   | 71.3    |
| Med-MoE-Phi [20]      | Vanilla                | 0.84B+2.52B         | 55.1 | 85.3   | 84.6 | 85.8   | 35.1 | 91.5   | 72.9    |
| (EMNLP 24)            | DeRS-SM (ours)         | 0.84B + 1.11M       | 55.5 | 85.3   | 84.3 | 86.3   | 35.0 | 91.6   | 73.0    |
|                       | DeRS-LM (ours)         | 0.84B+2.42M         | 55.7 | 84.9   | 84.3 | 85.6   | 35.2 | 91.6   | 72.9    |

to remove 90% of the elements in the delta weights hardly affects the model's performance, but the parameter count of a MoE layer can be reduced from the original 4 experts' parameters to the equivalent of  $(1+4\times0.1)$  experts' parameters. Similarly, we can quantize the expert-specific delta weights from the original 16 bits down to 2 bits using DeRS-Quantization, reducing the parameter storage cost of a MoE layer's 4 experts from original  $4\times16$  to  $16+4\times2$ , while demonstrating the same or even better performance.

Moreover, we can observe that even under extreme DeRS compression settings, such as sparsification with a 0.99 drop rate or quantization with the 1-bit width, there is no significant degradation in the model's performance. This may be attributed to the fact that the three dense models used in the MoE-LLaVA framework have been previously fine-tuned on some multi-modal data. In other words, the expert-shared base weight obtained through decomposition during DeRS compression already has a certain degree of multi-modal understanding capability. As a result, even with the extreme compression of the redundant expert-specific delta weights, the model's performance remains robust.

Results of DeRS Upcycling. Tab. 1 shows the performance of different upcycling methods on three MoE-LLaVA architectures. As we can see, vanilla upcycling results in billions of additional parameters by duplicating the original FFN layer to construct experts, while our DeRS upcycling introduces only millions of extra parameters to achieve the same or superior performance. For example, when upcycling Phi-2-2.7B into the MoE-LLaVA-Phi architecture, vanilla upcycling adds 2.52 billion parameters. In comparison, our DeRS upcycling based on Sparse Matrix (DeRS-SM) and

Low-rank Matrix (DeRS-LM) improve overall performance by 0.3% and 0.2% with just 1.11 and 2.42 million extra parameters ( $2270\times$  and  $1041\times$  reduction), respectively.

These experimental results highlight the efficiency and effectiveness of our DeRS upcycling, which reduces the number of trainable expert parameters by sharing a base FFN across experts and employing multiple expert-specific lightweight parameters.

# 4.2. Medical Multi-Modal Task

**Model architecture.** We adopt the settings in Med-MoE [20] to conduct our medical multi-modal experiments. Similarly, the CLIP-Large model is used as the vision encoder. Two dense models, StableLM-2-1.6B and Phi-2-2.7B, which have been fine-tuned on the medical language-image instructionfollowing data of [26], are upcycled into MoE architectures to serve as the language backbones. Specifically, the original FFN layer of every other block is converted into a parallel structure, which consists of a universal FFN layer and a MoE layer with four experts. The MoE layer activates only the top-1 expert for a given input, while the universal FFN processes all inputs. The outputs from the MoE layer and the universal FFN are summed to produce the final output. **Datasets.** We use three medical image question-answering datasets with open- and closed-end question-answering pairs, including VQA-RAD [22], SLAKE [32], and PathVQA [16], to fine-tune and evaluate upcycled MoE models.

**Results of DeRS Compression.** Fig. 5 shows the performance of applying DeRS compression to two vanilla upcycled Med-MoE models. It can be observed that extreme compression of expert-specific delta weights (removing 99%)

<span id="page-6-2"></span><span id="page-6-1"></span>Table 3. Performance comparison between vanilla upcycling and our DeRS upcycling on the code generation task. DeRS-SM and DeRS-LM denote the Sparse-Matrix-based and Low-rank-Matrix-based DeRS upcycling respectively. The light-gray Added Params denotes the additional parameters introduced by the universal FFN layers that are not considered as experts of MoE layers.

| MoE Model                 | Upcycling<br>Method | Added<br>Params. | HumanEval | HumanEval+ | MBPP | MBPP+ | Overall |
|---------------------------|---------------------|------------------|-----------|------------|------|-------|---------|
| Coder-MoE [6]<br>(ACL 24) | Vanilla             | 0.81B+2.43B      | 64.6      | 61.0       | 63.9 | 51.4  | 60.2    |
|                           | DeRS-SM (ours)      | 0.81B + 325M     | 66.5      | 62.8       | 63.4 | 51.4  | 61.0    |
|                           | DeRS-LM (ours)      | 0.81B+9.09M      | 65.9      | 62.8       | 62.9 | 51.9  | 60.9    |

of their elements or quantizing them to 1bit) has negligible impact on model performance. This may be attributed to two factors. First, the two dense models utilized within the Med-MoE framework have been previously fine-tuned on relevant yet non-overlapping medical multi-modal datasets. Second, the Med-MoE framework involves replacing the original FFN layer in the dense models with a parallel structure comprising a MoE layer and a universal FFN layer. And in our main experiments, the universal FFN layer is not considered as an expert for compression, as it's not sparsely activated by the router. Consequently, despite the decomposition and delta weight compression applied to the MoE layer's experts, the overall model performance is sustained by the universal FFN. In the Appendix, we further provide experimental results of applying DeRS compression to both experts and the universal FFN. Even with additional compression of the universal FFN, DeRS-Sparsification with a 0.8 drop rate or DeRS-Quantization with a 4-bit width can still significantly reduce redundancy while maintaining performance.

Results of DeRS Upcycling. As shown in Tab. 2, our DeRS upcycling remains efficient and effective across two Med-MoE models on the medical multi-modal task. For the Med-MoE-StableLM, our DeRS-SM and DeRS-LM upcycling strategies significantly reduce the additional parameter count by 1.24 billion compared to vanilla upcycling, while achieving performance improvement of 0.2% and 0.3%, respectively. Furthermore, for the Med-MoE-Phi architecture, the two DeRS upcycling methods reduce the number of additional parameters by 2.52 billion while maintaining comparable performance. Due to the presence of the universal FFN layers, the Med-MoE model obtained through DeRS upcycling still incurs an increase of 0.42 billion or 0.84 billion parameters compared to its corresponding dense model. In the Appendix, we further extend DeRS upcycling to the universal FFN, treating the construction of the universal FFN and the MoE layer's experts as a unified entity. This allows us to achieve performance comparable to that of vanilla upcycling with only a million-level increase in additional parameters over the dense model.

#### 4.3. Code Generation Task

**Model architecture.** Following the settings in [6], we further evaluate our DeRS compression and DeRS upcycling on the code generation task. We directly upcycle the open-source

<span id="page-6-0"></span>![](_page_6_Figure_6.jpeg)

Figure 6. Performance of applying different DeRS compression methods to compress the vanilla upcycled Coder-MoE model.

language model, DeepSeek-Coder-Base-1.3B [11] into the Coder-MoE architecture for fine-tuning. Specifically, each block's FFN layer is replaced with a combination of a MoE layer and a universal FFN layer. The MoE layer contains four experts, with the top-1 expert activated for each input.

**Datasets.** We utilize *evol-codealpaca-v1*, an open-source Evol-Instruct [40] dataset with 110K instruction-output pairs, to fine-tune the upcycled MoE model. After fine-tuning, we evaluate the model on widely used Python code generation benchmarks, including HumanEval [5] and MBPP [2], as well as on extended HumanEval+ and MBPP+ that contain additional tests generated by EvalPlus [36].

Results of DeRS Compression. As shown in Fig. 6, for the expert-specific delta weights in the Coder-MoE model, removing 60% of their elements or quantizing them to 2 bits can effectively eliminate redundancy without degrading performance. In comparison to the compression results of Med-MoE models shown in Fig. 5, the delta weight redundancy in the Coder-MoE model is notably lower, despite both experimental setups involving a parallel structure comprising a universal FFN and a MoE layer. The lower redundancy may be linked to the fact that the dense model utilized for Coder-MoE has not undergone any prior fine-tuning. Further experiments of compressing both the universal FFN and MoE experts are also provided in the Appendix.

Results of DeRS Upcycling. As shown in Tab 3, our DeRS-SM and DeRS-LM upcycling methods yield significant overall performance improvement of 0.7%-0.8% on the code generation task for the Coder-MoE model, while reducing over 2 billion extra parameters compared to vanilla upcycling. This substantial performance improvement may be attributed to the sparse activation mechanism and the low-budget training setting. In vanilla upcycling, each expert is an independent FFN with extensive parameters, struggling to

<span id="page-7-1"></span>Table 4. Ablation studies on the hyper-parameters of two DeRS upcycling methods. DeRS-SM and DeRS-LM denote the Sparse-Matrix-based and Low-rank-Matrix-based DeRS upcycling respectively. The light-gray Added Params represents the additional parameters introduced by the universal FFN layers that are not considered as experts of MoE layers.

(a) Different sparse rates for DeRS-SM upcycling

| DeRS-SM<br>Rate | Added<br>Params.   | HumanEval (+) | MBPP (+)    | Overall |
|-----------------|--------------------|---------------|-------------|---------|
| 0.9999          | 0.81B+0.72M        | 62.8 (60.4)   | 63.7 (51.6) | 59.6    |
| 0.999           | 0.81B+3.64M        | 64.0 (60.4)   | 63.4 (51.6) | 59.9    |
| 0.99            | 0.81B+32.9M        | 64.6 (61.6)   | 62.7 (51.4) | 60.1    |
| 0.9             | 0.81B <b>+325M</b> | 66.5 (62.8)   | 63.4 (51.4) | 61.0    |

<span id="page-7-0"></span>Table 5. Ablation studies on whether freezing the expert-shared base FFN for our two DeRS upcycling methods.

| Upcycling<br>Method | Freeze<br>Shared | HumanEval (+)              | <b>MBPP</b> (+)            | Overall      |
|---------------------|------------------|----------------------------|----------------------------|--------------|
| DeRS-SM             | ×                | 66.5 (62.8)<br>65.2 (61.6) | 62.4 (51.4)<br>61.4 (50.4) | 61.0<br>59.7 |
| DeRS-LM             | ×                | 65.9 (62.8)<br>64.0 (61.0) | 62.9 (51.9)<br>61.7 (50.6) | 60.9<br>59.3 |

learn effectively due to the limited training data. In contrast, our DeRS upcycling allows multiple experts to share a base FFN while retaining their specific lightweight incremental parameters. This design enables the base FFN to leverage all training data for robust learning, while partial training data suffices to effectively train the sparsely activated lightweight incremental parameters. In the Appendix, we also present the experimental results of extending DeRS upcycling to the universal FFN within the Coder-MoE architecture, highlighting the ability of DeRS upcycling to achieve extremely efficient upcycled MoE models.

# 4.4. Ablation and Analysis

In this section, we conduct ablation studies and cost analysis of our DeRS upcycling on the code generation task.

Effect of freezing the expert-shared base FFN. As shown in Tab. 5, freezing the expert-shared base FFN results in a significant overall performance drop for both DeRS-SM and DeRS-LM upcycling strategies, with declines of 1.3% and 1.6%, respectively. These results validate the necessity of setting the expert-shared base FFN as trainable, as it represents foundational knowledge for the MoE experts and plays a crucial role in enabling effective learning.

Effect of the hyper-parameter. In Tab. 4, we explore the effect of the hyper-parameter of DeRS upcycling, which determines the parameter count of expert-specific incremental weights, specifically the sparse rate for DeRS-SM and the rank for DeRS-LM. As shown in Tab. 4a, for DeRS-SM, where the incremental weights take the form of sparse matrixes, a lower sparsity generally leads to better performance, albeit with an increase in the number of additional parameters. This can be attributed to the inherent limitation of the sparse matrix, which can only modify a subset of

(b) Different ranks for DeRS-LM upcycling

| DeRS-LM<br>Rank | Added<br>Params.    | HumanEval (+) | MBPP (+)    | Overall |
|-----------------|---------------------|---------------|-------------|---------|
| 1               | 0.81B+2.57M         | 64.6 (61.6)   | 63.4 (51.1) | 60.2    |
| 4               | 0.81B <b>+9.09M</b> | 65.9 (62.8)   | 62.9 (51.9) | 60.9    |
| 16              | 0.81B+35.2M         | 65.9 (62.2)   | 63.2 (51.4) | 60.7    |
| 64              | 0.81B <b>+140M</b>  | 63.4 (59.1)   | 62.7 (50.9) | 59.0    |

<span id="page-7-2"></span>Table 6. Cost efficiency comparison between vanilla upcycling and our two DeRS upcycling methods.

| Upcycling<br>Method | Model<br>Size | Training<br>Memory | Inference<br>Memory | Overall<br>Performance |
|---------------------|---------------|--------------------|---------------------|------------------------|
| Vanilla             | 4.59B         | 25.9G              | 10.5G               | 60.2                   |
| DeRS-SM (0.9rate)   | 2.48B         | 24.9G              | 9.0G                | 61.0                   |
| DeRS-SM (0.99rate)  | 2.19B         | 21.0G              | 6.1G                | 60.1                   |
| DeRS-LM (4rank)     | 2.17B         | 20.4G              | 5.9G                | 60.9                   |

elements in another matrix of the same shape, as shown in Fig 3(b). Therefore, to enhance each expert's distinctiveness for better performance, a lower sparsity is necessary to allow the sparse-matrix-based incremental weights to make more substantial adjustments to the expert-shared base weight.

In contrast, as shown in Tab. 4b, our DeRS-LM upcycling method can achieve strong performance with minimal extra parameters by using only a low rank (1, 4 or 16). This is due to the fact that the low-rank-matrix-based incremental weights are always able to make global adjustments to the expert-shared base weight, regardless of the rank, as shown in Fig 3(c). Moreover, if the rank is set too high (64), it can lead to performance degradation. This is likely because a high rank introduces significant redundancy in the incremental weights, making them less efficient for training.

Cost Analysis. Tab. 6 demonstrates the cost efficiency of our DeRS upcycling compared to vanilla upcycling. As shown, both DeRS upcycling methods achieve comparable or even superior performance while reducing model size and GPU memory consumption during both training and inference. For instance, our DeRS-LM upcycling significantly reduces model size by 52.7%, training memory by 21.2%, and inference memory by 43.8%, while achieving an overall performance improvement of 0.7%. These results showcase that DeRS upcycling propels upcycled MoE models towards a new level of efficiency.

#### 5. Conclusion

In this work, we investigate the distinctive redundancy mechanisms of upcycled MoE models, and introduce an innovative DeRS paradigm that remodels MoE experts into one shared base weight and multiple exclusive compact weights. Further, we propose DeRS compression and DeRS upcycling

to enhance the efficiency of expert parameters in upcycled MoE models during inference and training, respectively. We conduct comprehensive experiments to support the effectiveness of our proposals. Future works could focus on improving the parameter-efficient techniques in DeRS compression and DeRS upcycling or extending the DeRS paradigm to scenarios with higher training budgets.

# 6. Acknowledgments

This work is supported by National Key Research and Development Program of China (No. 2022ZD0160101), Shanghai Natural Science Foundation (No. 23ZR1402900), Shanghai Municipal Science and Technology Major Project (No. 2021SHZDZX0103). The computations in this research were performed using the CFFF platform of Fudan University.

# References

- <span id="page-8-0"></span>[1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. *arXiv preprint arXiv:2404.14219*, 2024. [1](#page-0-1)
- <span id="page-8-20"></span>[2] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021. [7](#page-6-2)
- <span id="page-8-12"></span>[3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. *arXiv preprint arXiv:2309.16609*, 2023. [5](#page-4-2)
- <span id="page-8-11"></span>[4] Marco Bellagente, Jonathan Tow, Dakota Mahan, Duy Phung, Maksym Zhuravinskyi, Reshinth Adithyan, James Baicoianu, Ben Brooks, Nathan Cooper, Ashish Datta, et al. Stable lm 2 1.6 b technical report. *arXiv preprint arXiv:2402.17834*, 2024. [5](#page-4-2)
- <span id="page-8-19"></span>[5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021. [7](#page-6-2)
- <span id="page-8-6"></span>[6] Yifeng Ding, Jiawei Liu, Yuxiang Wei, Terry Yue Zhuo, and Lingming Zhang. Xft: Unlocking the power of code instruction tuning by simply merging upcycled mixture-of-experts. *arXiv preprint arXiv:2404.15247*, 2024. [1,](#page-0-1) [2,](#page-1-0) [7](#page-6-2)
- <span id="page-8-9"></span>[7] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In *International Conference on Machine Learning*, pages 5547–5569. PMLR, 2022. [2](#page-1-0)
- <span id="page-8-1"></span>[8] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022. [1,](#page-0-1) [2](#page-1-0)

- <span id="page-8-3"></span>[9] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. Megablocks: Efficient sparse training with mixtureof-experts. *Proceedings of Machine Learning and Systems*, 5: 288–304, 2023. [1](#page-0-1)
- <span id="page-8-14"></span>[10] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 6904–6913, 2017. [5](#page-4-2)
- <span id="page-8-18"></span>[11] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming–the rise of code intelligence. *arXiv preprint arXiv:2401.14196*, 2024. [7](#page-6-2)
- <span id="page-8-16"></span>[12] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3608–3617, 2018. [5](#page-4-2)
- <span id="page-8-7"></span>[13] Xumeng Han, Longhui Wei, Zhiyang Dou, Zipeng Wang, Chenhui Qiang, Xin He, Yingfei Sun, Zhenjun Han, and Qi Tian. Vimoe: An empirical study of designing vision mixtureof-experts. *arXiv preprint arXiv:2410.15732*, 2024. [1](#page-0-1)
- <span id="page-8-5"></span>[14] Ethan He, Abhinav Khattar, Ryan Prenger, Vijay Korthikanti, Zijie Yan, Tong Liu, Shiqing Fan, Ashwath Aithal, Mohammad Shoeybi, and Bryan Catanzaro. Upcycling large language models into mixture of experts. *arXiv preprint arXiv:2410.07524*, 2024. [1](#page-0-1)
- <span id="page-8-10"></span>[15] Shwai He, Daize Dong, Liang Ding, and Ang Li. Demystifying the compression of mixture-of-experts through a unified framework. *arXiv preprint arXiv:2406.02500*, 2024. [2](#page-1-0)
- <span id="page-8-17"></span>[16] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. *arXiv preprint arXiv:2003.10286*, 2020. [6](#page-5-2)
- <span id="page-8-15"></span>[17] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 6700–6709, 2019. [5](#page-4-2)
- <span id="page-8-13"></span>[18] Mojan Javaheripi, Sebastien Bubeck, Marah Abdin, Jyoti ´ Aneja, Sebastien Bubeck, Caio Cesar Teodoro Mendes, ´ Weizhu Chen, Allie Del Giorno, Ronen Eldan, Sivakanth Gopi, et al. Phi-2: The surprising power of small language models. *Microsoft Research Blog*, 2023. [5](#page-4-2)
- <span id="page-8-2"></span>[19] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024. [1](#page-0-1)
- <span id="page-8-8"></span>[20] Songtao Jiang, Tuo Zheng, Yan Zhang, Yeying Jin, Li Yuan, and Zuozhu Liu. Med-moe: Mixture of domain-specific experts for lightweight medical vision-language models, 2024. [1,](#page-0-1) [2,](#page-1-0) [6](#page-5-2)
- <span id="page-8-4"></span>[21] Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay,

- Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-of-experts from dense checkpoints. *arXiv preprint arXiv:2212.05055*, 2022. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-9-21"></span>[22] Jason J Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. *Scientific data*, 5(1):1–10, 2018. [6](#page-5-2)
- <span id="page-9-1"></span>[23] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*, 2020. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-9-6"></span>[24] Bo Li, Yifei Shen, Jingkang Yang, Yezhen Wang, Jiawei Ren, Tong Che, Jun Zhang, and Ziwei Liu. Sparse mixture-ofexperts are domain generalizable learners. *arXiv preprint arXiv:2206.04046*, 2022. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-9-13"></span>[25] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimic-it: Multi-modal in-context instruction tuning. *arXiv preprint arXiv:2306.05425*, 2023. [5](#page-4-2)
- <span id="page-9-20"></span>[26] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large languageand-vision assistant for biomedicine in one day. *Advances in Neural Information Processing Systems*, 36, 2024. [6](#page-5-2)
- <span id="page-9-7"></span>[27] Jiachen Li, Xinyao Wang, Sijie Zhu, Chia-Wen Kuo, Lu Xu, Fan Chen, Jitesh Jain, Humphrey Shi, and Longyin Wen. Cumo: Scaling multimodal llm with co-upcycled mixture-ofexperts. *arXiv preprint arXiv:2405.05949*, 2024. [1](#page-0-1)
- <span id="page-9-9"></span>[28] Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, and Tianlong Chen. Merge, then compress: Demystify efficient smoe with hints from its routing policy. *arXiv preprint arXiv:2310.01334*, 2023. [2](#page-1-0)
- <span id="page-9-18"></span>[29] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. *arXiv preprint arXiv:2305.10355*, 2023. [5](#page-4-2)
- <span id="page-9-4"></span>[30] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Junwu Zhang, Munan Ning, and Li Yuan. Moe-llava: Mixture of experts for large vision-language models. *arXiv preprint arXiv:2401.15947*, 2024. [1,](#page-0-1) [2,](#page-1-0) [5](#page-4-2)
- <span id="page-9-2"></span>[31] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. *arXiv preprint arXiv:2405.04434*, 2024. [1](#page-0-1)
- <span id="page-9-22"></span>[32] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. Slake: A semantically-labeled knowledgeenhanced dataset for medical visual question answering. In *2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)*, pages 1650–1654. IEEE, 2021. [6](#page-5-2)
- <span id="page-9-10"></span>[33] Enshu Liu, Junyi Zhu, Zinan Lin, Xuefei Ning, Matthew B Blaschko, Shengen Yan, Guohao Dai, Huazhong Yang, and Yu Wang. Efficient expert pruning for sparse mixture-of-experts language models: Enhancing performance and reducing inference costs. *arXiv preprint arXiv:2407.00945*, 2024. [2](#page-1-0)
- <span id="page-9-14"></span>[34] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal

- model with robust instruction tuning. *arXiv preprint arXiv:2306.14565*, 2023. [5](#page-4-2)
- <span id="page-9-8"></span>[35] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 26296–26306, 2024. [2,](#page-1-0) [5](#page-4-2)
- <span id="page-9-24"></span>[36] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. *Advances in Neural Information Processing Systems*, 36, 2024. [7](#page-6-2)
- <span id="page-9-19"></span>[37] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In *European Conference on Computer Vision*, pages 216–233. Springer, 2025. [5](#page-4-2)
- <span id="page-9-16"></span>[38] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. *Advances in Neural Information Processing Systems*, 35:2507–2521, 2022. [5](#page-4-2)
- <span id="page-9-11"></span>[39] Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models. *arXiv preprint arXiv:2402.14800*, 2024. [2](#page-1-0)
- <span id="page-9-23"></span>[40] Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. *arXiv preprint arXiv:2306.08568*, 2023. [7](#page-6-2)
- <span id="page-9-5"></span>[41] Joan Puigcerver, Carlos Riquelme, Basil Mustafa, and Neil Houlsby. From sparse to soft mixtures of experts. *arXiv preprint arXiv:2308.00951*, 2023. [1](#page-0-1)
- <span id="page-9-12"></span>[42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021. [5](#page-4-2)
- <span id="page-9-3"></span>[43] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andre Susano Pinto, Daniel ´ Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. *Advances in Neural Information Processing Systems*, 34:8583–8595, 2021. [1](#page-0-1)
- <span id="page-9-0"></span>[44] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixtureof-experts layer. *arXiv preprint arXiv:1701.06538*, 2017. [1](#page-0-1)
- <span id="page-9-17"></span>[45] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8317–8326, 2019. [5](#page-4-2)
- <span id="page-9-15"></span>[46] Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting

- gpt-4v for better visual instruction tuning. *arXiv preprint arXiv:2311.07574*, 2023. [5](#page-4-2)
- <span id="page-10-2"></span>[47] Tianwen Wei, Bo Zhu, Liang Zhao, Cheng Cheng, Biye Li, Weiwei Lu, Peng Cheng, Jianhao Zhang, Xiaoyu Zhang, ¨ Liang Zeng, et al. Skywork-moe: A deep dive into training techniques for mixture-of-experts language models. *arXiv preprint arXiv:2406.06563*, 2024. [1](#page-0-1)
- <span id="page-10-3"></span>[48] Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You. Openmoe: An early effort on open mixture-of-experts language models. *arXiv preprint arXiv:2402.01739*, 2024. [2](#page-1-0)
- <span id="page-10-0"></span>[49] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. *arXiv preprint arXiv:2407.10671*, 2024. [1](#page-0-1)
- <span id="page-10-4"></span>[50] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In *Forty-first International Conference on Machine Learning*, 2024. [4](#page-3-1)
- <span id="page-10-6"></span>[51] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. *arXiv preprint arXiv:2308.02490*, 2023. [5](#page-4-2)
- <span id="page-10-5"></span>[52] Bo Zhao, Boya Wu, Muyang He, and Tiejun Huang. Svit: Scaling up visual instruction tuning. *arXiv preprint arXiv:2307.04087*, 2023. [5](#page-4-2)
- <span id="page-10-1"></span>[53] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. Stmoe: Designing stable and transferable sparse expert models. *arXiv preprint arXiv:2202.08906*, 2022. [1](#page-0-1)

### A. Extended DeRS Compression and Upcycling

In our medical multi-modal and code generation experiments, the original FFN layer in a pre-trained dense model is upcycled into a parallel structure consisting of a universal FFN layer and a MoE layer containing N FFN experts. The universal FFN and the N experts are all initialized from the original FFN weight. The universal FFN processes all inputs, while the N experts are sparsely activated by a router for each input. The outputs from the universal FFN and the MoE layer are then summed to form the final output.

In the main body, we applied the proposed DeRS paradigm only to the N experts in the MoE layer, since the universal FFN is not sparsely activated by the router, meaning it cannot be strictly considered as a MoE expert. Here, considering that both the universal FFN and the N MoE experts share the same initial weight, we extend our DeRS compression and DeRS upcycling to the universal FFN layer to further reduce parameter redundancy.

Specifically, when applying the extended DeRS compression to compress a vanilla upcycled MoE model, both the universal FFN and the N MoE experts are treated as a whole and decomposed into one expert-shared base weight and N+1 delta weights. Subsequently, sparsification or quantization techniques are applied to the N+1 delta weights to reduce redundancy. Similarly, when applying the extended DeRS upcycling to convert a pre-trained dense model into the MoE architecture, the universal FFN and the N MoE experts are treated as a whole, sharing one base FFN and introducing N+1 unique, parameter-efficient weights in the form of sparse or low-rank matrixes.

#### A1. Extended DeRS Compression on Medical Task

Fig. S7 shows the performance of applying the extended DeRS compression to two vanilla upcycled Med-MoE models on the medical multimodal task. The detailed results are presented in Tab. S11 and Tab. S12. As we can see, even when simultaneously compressing the universal FFN and MoE experts, the extended DeRS-Sparsification with a 0.8 drop rate and the extended DeRS-Quantization with a 4-bit width can reduce the additional parameter count by 75% and 69%, respectively, while maintaining performance.

Different from the results shown in Fig. 5, where only the MoE experts were compressed, simultaneously compressing the universal FFN and MoE experts leads to a slight performance drop under extreme compression settings (0.99 drop rate or 1-bit width). This degradation occurs because extreme compression of the universal FFN significantly impacts the model's output, as the universal FFN processes all input tokens. However, since the two dense models utilized within the Med-MoE framework have been previously finetuned on relevant yet non-overlapping medical multi-modal datasets, the overall performance of upcycled MoE models does not collapse under these extreme compression settings.

<span id="page-11-0"></span>![](_page_11_Figure_7.jpeg)

Figure S7. Performance of applying the extended DeRS compression to compress two vanilla upcycled Med-MoE models respectively. For each dataset, we report the average performance of the open-set and closed-set.

### A2. Extended DeRS Upcycling on Medical Task

As shown in Tab. S7, when treating the construction of the universal FFN and MoE experts as a whole, both of our extended DeRS upcycling methods achieve comparable performance to vanilla upcycling while introducing significantly fewer additional parameters. For example, when achieving the same performance on the Med-MoE-Phi architecture, our extended DeRS-SM and DeRS-LM upcycling strategies introduce only 5.18 million and 9.18 million additional parameters respectively, while vanilla upcycling introduces a massive 3.36 billion parameters. These results highlight the ability of our DeRS upcycling to achieve extremely efficient upcycled MoE models.

### A3. Extended DeRS Compression on Code Task

Fig. S8 shows the performance of applying the extended DeRS compression to the vanilla upcycled Coder-MoE model on the code generation task, with detailed results presented in Tab. S9 and Tab. S10. As we can see, for the delta weights obtained by the unified decomposition of the universal FFN and MoE experts, removing 40% of their elements or quantizing them to 4 bits can effectively eliminate redundancy without degrading performance. However, since the dense model utilized for constructing Coder-MoE has not undergone any prior fine-tuning, excessive simultaneous compression of both the universal FFN and MoE experts can lead to a collapse in the performance of the vanilla upcycled Coder-MoE model.

<span id="page-12-0"></span>Table S7. Performance comparison between vanilla upcycling and our extended DeRS upcycling on two Med-MoE models on the medical multi-modal task. DeRS-SM<sup>†</sup> and DeRS-LM<sup>†</sup> denote the extended Sparse-Matrix-based and Low-rank-Matrix-based DeRS upcycling respectively. **Added Params** represents the number of additional parameters of the upcycled MoE model compared to its corresponding dense model.

| MoE Model                      | Upcycling            | Added   | VQA  | VQA-RAD |      | SLAKE  |      | PathVQA |         |
|--------------------------------|----------------------|---------|------|---------|------|--------|------|---------|---------|
| MOE Model                      | Method               | Params. | Open | Closed  | Open | Closed | Open | Closed  | Overall |
| Med-MoE-StableLM<br>(EMNLP 24) | Vanilla              | 1.66B   | 51.0 | 82.3    | 82.4 | 85.3   | 33.4 | 91.4    | 71.0    |
|                                | DeRS-SM <sup>†</sup> | 2.17M   | 51.2 | 81.3    | 84.5 | 84.4   | 33.6 | 90.9    | 71.0    |
|                                | DeRS-LM <sup>†</sup> | 5.63M   | 50.4 | 81.6    | 83.6 | 84.4   | 33.9 | 91.4    | 70.9    |
| Med-MoE-Phi<br>(EMNLP 24)      | Vanilla              | 3.36B   | 55.1 | 85.3    | 84.6 | 85.8   | 35.1 | 91.5    | 72.9    |
|                                | DeRS-SM <sup>†</sup> | 5.18M   | 54.8 | 84.6    | 84.0 | 87.2   | 35.0 | 91.6    | 72.9    |
|                                | DeRS-LM <sup>†</sup> | 9.18M   | 55.3 | 83.8    | 84.3 | 86.5   | 35.6 | 91.9    | 72.9    |

#### A4. Extended DeRS Upcycling on Code Task

As shown in Tab. S8, our extended DeRS upcycling remains effective and extremely efficient on the code generation task. For example, our extended DeRS-LM upcycling strategy achieves an overall performance improvement of 0.7%, while only introducing only 11.3 million additional parameters, whereas vanilla upcycling introduces a significant 3.24 billion extra parameters. These results demonstrate that our proposed DeRS upcycling method propels upcycled MoE models towards a new level of efficiency.

# **B. Detailed Results of DeRS Compression**

Detailed results of DeRS compression in the main body are provided, namely Tab. S13 and Tab. S14 for the general multi-modal task, Tab. S15 and Tab. S16 for the medical multi-modal task, and Tab. S17 and Tab. S18 for the code generation task.

# C. Training settings

The detailed training hyper-parameters and our DeRS upcycling hyper-parameters for experiments on three tasks are provided in Tab. S19.

#### **D. Recommended Application Choices**

Based on extensive experiments, we empirically summarize recommended application choices for different scenarios. If the pre-trained dense model has undergone prior fine-tuning before upcycling, we recommend applying the sparsification-based DeRS compression to efficiently compress the vanilla upcycled MoE model, as well as utilizing sparse-matrix-based DeRS upcycling to efficiently upcycle the dense model into the MoE architecture for training. This is because, in this case, the redundancy in the delta weights is extremely high, and both sparsification and sparse matrixes can significantly reduce redundancy while maintaining performance. Conversely, if the pre-trained dense model has not under-

<span id="page-12-1"></span>![](_page_12_Figure_10.jpeg)

Figure S8. Performance of applying the extended DeRS compression to compress the vanilla upcycled Coder-MoE model. HumanEval(+) represents the average performance of HumanEval and HumanEval+, similarly for MBPP(+).

gone any prior fine-tuning, we recommend employing the quantization-based DeRS compression and the low-rank-matrix-based DeRS upcycling, as these two methods can effectively reduce redundancy while preserving global modification capabilities.

Since our proposed DeRS compression is based on the assumption that MoE experts share the same pre-trained weight initialization for the decomposition of experts and compression of redundant delta weights, it is not applicable to compressing MoE models trained from scratch. This is because training MoE models from scratch involves randomly initializing the MoE experts, making it impossible to extract redundant delta weights from the trained experts. Moreover, although our proposed DeRS upcycling has the potential to be used for training MoE models from scratch by randomly initializing the expert-shared base FFN, its performance may be limited due to insufficient model capacity.

<span id="page-13-2"></span>Table S8. Performance comparison between vanilla upcycling and our extended DeRS upcycling on the code generation task. DeRS-SM† and DeRS-LM† denote the extended Sparse-Matrix-based and Low-rank-Matrix-based DeRS upcycling respectively. Added Params represents the number of additional parameters of the upcycled MoE model compared to its corresponding dense model.

| MoE Model             | Upcycling<br>Method | Added<br>Params. | HumanEval | HumanEval+ | MBPP | MBPP+ | Overall |
|-----------------------|---------------------|------------------|-----------|------------|------|-------|---------|
| Coder-MoE<br>(ACL 24) | Vanilla             | 3.24B            | 64.6      | 61.0       | 63.9 | 51.4  | 60.2    |
|                       | DeRS-SM†            | 406M             | 64.6      | 60.4       | 63.7 | 52.4  | 60.3    |
|                       | DeRS-LM†            | 11.3M            | 65.9      | 62.2       | 63.4 | 51.9  | 60.9    |

<span id="page-13-0"></span>Table S9. Detailed results of applying the extended DeRS-Sparsification (with different drop rates) to compress the vanilla upcycled Coder-MoE model on the code generation task. Added Params represents the number of additional parameters of the compressed MoE model compared to its corresponding dense model.

| Vanilla Upcycled<br>MoE Model | Drop<br>Rate | Added<br>Params. | HumanEval | HumanEval+ | MBPP | MBPP+ |
|-------------------------------|--------------|------------------|-----------|------------|------|-------|
|                               | 0.0          | 3.24B            | 64.6      | 61.0       | 63.9 | 51.4  |
|                               | 0.2          | 3.24B            | 63.4      | 59.8       | 64.7 | 52.9  |
|                               | 0.4          | 2.43B            | 63.4      | 60.4       | 62.9 | 52.4  |
| Coder-MoE                     | 0.6          | 1.62B            | 61.0      | 57.9       | 61.2 | 50.6  |
| (ACL 24)                      | 0.8          | 0.81B            | 62.2      | 57.3       | 61.4 | 49.6  |
|                               | 0.9          | 0.41B            | 58.5      | 54.3       | 55.4 | 44.6  |
|                               | 0.99         | 0.04B            | 0.0       | 0.0        | 0.0  | 0.0   |

<span id="page-13-1"></span>Table S10. Detailed results of applying the extended DeRS-Quantization (with different bit width) to compress the vanilla upcycled Coder-MoE model on the code generation task. Added Params represents the number of additional parameters of the compressed MoE model compared to its corresponding dense model.

| Vanilla Upcycled<br>MoE Model | Bit<br>Added<br>Width<br>Params. |       | HumanEval | HumanEval+ | MBPP | MBPP+ |
|-------------------------------|----------------------------------|-------|-----------|------------|------|-------|
|                               | 16                               | 3.24B | 64.6      | 61.0       | 63.9 | 51.4  |
|                               | 8                                | 2.03B | 64.6      | 60.4       | 63.7 | 51.6  |
| Coder-MoE                     | 4                                | 1.01B | 63.4      | 60.4       | 63.7 | 52.1  |
| (ACL 24)                      | 2                                | 0.51B | 6.0       | 6.0        | 0.0  | 0.0   |
|                               | 1                                | 0.25B | 0.0       | 0.0        | 0.0  | 0.0   |

<span id="page-14-0"></span>Table S11. Detailed results of applying the extended DeRS-Sparsification (with different drop rates) to compress two vanilla upcycled Med-MoE models on the medical multi-modal task. Added Params represents the number of additional parameters of the compressed MoE model compared to its corresponding dense model.

| Vanilla Upcycled | Drop | Added   |      | VQA-RAD |      | SLAKE  |      | PathVQA |
|------------------|------|---------|------|---------|------|--------|------|---------|
| MoE Model        | Rate | Params. | Open | Closed  | Open | Closed | Open | Closed  |
|                  | 0.0  | 1.66B   | 51.0 | 82.3    | 82.4 | 85.3   | 33.4 | 91.4    |
|                  | 0.2  | 1.66B   | 51.0 | 82.3    | 82.5 | 85.3   | 33.3 | 91.4    |
|                  | 0.4  | 1.25B   | 50.8 | 82.3    | 82.5 | 85.1   | 33.2 | 91.3    |
| Med-MoE-StableLM | 0.6  | 0.83B   | 50.3 | 82.0    | 82.4 | 85.6   | 33.1 | 91.4    |
| (EMNLP 24)       | 0.8  | 0.42B   | 48.6 | 82.3    | 82.7 | 85.3   | 33.2 | 91.5    |
|                  | 0.9  | 0.21B   | 48.8 | 82.3    | 82.4 | 85.3   | 33.0 | 91.4    |
|                  | 0.99 | 0.02B   | 42.5 | 79.0    | 81.7 | 85.3   | 32.3 | 91.3    |
|                  | 0.0  | 3.36B   | 55.0 | 85.3    | 84.6 | 85.8   | 35.1 | 91.5    |
|                  | 0.2  | 3.36B   | 55.0 | 84.9    | 84.7 | 85.8   | 35.1 | 91.5    |
|                  | 0.4  | 2.52B   | 55.0 | 85.3    | 85.0 | 85.8   | 35.1 | 91.5    |
| Med-MoE-Phi      | 0.6  | 1.68B   | 55.1 | 84.9    | 84.8 | 85.8   | 34.9 | 91.6    |
| (EMNLP 24)       | 0.8  | 0.84B   | 55.1 | 84.9    | 84.9 | 85.3   | 35.0 | 91.3    |
|                  | 0.9  | 0.42B   | 55.3 | 84.9    | 84.8 | 85.3   | 35.2 | 91.4    |
|                  | 0.99 | 0.21B   | 57.0 | 85.7    | 83.7 | 85.1   | 34.9 | 91.2    |

<span id="page-14-1"></span>Table S12. Detailed results of applying the extended DeRS-Quantization (with different bit width) to compress two vanilla upcycled Med-MoE models on the medical multi-modal task. Added Params represents the number of additional parameters of the compressed MoE model compared to its corresponding dense model.

| Vanilla Upcycled          | Bit   | Added   |      | VQA-RAD |      | SLAKE  |      | PathVQA |
|---------------------------|-------|---------|------|---------|------|--------|------|---------|
| MoE Model                 | Width | Params. | Open | Closed  | Open | Closed | Open | Closed  |
|                           | 16    | 1.66B   | 51.0 | 82.3    | 82.4 | 85.3   | 33.4 | 91.4    |
|                           | 8     | 1.04B   | 51.0 | 82.3    | 82.5 | 85.3   | 33.3 | 91.4    |
| Med-MoE-StableLM          | 4     | 0.52B   | 50.8 | 82.3    | 82.5 | 85.1   | 33.2 | 91.3    |
| (EMNLP 24)                | 2     | 0.26B   | 51.5 | 80.1    | 82.8 | 86.0   | 32.4 | 91.1    |
|                           | 1     | 0.13B   | 33.7 | 77.6    | 66.7 | 80.3   | 23.4 | 87.8    |
|                           | 16    | 3.36B   | 55.0 | 85.3    | 84.6 | 85.8   | 35.1 | 91.5    |
| Med-MoE-Phi<br>(EMNLP 24) | 8     | 2.10B   | 55.0 | 85.3    | 84.6 | 85.8   | 35.1 | 91.5    |
|                           | 4     | 1.05B   | 54.9 | 85.3    | 84.9 | 86.0   | 35.1 | 91.5    |
|                           | 2     | 0.52B   | 56.7 | 85.7    | 83.7 | 85.3   | 33.5 | 91.4    |
|                           | 1     | 0.26B   | 43.6 | 79.4    | 64.2 | 79.8   | 20.1 | 86.6    |

<span id="page-15-0"></span>Table S13. Detailed results of applying DeRS-Sparsification (with different drop rates) to compress three vanilla upcycled MoE-LLaVA models on the general multi-modal task. Added Params represents the number of additional parameters of the compressed MoE model compared to its corresponding dense model.

| Vanilla Upcycled<br>MoE Model | Drop<br>Rate | Added<br>Params. | VQAv2 | GQA  | VQAT |
|-------------------------------|--------------|------------------|-------|------|------|
|                               | 0.0          | 1.24B            | 76.3  | 60.6 | 50.2 |
|                               | 0.2          | 1.33B            | 76.4  | 60.8 | 50.1 |
|                               | 0.4          | 1.00B            | 76.4  | 60.8 | 50.2 |
| MoE-LLaVA-StableLM            | 0.6          | 0.66B            | 76.3  | 60.7 | 50.1 |
| (ICML 24)                     | 0.8          | 0.33B            | 76.3  | 60.7 | 50.2 |
|                               | 0.9          | 0.17B            | 76.3  | 60.5 | 50.0 |
|                               | 0.99         | 0.02B            | 74.8  | 59.4 | 47.4 |
|                               | 0.0          | 1.22B            | 76.2  | 61.2 | 48.1 |
|                               | 0.2          | 1.30B            | 76.2  | 61.3 | 47.7 |
|                               | 0.4          | 0.97B            | 76.2  | 61.1 | 48.0 |
| MoE-LLaVA-Qwen                | 0.6          | 0.65B            | 76.2  | 61.3 | 47.5 |
| (ICML 24)                     | 0.8          | 0.32B            | 76.1  | 61.0 | 47.8 |
|                               | 0.9          | 0.16B            | 76.1  | 61.1 | 47.5 |
|                               | 0.99         | 0.02B            | 73.9  | 59.3 | 42.7 |
|                               | 0.0          | 2.52B            | 77.5  | 61.4 | 50.8 |
|                               | 0.2          | 2.68B            | 77.5  | 61.1 | 50.8 |
|                               | 0.4          | 2.01B            | 77.5  | 61.1 | 50.9 |
| MoE-LLaVA-Phi                 | 0.6          | 1.34B            | 77.4  | 61.4 | 50.9 |
| (ICML 24)                     | 0.8          | 0.67B            | 77.5  | 61.4 | 51.0 |
|                               | 0.9          | 0.34B            | 77.4  | 61.3 | 50.9 |
|                               | 0.99         | 0.03B            | 76.9  | 60.6 | 50.2 |

<span id="page-15-1"></span>Table S14. Detailed results of applying DeRS-Quantization (with different bit width) to compress three vanilla upcycled MoE-LLaVA models on the general multi-modal task. Added Params represents the number of additional parameters of the compressed MoE model compared to its corresponding dense model.

| Vanilla Upcycled<br>MoE Model | Bit<br>Width | Added<br>Params. | VQAv2 | GQA  | VQAT |
|-------------------------------|--------------|------------------|-------|------|------|
|                               | 16           | 1.24B            | 76.3  | 60.6 | 50.2 |
|                               | 8            | 0.83B            | 76.4  | 60.4 | 50.2 |
| MoE-LLaVA-StableLM            | 4            | 0.42B            | 76.3  | 60.6 | 50.1 |
| (ICML 24)                     | 2            | 0.21B            | 76.2  | 60.5 | 50.7 |
|                               | 1            | 0.10B            | 74.1  | 55.8 | 48.1 |
|                               | 16           | 1.22B            | 76.2  | 61.2 | 48.1 |
|                               | 8            | 0.81B            | 76.2  | 61.1 | 48.0 |
| MoE-LLaVA-Qwen                | 4            | 0.41B            | 76.2  | 61.0 | 47.9 |
| (ICML 24)                     | 2            | 0.20B            | 76.1  | 60.9 | 48.7 |
|                               | 1            | 0.10B            | 74.4  | 57.5 | 47.8 |
|                               | 16           | 2.52B            | 77.5  | 61.4 | 50.8 |
|                               | 8            | 1.68B            | 77.5  | 61.2 | 51.1 |
| MoE-LLaVA-Phi                 | 4            | 0.84B            | 77.5  | 61.2 | 50.8 |
| (ICML 24)                     | 2            | 0.42B            | 77.5  | 61.4 | 50.7 |
|                               | 1            | 0.21B            | 75.9  | 58.8 | 49.8 |

<span id="page-16-0"></span>Table S15. Detailed results of applying DeRS-Sparsification (with different drop rates) to compress two vanilla upcycled Med-MoE models on the medical multi-modal task. Added Params represents the number of additional parameters of the compressed MoE model compared to its corresponding dense model. The light-gray Added Params denotes the additional parameters introduced by the universal FFN layers that are not considered as experts of MoE layers.

| Vanilla Upcycled          | Drop | Added       |      | VQA-RAD |      | SLAKE  |      | PathVQA |
|---------------------------|------|-------------|------|---------|------|--------|------|---------|
| MoE Model                 | Rate | Params.     | Open | Closed  | Open | Closed | Open | Closed  |
|                           | 0.0  | 0.42B+1.24B | 51.0 | 82.3    | 82.4 | 85.3   | 33.4 | 91.4    |
|                           | 0.2  | 0.42B+1.33B | 50.6 | 82.3    | 82.3 | 85.3   | 33.3 | 91.3    |
|                           | 0.4  | 0.42B+1.00B | 50.8 | 82.3    | 82.4 | 85.3   | 33.3 | 91.2    |
| Med-MoE-StableLM          | 0.6  | 0.42B+0.66B | 50.6 | 82.3    | 82.4 | 85.3   | 33.2 | 91.4    |
| (EMNLP 24)                | 0.8  | 0.42B+0.33B | 49.8 | 82.7    | 82.9 | 85.6   | 33.3 | 91.3    |
|                           | 0.9  | 0.42B+0.17B | 49.9 | 82.0    | 82.6 | 85.6   | 33.2 | 91.3    |
|                           | 0.99 | 0.42B+0.02B | 49.4 | 80.9    | 81.6 | 85.3   | 32.9 | 91.4    |
|                           | 0.0  | 0.84B+2.52B | 55.0 | 85.3    | 84.6 | 85.8   | 35.1 | 91.5    |
|                           | 0.2  | 0.84B+2.68B | 55.0 | 85.3    | 84.7 | 85.8   | 35.0 | 91.5    |
|                           | 0.4  | 0.84B+2.01B | 55.0 | 85.3    | 84.6 | 86.0   | 35.1 | 91.5    |
| Med-MoE-Phi<br>(EMNLP 24) | 0.6  | 0.84B+1.34B | 55.0 | 85.3    | 84.7 | 86.0   | 35.1 | 91.4    |
|                           | 0.8  | 0.84B+0.67B | 55.0 | 84.6    | 84.9 | 85.6   | 35.2 | 91.5    |
|                           | 0.9  | 0.84B+0.34B | 55.2 | 84.6    | 84.9 | 85.1   | 35.0 | 91.6    |
|                           | 0.99 | 0.84B+0.03B | 55.7 | 84.9    | 84.0 | 85.6   | 35.0 | 91.5    |

<span id="page-16-1"></span>Table S16. Detailed results of applying DeRS-Quantization (with different bit width) to compress two vanilla upcycled Med-MoE models on the medical multi-modal task. Added Params represents the number of additional parameters of the compressed MoE model compared to its corresponding dense model. The light-gray Added Params denotes the additional parameters introduced by the universal FFN layers that are not considered as experts of MoE layers.

| Vanilla Upcycled          | Bit   | Added       |      | VQA-RAD |      | SLAKE  |      | PathVQA |
|---------------------------|-------|-------------|------|---------|------|--------|------|---------|
| MoE Model                 | Width | Params.     | Open | Closed  | Open | Closed | Open | Closed  |
|                           | 16    | 0.42B+1.24B | 51.0 | 82.3    | 82.4 | 85.3   | 33.4 | 91.4    |
|                           | 8     | 0.42B+0.83B | 50.8 | 82.3    | 82.3 | 85.1   | 33.3 | 91.4    |
| Med-MoE-StableLM          | 4     | 0.42B+0.42B | 50.8 | 82.3    | 82.3 | 85.3   | 33.3 | 91.3    |
| (EMNLP 24)                | 2     | 0.42B+0.21B | 50.5 | 82.3    | 82.5 | 85.3   | 32.9 | 91.4    |
|                           | 1     | 0.42B+0.10B | 43.3 | 80.5    | 79.5 | 84.1   | 31.2 | 91.1    |
|                           | 16    | 0.84B+2.52B | 55.0 | 85.3    | 84.6 | 85.8   | 35.1 | 91.5    |
| Med-MoE-Phi<br>(EMNLP 24) | 8     | 0.84B+1.68B | 55.0 | 85.3    | 84.6 | 85.8   | 35.1 | 91.5    |
|                           | 4     | 0.84B+0.84B | 54.9 | 85.3    | 84.9 | 86.3   | 35.1 | 91.5    |
|                           | 2     | 0.84B+0.42B | 54.6 | 85.0    | 84.6 | 85.6   | 34.8 | 91.4    |
|                           | 1     | 0.84B+0.21B | 54.0 | 83.1    | 80.2 | 83.2   | 31.6 | 90.7    |

<span id="page-17-0"></span>Table S17. Detailed results of applying DeRS-Sparsification (with different drop rates) to compress the vanilla upcycled Coder-MoE model on the code generation task. Added Params represents the number of additional parameters of the compressed MoE model compared to its corresponding dense model. The light-gray Added Params denotes the additional parameters introduced by the universal FFN layers that are not considered as experts of MoE layers.

| Vanilla Upcycled<br>MoE Model | Drop<br>Rate | Added<br>Params. | HumanEval | HumanEval+ | MBPP | MBPP+ |
|-------------------------------|--------------|------------------|-----------|------------|------|-------|
|                               | 0.0          | 0.81B+2.43B      | 64.6      | 61.0       | 63.9 | 51.4  |
| Coder-MoE<br>(ACL 24)         | 0.2          | 0.81B+2.60B      | 63.4      | 60.4       | 63.7 | 51.4  |
|                               | 0.4          | 0.81B+1.95B      | 63.4      | 59.8       | 63.9 | 51.6  |
|                               | 0.6          | 0.81B+1.30B      | 64.0      | 59.8       | 64.4 | 53.1  |
|                               | 0.8          | 0.81B+0.65B      | 62.2      | 59.1       | 63.7 | 51.9  |
|                               | 0.9          | 0.81B+0.32B      | 62.2      | 57.3       | 63.4 | 51.6  |
|                               | 0.99         | 0.81B+0.03B      | 56.7      | 53.0       | 56.1 | 45.6  |

<span id="page-17-1"></span>Table S18. Detailed results of applying DeRS-Quantization (with different bit width) to compress the vanilla upcycled Coder-MoE model on the code generation task. Added Params represents the number of additional parameters of the compressed MoE model compared to its corresponding dense model. The light-gray Added Params denotes the additional parameters introduced by the universal FFN layers that are not considered as experts of MoE layers.

| Vanilla Upcycled<br>MoE Model | Bit<br>Width | Added<br>Params. | HumanEval | HumanEval+ | MBPP | MBPP+ |
|-------------------------------|--------------|------------------|-----------|------------|------|-------|
|                               | 16           | 0.81B+2.43B      | 64.6      | 61.0       | 63.9 | 51.4  |
| Coder-MoE<br>(ACL 24)         | 8            | 0.81B+1.62B      | 64.0      | 60.4       | 63.7 | 51.6  |
|                               | 4            | 0.81B+0.81B      | 63.4      | 59.8       | 63.7 | 52.1  |
|                               | 2            | 0.81B+0.41B      | 64.0      | 61.0       | 62.4 | 51.1  |
|                               | 1            | 0.81B+0.20B      | 9.1       | 9.1        | 6.8  | 6.3   |

<span id="page-17-2"></span>Table S19. Detailed training hyper-parameters and our DeRS upcycling hyper-parameters for experiments on three tasks. DeRS-SM Rate denotes the sparse rate for the Sparse-Matrix-based DeRS upcycling while DeRS-LM Rate denotes the rank for the Low-rank-Matrix-based DeRS upcycling. † denotes the extended DeRS upcycling implementation.

| Config                      | Task                |                     |                   |  |  |  |
|-----------------------------|---------------------|---------------------|-------------------|--|--|--|
|                             | General Multi-Modal | Medical Multi-Modal | Code Generation   |  |  |  |
| Training Epochs             | 1                   | 9                   | 4                 |  |  |  |
| Learning rate               | 2e-5                | 2e-5                | 5e-5              |  |  |  |
| Learning rate schedule      | Cosine              | Cosine              | Linear            |  |  |  |
| Training Batch size per GPU | 4                   | 8                   | 4                 |  |  |  |
| Gradient Accumulation Steps | 4                   | 2                   | 2                 |  |  |  |
| Number of GPU               | 8 ×<br>A100 (80G)   | 4 ×<br>A100 (80G)   | 8 ×<br>A100 (80G) |  |  |  |
| Precision                   | Bfloat16            | Bfloat16            | Bfloat16          |  |  |  |
| DeRS-SM Rate                | 0.9999              | 0.9999              | 0.9               |  |  |  |
| DeRS-LM Rank                | 1                   | 1                   | 4                 |  |  |  |
| DeRS-SM† Rate               | -                   | 0.999               | 0.9               |  |  |  |
| DeRS-LM† Rank               | -                   | 4                   | 4                 |  |  |  |