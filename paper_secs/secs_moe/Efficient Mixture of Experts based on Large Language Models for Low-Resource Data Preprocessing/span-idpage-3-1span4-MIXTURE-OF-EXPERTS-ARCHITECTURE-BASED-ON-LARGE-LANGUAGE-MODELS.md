# <span id="page-3-1"></span>4 MIXTURE OF EXPERTS ARCHITECTURE BASED ON LARGE LANGUAGE MODELS

The overall architecture of MELD is presented in Figure 4, and it consists of the following four components.

• The enhanced RAG component. It takes few-shot labeled data as input, and enlarge/enrich labeled data in  $\mathcal{X}_i$  as output  $\mathbb{X}_i$  in a self-supervised manner for task  $\mathcal{T}_i$ . A fine-tuned sentence-bert model is used as the backbone of RAG system; such design can effectively encode data entries from different domains to a unified representation space. It also retrieves relevant demonstrations

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

Figure 4: Architecture Overview

D for each data entry and initializes a set E of experts.

- The meta-path search component. It takes the enlarged training data  $X_i$  and the expert set E as input, and finds a meta-path  $\mathcal{E}_i$  (i.e., a sequence of experts in E) for task  $\mathcal{T}_i$ , to augment  $X_i$  to  $X_i^{aug}$ , by revising and adding attributes for each query  $q \in X_i$ .
- $\circ$  The expert refinement. It takes augmented training data  $\mathbb{X}_i^{aug}$  and the expert set E as input, and fine-tunes the experts E to  $\mathsf{E}^{aug}$ , guided by the information bottleneck theory.
- $\circ$  The router network  $\mathcal{N}$ . It takes the (fixed) refined expert set  $\mathbf{E}^{aug}$  as input, and designs with a sparse multi-gate network, to select top-k experts for answering a query  $q \in \mathbb{X}$ .

Below we elaborate each component one by one.

#### <span id="page-4-1"></span>4.1 Enhanced RAG for Cross-domain Retrieval

Retrieval-Augmented Generation(RAG) is a method to retrieve relevant contextual data entries or chunks from a large corpus (e.g., knowledge graph, book) and provide to the model as reference, to improve the quality of LLM responses. However, data entries from multiple DP tasks may have different structures, which are hard to compare and retrieve. In this section, we propose a simple yet effective method to serialize and align data from different domains.

Entry Alignment. For structure and semi-structured data, the structure similarity holds equal importance as semantic similarity, e.g., if  $t_1$  and  $t_2$  share the same brand and category attribute, we can align  $t_1$  and  $t_2$  as similar entities. For tasks across tables, e.g., CTA, columns with same semantic type or knowledge graph relations should also be aligned; for binary classification tasks, e.g., EM, if  $t_1$  and  $t_2$  are labeled as match, they should be grouped as similar entries.

Based on this, for each q in  $\mathbb{X}$ , we search a positive set  $\mathcal{P}_q$  (resp. a negative set  $\mathcal{N}_q$ ) containing all aligned (resp. unaligned) entries.

Fine-tuning RAG Model. Given  $(q, \mathcal{P}_q, \mathcal{N}_q)$  as training data, we tokenize and pass them to a sentence-bert [98] model  $\mathcal{M}_{RAG}$ , and fine-tune the model with the contrastive learning loss [28]:

$$\min \sum_{p \in \mathcal{P}_q} -\log \frac{\exp \left(\left\langle emb_q, emb_p \right\rangle / \tau \right)}{\sum_{p' \in \mathcal{P}_q \cup \mathcal{N}_q} \exp \left(\left\langle emb_q, emb_{p'} \right\rangle / \tau \right)},$$

where emb is an embedding and  $\tau$  is the temperature parameter.

Moreover, we serialize each query q to a dict format, which also contains meta-data for q, e.g., table title, column header; if  $\mathcal{N}_q = \emptyset$ , we conduct hard negative sample search with the initial model  $\mathcal{M}_{RAG}$  over  $\mathbb{X}$ , to add negative examples for  $\mathcal{N}_q$ .

<u>Self-Annotation.</u> When the training of  $\mathcal{M}_{RAG}$  is finished, we apply  $\overline{\mathcal{M}_{RAG}}$  to self-annotate unlabeled queries in  $\widetilde{\mathcal{X}}_i$ . *e.g.*, for EM, given

an unlabeled query  $q_i \in \widetilde{\mathcal{X}_{EM}}$ ,  $\mathcal{M}_{RAG}$  can search the most similar  $q_j$  over entire  $\mathbb{X}$  and self-annotate the entries in  $q_i$  and  $q_j$  as match. This procedure follows a self-supervised learning paradigm, and effectively enlarge the labeled data  $\mathcal{X}_i$  to  $\mathbb{X}_i$  by adding self-annotated data. Besides, we can also apply the transformation technique in Section 2.2 to further enlarge  $\mathbb{X}_i$  with labeled queries from other tasks. Figure 2 gives an example of both ways for enlarging labeled data.

Expert Initialization. For each task  $\mathcal{T}_i$ ,  $\mathbb{X}_i$  is used to initialize the training of each expert  $e_i$ , by fine-tuning a LLM, denoted by  $\mathcal{M}_G$ .

## 4.2 Heuristic Meta-path Search

There are a host of data augmentation methods [30, 76] for DP. However, such methods are either statistical or they use pre-defined global operators for augmentation. Alternatively, we consider a fixed set of experts  $\mathbf{E} = \{e_1, \dots, e_n\}$ , and find a meta-path (*i.e.*, a sequence of experts in  $\mathbf{E}$ ) for task  $\mathcal{T}_i$ . Such meta-path can help to augment data in  $\mathbb{X}_i$  reasonably. Below we define such expert-based meta-path and data augmentation over the meta-path.

**Definition 4.1:** (Meta-path over Experts): A meta-path  $\mathcal{E}_i$  for task  $\mathcal{T}_i$  is a sequence of experts  $e_{j_1}, \dots, e_{j_n}$  from the experts set  $\mathbf{E}$ ; it describes the order of experts to be applied for task  $\mathcal{T}_i$ .

**Definition 4.2:** (Data Augmentation Over Meta-path): Given  $\mathbb{X}_i$  for task  $\mathcal{T}_i$ , we denote  $\mathbb{X}_i^{j_1}$  as the augmented set of  $\mathbb{X}_i$  by querying expert  $e_{j_1}$ . Similarly,  $\mathbb{X}_i^{\mathcal{E}_i}$  is the augmented set of  $\mathbb{X}_i$  by a meta-path  $\mathcal{E}_i = \{e_{j_1}, \cdots, e_{j_n}\}$ , *i.e.*, by querying the experts in  $\mathcal{E}_i$  in order.  $\square$ 

We show an example in Figure 1, in which  $\mathbb{X}_{EM}$  is augmented by the meta-path  $\mathcal{E}_{EM} = \{e_{\text{blocking}}, e_{\text{DI}}, e_{\text{AVE}}, e_{\text{EM}}\}$  in order.

<u>Heuristic Meta-path search</u>. Given labeled data  $X_i$  for task  $\mathcal{T}_i$ , we want to find a sequence of experts  $\mathcal{E}_i = \{e_{j_1}, \dots, e_{j_n}\}$  such that the performance of the augmented data, *i.e.*,  $Eval(e_i, X_i^{\mathcal{E}})$ , is the best.

Here we apply a greedy search algorithm for finding a meta-path  $\mathcal{E}_i$  for task  $\mathcal{T}_i$ , which reduces the search space by incorporating user-defined sub-optimal paths, e.g.,  $\{e_{\mathsf{Blocking}}, e_{\mathsf{EM}}\}$  (resp.  $\{e_{\mathsf{EL}}, e_{\mathsf{CTA}}\}$ ) is widely used in EM (resp. tabular interpretation learning [27]).

After finding a meta-path  $\mathcal{E}_i$  for task  $\mathcal{T}_i$ , we can query  $\mathcal{E}_i$  to augment training data  $\mathbb{X}_i$  to  $\mathbb{X}_i^{aug}$  with self-supervised annotation.

## 4.3 Expert Refinement

Note that for each initialized expert  $e_i$  for task  $\mathcal{T}_i$ , there is a high risk that  $e_i$  may overfit to the biased distribution of  $X_i$ , since  $X_i^{aug}$  are augmented from  $X_i$  and may share a similar biased distribution. As discussed in [9], such distribution may lead to a higher empirical

error  $\epsilon_C$  ( $h_N$ ) in Theorem 2. To alleviate such concern, we introduce the Min-Max optimization target guided by the information bottleneck theory, to improve the generalizability of each expert  $e_i$ .

<u>Information Bottleneck.</u> Information bottleneck [109, 110] was used to balance the complexity of representation and the power of predicting, based on the notion of minimal sufficient statistics for extracting information about target Y from input X into representation Z. It imposes regularization at representation Z by minimizing the mutual information between input X and the learned representation Z, *i.e.*, min I(X;Z), while maximizing the mutual information between target output Y and X, *i.e.*, max I(Y;Z) [64].

In expert training, the information bottleneck theory provides useful insights: consider training expert  $e_i$  with training data  $(\mathbb{X}_i, \mathcal{Y}_i)$ ; it is equivalent to find the most relevant task vector  $\theta_i$  as representation. On the one hand, the distribution of  $\mathbb{X}_i$  should be diverse, a.k.a. minimize  $I(\mathbb{X}_i; \theta_i)$ . Otherwise  $e_i$  may overfit to a biased distribution of sampled training data  $\mathbb{X}_i$ , and cannot learn the high-level and intrinsic features. On the other hand, the distribution of  $\mathbb{X}_i$  should fall in the same cluster with  $\theta_i$  in ITS, as shown in Theorem 3, a.k.a. maximize  $I(\mathcal{Y}_i; \theta_i)$ . Otherwise  $e_i$  may suffer from underfitting issue with low performance, due to the lack of relevant information.

Training Process. We denote  $\theta_{\mathcal{M}_{RAG}}$  as the parameters for fine-tuned RAG model in Section 4.1, and  $\theta_{\mathcal{M}_G}$  as the parameter of the base LLM-model of each expert, and RAG( $\mathcal{X}_i$ )) as the operations we use to augment  $\mathcal{X}_i$ , including both self-annotation and meta-path augmentation. The optimization function of training LLM-based  $e_i$  is:

$$\arg\min_{\theta_{\mathcal{M}_{\mathsf{RAG}}}} \max_{\theta_{\mathcal{M}_{G}}} I(\mathcal{M}_{G}(X_{i}); \mathcal{M}_{G}(\mathsf{RAG}(X_{i}))) \tag{1}$$

Intuitively, (a)  $\max \theta_{\mathcal{M}_G}$  is explicitly conducted, by parameter-efficient fine-tuning  $\mathcal{M}_G$  and maximizing the mutual information between the output of  $\mathcal{M}_G$  and label  $\mathcal{Y}_i$ ; and (b)  $\min \theta_{\mathcal{M}_{RAG}}$  is implicitly enforced, by controlling the sample parameter for  $\mathcal{M}_{RAG}$  and meta-path  $\mathcal{E}_i$  and adding  $\Delta \mathcal{X}_i = RAG(\mathcal{X}_i)$  as supplement training data for  $\mathcal{M}_G$ , while minimizing the mutual information between the labeled training data  $\mathcal{X}_i$  and external training data  $\Delta \mathcal{X}_i$ .

In practice, we adopt a iterative optimization strategy to fulfill the target function. Specifically,  $\mathcal{M}_G$  is initialized with expert  $e_i$  (Section 4.1). Then we iteratively control RAG( $\mathcal{X}_i$ ) to add diverse training data  $\Delta \mathcal{X}_i$  by extracting cross-domain examples and demonstrations, as well as implementing data augmentation with meta-path  $\mathcal{E}_i$ . After adding  $\Delta \mathcal{X}_i$  to  $\mathcal{X}_i$ , we further fine-tune  $\mathcal{M}_G$  with new data until convergence. Such iterations continue  $\sigma$  times.

After refinement,  $e_i$  is refined to  $e_i^{aug}$ , which is more robust to various DP tasks and cross-domain queries, while retaining high performance on its own  $\mathcal{T}_i$ . Denote the set of refined experts by  $\mathbf{E}^{aug}$ . We apply low-rank adaptation [56] (a.k.a. LoRA) to fine-tune  $\mathcal{M}_G$  for training and refining each expert  $e_i \in \mathbf{E}^{aug}$ .

#### 4.4 Router Network

In this section, we train a light-weighted sparse-gated router network N to select top-k experts in  $\mathbf{E}^{aug}$  for each input query.

The information bottleneck theory also provides insight in optimizing  $\mathcal{N}$ . Given query  $q_i$ , on the one hand, the selected top-k experts should be diverse to provide different yet valuable views of  $q_i$ ; this is equivalent to minimize the mutual information between the

<span id="page-5-2"></span>![](_page_5_Figure_13.jpeg)

Figure 5: Model architecture of MELD

selected experts. On the other hand, the selected experts should be relevant to  $q_i$ ; this is equivalent to maximize the mutual information between the output of selected experts and corresponding labels.

<u>Router Network.</u> Given a labeled query  $q_u \in \mathbb{X}_u^{aug}$ , let  $\mathcal{N}(q_u)$  be the top-k experts selected by the sparse gated network  $\mathcal{N}$  for  $q_u$  and task  $\mathcal{T}_u$ , and  $(q_u^i, l_u^i)$  be the transformed query-label pair from task  $\mathcal{T}_u$  to  $\mathcal{T}_i$  with self-annotation. The optimization function is:

<span id="page-5-1"></span>
$$\max \sum_{e_i \in \mathcal{N}(q_u)} I(e_i(q_u^i); l_u^i); \min \sum_{e_i, e_j \in \mathcal{N}(q_u)}^{i \neq j} I(e_i(q_u^i); e_j(q_u^j)) \quad (2)$$

In practice, Eq.2 can be approximated with contrastive training loss [92, 106]. Thus, we apply a transformer network that shares the encoding layers with  $\mathcal{M}_{RAG}$ , for  $\mathcal{N}$  and further fine-tune it with contrastive loss. The positive and negative examples are extracted from labeled data across all tasks. Figure 5 gives an illustration of  $\mathcal{N}$ .

#### <span id="page-5-0"></span>5 EXPERIMENTAL STUDY

Our experiments focus on answering the following questions:

- How does MELD perform compare with other non-LLM methods and local-LLM methods, especially in few-shot scenarios?
- How does MELD benefit from the MoE architecture design, especially in cross-dataset and cross-task scenarios?
- The effectiveness and efficiency comparison between the lightweighted standalone router network architecture, e.g., MELD, and the built-in MoE layer based model, e.g., Mixtral 8×7B?
- How does the number of experts, as well as the meta-path selection, affect the overall performance of MELD?

#### 5.1 Setup

<u>Statistics.</u> As shown in Table 6 in Appendix A.1, as well as the abbreviation of each DP tasks. We selected 19 datasets over 10 typical DP tasks to show the performance of MELD. In all tasks except schema matching, we use few-shot labeled data (usually  $\leq$  10%), as shown in column #Instance (few-shot). The selection of few-shot examples are kept the same among all methods.

<u>Methods.</u> We categorized the baselines as follows. (1) Non-LLM methods . (a) ED: Raha[83], (b) DI: IPM[86], (c) Blocking: DeepBlocker[107], (d) EM: Ditto[76] and PromptEM[112], (e) DC: Baran[82] and Garf[95], (f) CTA: RECA[32], (g) RE/EL: TURL[27], (h) SM: CONSchema[116] and SMAT[128], and (k) AVE: MAVE[120]. Other methods, *e.g.*, HoloClean [99], DODUO [31] have been shown to be outperformed by the listed competitors [32, 83], and hence not compared. (2) LLM-based methods. JellyFish[126] uses a 13B LLM model (1.8× than MELD) to solve multiple DP tasks. For table interpretation tasks (*e.g.*, CTA, RE, EL), we compared TableLLaMa[131] which applies a 7B foundation model. For AVE task, we used

**Table 1: Overall Performance** 

<span id="page-6-1"></span>

|         |                        |                  |                                 | -                           |                     |
|---------|------------------------|------------------|---------------------------------|-----------------------------|---------------------|
| Task    | Dataset                | MELD<br>Few-shot | Non-LLM<br>Baseline<br>Few-shot | LLM<br>Baseline<br>Few-Shot | Mixtral<br>Few-shot |
|         | Amazon-<br>Google      | 83.41(74.12)     | 61.88(50.47)                    | 65.98(/)                    | 51.28(/)            |
| EM<br>& | Walmart-<br>Amazon     | 91.42(78.80)     | 79.09(58.21)                    | 42.03(/)                    | 39.78(/)            |
| (BLK)   | WDC-All                | 91.97(31.50)     | 34.35(1.70)                     | 49.80(/)                    | 48.97(/)            |
|         | Ant-Buy                | 91.12(86.20)     | 84.89(40.66)                    | 71.40(/)                    | 60.42(/)            |
|         | Semi-Text-<br>Watch    | 78.28(59.23)     | 23.60(2.66)                     | 54.27(/)                    | 40.55(/)            |
|         | Semi-Text-<br>Computer | 86.46(30.85)     | 33.90(8.09)                     | 76.80(/)                    | 73.15(/)            |
|         | Hospital               | 95.01            | 67.10                           | 49.30                       | 53.20               |
| DC      | Rayyan                 | 82.15            | 28.50                           | 9.39                        | 6.68                |
|         | Beer                   | 97.30            | 90.31                           | 51.30                       | 56.27               |
|         | Hospital               | 98.51            | 95.23                           | 89.41                       | 69.14               |
| ED      | Rayyan                 | 90.37            | 80.21                           | 69.67                       | 31.96               |
|         | Beer                   | 99.10            | 100.00                          | 81.64                       | 70.23               |
| СТА     | SemTab19               | 89.35            | 69.70                           | 87.77                       | 89.35               |
| _ LIA   | WebTables              | 96.30            | 90.93                           | 94.77                       | 80.16               |
| RE      | WikiGS-RE              | 89.30            | 73.50                           | 60.38                       | 65.88               |
| EL      | WikiGS-EL              | 87.05            | 60.55                           | 82.20                       | 73.25               |
| SM      | CMS                    | 60.27            | 50.00                           | 59.29                       | 31.01               |
|         | Synthea                | 56.00            | 38.50                           | 40.00                       | 23.53               |
| DI      | Walmart                | 87.50            | 65.70                           | 57.69                       | 79.82               |
|         | Amazon                 | 75.12            | 60.35                           | 60.05                       | 62.62               |
|         | Restaurant             | 93.10            | 37.50                           | 68.97                       | 72.41               |
| AVE     | OA-mine                | 74.62            | 67.00                           | 65.70                       | 77.36               |
| AVE     |                        |                  |                                 |                             |                     |

ExtractGPT[6], compared to its local LLM model with up to 70B parameters (10× than MELD). (3) MoE models. We compared the state-of-the-art MoE foundation model Mixtral-8×7B[62] (*i.e.*, Mixtral), which embeds the MoE layer  $\mathcal N$  in model parameters, and jointly train  $\mathcal N$  with a set E of 8 experts, each of which is a 7B LLM.

<u>Default Parameters.</u> For Blocking, ED and EL, we only apply our RAG model  $\mathcal{M}_{RAG}$  due to the large search space. For other tasks, we uses the LLM-based MoE system. Default number of k is set to 3, the number of iterations  $\sigma$  for expert refinement is set to 3, the demonstration number  $|D_i|$  is set to 8.  $\tau$  in RAG is 0.02. Detailed implementation is listed in Appendix A.2 and full version[1].

<u>Metrics</u> To evaluate DP tasks, we measured accuracy for DI, AVE; top-1 accuracy for EL; top-1 recall for blocking; F1 score for EM, ED, DC, SM, and micro-F1 score for CTA, RE tasks in a 100-scale.

<u>Environment</u> We select bge-large-en[34] as the backbone for the RAG models  $\mathcal{M}_{RAG}$ , and Mistral-7B[61] as the backbone of expert model by default. We conducted the experiments on a single machine powered by 256GB RAM and 32 processors with Intel(R) Xeon(R) Gold 5320 CPU @2.20GHz and 4 Nvidia GeForce RTX 3090 GPUs. Each experiment was run 3 times and the average is reported.

#### <span id="page-6-0"></span>5.2 Effectiveness Evaluation

We compared the performance of MELD with various non-LLM and LLM baselines in Table 1. In few-shot scenarios, MELD consistently

<span id="page-6-2"></span>![](_page_6_Figure_10.jpeg)

Figure 6: Efficiency among different LLMs-based models (4-bit quantization for Jellyfish and Mixtral on  $1 \times 3090$ )

outperforms all non-LLM baselines, which means that MELD has better data utilization. In particular, 10%-20% labeled training data suffices to train a robust expert  $e_i$  for task  $\mathcal{T}_i$ , while the shared parameter from other experts can prevent  $e_i$  from being overfitting.

In low-resource settings where labeled data is extremely limited, LLM baselines are prone to issues such as overfitting and hallucination problem, due to insufficient relevant demonstration data[135]; While non-LLM baselines often utilize rule-based approaches or rely on structural information, and are inherently robust in few-shot scenarios. MELD compensate such information incompleteness with  $\mathcal{M}_{RAG}$  and self-distilled data augmentation with meta-path.

Compared to LLM baselines, which are trained over MTL paradigm, MELD beats them with significant fewer parameters. This indicates that the MoE architecture is good at handling MTL, and multiple sparse experts can outperform one dense one. Besides, we argue that several LLM baselines, including Jellyfish and TableL-LaMa, require high-cost pre-training over enormous task-specific corpus with thousands of GPU hours (e.g., millions of Wikipedia webtables[131]), while MELD only needs low-cost fine-tuning for training each expert from a base model with less than 20 GPU hours.

Compared to Mixtral, which also applies a build-in MoE layer, we can see that Mixtral outperforms MELD in a few tasks (i.e., AVE, CTA). However, Mixtral fails to apply a good routing strategy, and Mixtral does not balance the load well for the task family  $\mathcal T$  to its 8 experts, leading to its better performance in open-domain/complex tasks with long context and information retrieval, e.g., DI, AVE, and poor performance in close-domain/simple tasks, e.g., EM, DC.

#### 5.3 Efficiency Evaluation

We compared the efficiency of MELD, Jellyfish and Mixtral in Figure 6, comparing the inference throughput speed and model process time. This comparison is conducted on two settings: 4×3090 GPUs and 1×3090 GPU with vLLM [68]. Due to the VRAM requirement of Mixtral, we only report its performance on the former.

Firstly, we report the throughput over 4 GPUs with vLLM [68]. Due to the small size of experts in MELD, a single 3090 GPU can hold a maximum of 16 experts for MELD, while the load-balance system of MELD and vLLM can gather similar queries within the same GPU. Therefore, MELD achieves data parallelism over 4 GPUs, and gain non-trivial 3.7× throughput improvement with 13B Jellyfish and 5.6× with 56B Mixtral, which have to apply tensor parallelism, and suffer from the communication overhead over multiple GPUs.

Secondly, we report the throughput over a single GPU, a prevalent consumer scenario. MELD perform well with full precision model, while Jellyfish has to apply a 4-bit quantization [42] to make

Table 2: Cross-Dataset(C-D) and Cross-Task(C-T)

<span id="page-7-1"></span>

| Task | Dataset         | MELD<br>C-D | MELD<br>C-T | LLM<br>Baseline<br>C-D | LLM<br>Baseline<br>C-T | Mixtral<br>C-D | Mixtral<br>C-T |
|------|-----------------|-------------|-------------|------------------------|------------------------|----------------|----------------|
| EM   | Amazon-Google   | 69.05       | 67.95       | 18.58                  | 18.58                  | 43.23          | 43.23          |
|      | Semi-Text-Watch | 65.07       | 51.13       | 20.52                  | 20.51                  | 37.12          | 37.12          |
| CTA  | SemTab19        | 76.84       | 61.21       | 15.79                  | 7.96                   | 64.83          | 61.64          |
|      | WebTables       | 86.76       | 88.95       | 38.92                  | 14.29                  | 79.72          | 67.64          |
| DI   | Walmart         | 54.80       | 54.80       | 43.26                  | 17.86                  | 79.82          | 78.85          |
|      | Restaurant      | 75.86       | 75.86       | 68.96                  | 6.95                   | 72.43          | 58.62          |

**Table 3: Performance for Ablation Study** 

<span id="page-7-2"></span>

| Task | Dataset                                                                                        | MELD<br>w/o MoE                           | MELD<br>w/o RAG                                    | MELD<br>w/o Meta-Path                              | MELD<br>with Mixtral                               |
|------|------------------------------------------------------------------------------------------------|-------------------------------------------|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------|
| EM   | Amazon-Google<br>Walmart-Amazon<br>WDC-All<br>Ant-Buy<br>Semi-Text-Watch<br>Semi-Text-Computer | 76.70<br>87.66<br>90.38<br>87.58<br>70.78 | 69.21<br>81.44<br>83.16<br>85.75<br>55.07<br>42.02 | 62.52<br>79.55<br>91.73<br>90.12<br>39.89<br>63.74 | 77.85<br>91.03<br>91.32<br>85.26<br>75.42<br>81.98 |

inference on a single GPU, and Mixtral cannot deploy on a single GPU even with 4-bit quantization, due to OOM issues. Although MELD is around 1.3× slower than 4-bit Jellyfish, the quantization is time-consuming and it leads to a significant performance drop.

We also report the model process time of each methods, *i.e.*, the time of merging trained LoRAs into the base model and preparing it for inference with vLLM. MELD applies a dynamic LoRA switch technique, which avoids merging multiple LoRA into a single model, and only needs to load and concatenate on multiple LoRAs, reducing the i/o cost. While Jellyfish and Mixtral have to apply a time-consuming merging and quantization operation. As a result, MELD is  $10\times$  and  $30\times$  faster than Jellyfish and Mixtral in model process.

#### 5.4 Cross-Dataset and Cross-Task Comparison

We evaluated the cross-dataset (i.e., C-D) and cross-task (i.e., C-T) performance of MELD, where C-D means we mask expert  $e_i$  and training data  $X_i$  for task  $\mathcal{T}_i$ , while C-T means we mask all experts and training data that are same as  $\mathcal{T}_i$  (e.g., mask all EM experts for evaluation on the Amazon-Google dataset). The result is presented in Table 2. To prevent the domain overlap, we select 6 datasets with different domains, and limit the overall experts of MELD into 6.

Compared with LLM baselines, MELD suffers less performance drop in C-D and C-T scenarios, which is contributed by the information bottleneck guided expert training, as well as the RAG system across datasets and tasks. Nonetheless, Mixtral also performs well in open-domain tasks, which means the MoE architecture is suitable in MTL. Besides, the shared parameters of experts in MELD and Mixtral effectively prevent them from being overfitting to few-shot data and specific task, or suffering hallucination problems.

#### 5.5 Ablation Study

We selected EM for ablation study, varying the following in Table 3:

- o MELD w/o MoE, a single expert fine-tuned per task;
- $\circ\,$  MELD w/o RAG, where each expert is fine-tuned without cross-domain data augmentation and RAG; and
- MELD-w/o Meta-Path, where each expert is fine-tuned without meta-path based data augmentation

<span id="page-7-4"></span>Table 4: Performance for Different LLM parameter size

| Task | Dataset           | F1-Score<br>(Mistral-7B) | F1-Score<br>(Vicuna-33B) | Train Time<br>(7B/33B) | Inference Speed<br>(7B/33B) |
|------|-------------------|--------------------------|--------------------------|------------------------|-----------------------------|
| EM   | Amazon-<br>Google | 83.14                    | 84.17                    | 2944/8389              | 19.56/2.93                  |
| DC   | Rayyan            | 82.15                    | 80.62                    | 1275/5494              | 27.24/6.10                  |
| CTA  | SemTab19          | 89.35                    | 87.77                    | 2792/5821              | 25.28/7.33                  |
| RE   | WikiGS-RE         | 89.30                    | 83.88                    | 501/2174               | 54.52/15.81                 |

Table 5: Performance compared with GPT-4

<span id="page-7-3"></span>

| Task | Dataset                         | MELD<br>Few-shot | GPT-4          | LLM<br>Baseline<br>Few-Shot | Mixtral<br>Few-shot |
|------|---------------------------------|------------------|----------------|-----------------------------|---------------------|
| EM   | Amazon-Google<br>Walmart-Amazon | 83.41<br>91.42   | 74.21<br>90.27 | 65.98<br>42.03              | 51.28<br>39.78      |
|      | Ant-Buy                         | 91.12            | 92.77          | 71.40                       | 60.42               |
| SM   | CMS                             | 60.27            | 59.29          | 59.29                       | 31.01               |
| SIVI | Synthea                         | 56.00            | 66.67          | 40.00                       | 23.53               |
| DI   | Restaurant                      | 93.10            | 97.75          | 68.97                       | 72.41               |
| AVE  | OA-mine                         | 74.62            | 80.20          | 65.70                       | 77.36               |

With only a domain-specific expert  $e_i$ , MELD w/o MoE, is not the good solution, since different experts can provide additional information to boost the performance. MELD w/o RAG suffers from performance drop over all scenarios, justifying the effectiveness of  $\mathcal{M}_{RAG}$ . For semi-structured or low-quality data, e.g., semi-text-w and amazon-google, the meta-path can augment structural information and significantly improve the performance. As remarked earlier, if we replace the router network  $\mathcal N$  with Mixtral, it also suffers performance drop, due to unbalance load between experts in Mixtral.

#### 5.6 Hyper-parameter Analysis

We tested the impact of k and the distribution of task vectors  $\Theta$  and attention weights across experts in E and tasks in  $\mathcal{T}$ . Following [54], we present the t-SNE plot figure in Figure 3, to visualize the embeddings generated by E and router network  $\mathcal{N}$ . This proves that  $\mathcal{N}$  dispatch queries based on the latent distribution of  $\Theta$ .

Figure 7 provides the sensitive analysis of number k. Initially, the performance rises with the number of experts. However, when  $k \ge 4$ , the overall performance shows a slight drop, while the parameter size still increases. This justifies that involving more experts are not always good, since their inherit parameters may conflict.

We provide the attention weights across experts in full version[1]. The utilization rate of experts diverged significantly across tasks and datasets. We also provide the comparison of MELD and online GPT-4 in Table 5, following the best performance in [6, 126]. Similar to Mixtral, GPT-4 shows better ability in complex/open-domain tasks.

In Table 4, we compare the performance with different backbone model for expert, and evaluate their performance, training/inference speed, and cost under the same conditions across several representative tasks and datasets. Our findings indicate that utilizing a larger model results in modest performance improvements but incurs significantly higher training and inference costs. As discussed in [85, 135], increasing the model size does not necessarily translate to enhanced performance in DP tasks. In light of our low-resource setting, we currently limit our base model parameter size to 7B.

#### <span id="page-7-0"></span>6 RELATED WORKS

We briefly review the related works as follows.

<span id="page-8-1"></span>![](_page_8_Figure_2.jpeg)

Figure 7: Performance for di"erent number of experts :

