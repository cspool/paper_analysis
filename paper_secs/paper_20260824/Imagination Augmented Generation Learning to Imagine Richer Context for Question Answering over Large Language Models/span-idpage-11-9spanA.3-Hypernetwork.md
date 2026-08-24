# <span id="page-11-9"></span>A.3 Hypernetwork

Hypernetworks have gained significant attention in recent years due to their potential to enhance various aspects of neural network performance. In this section, we analyze the reasons for employing hypernetworks in detail:

Hypernetworks [\(Ha et al.,](#page-9-13) [2016\)](#page-9-13) offer a solution that reduces the dependency on gradient descent for specific domains. Methods such as Hypertuning [\(Phang et al.,](#page-10-9) [2022\)](#page-10-9) and HINT [\(Ivison et al.,](#page-9-14) [2023\)](#page-9-14) use hypernetworks to transform inputs into parameter-efficient modules, thereby reducing computation and enhancing model generalization.

Hypernetworks, which are neural networks designed to generate the weights of other networks, allow for dynamic adjustment of model parameters. This adaptability enables the model to better suit different tasks and datasets, thereby improv-

<span id="page-12-2"></span>

|     | Document Relevance | Context Length | Inference Time | Inference Dependence       |
|-----|--------------------|----------------|----------------|----------------------------|
| RAG | Medium             | Too Long       | Very High      | Retriever                  |
| GAG | High               | Long           | High           | Larger Model (InstructGPT) |
| AAG | High               | Short          | Low            | None                       |

Table 5: Comparison of Different Paradigms

ing overall performance. By utilizing hypernetworks, the number of models that need to be trained individually can be significantly reduced. Traditional methods require separate models for each task, whereas hypernetworks can generate weights for multiple tasks. This capability enhances training efficiency. In our task, we use hypernetworks to generate adapters for the question and input, which are then inserted into the model. This helps the model incorporate the knowledge targeted by the question, corresponding to implicit awakening. Compared to traditional efficient fine-tuning, this process is more aligned with the goal of awakening.

Hypernetworks can capture the commonalities and differences between various tasks by learning to generate weights. This ability to generalize across tasks improves the model's performance on unseen data, making it more robust in diverse scenarios. In multi-task learning or meta-learning scenarios, hypernetworks can considerably reduce the need for storing multiple independent models. A hypernetwork only needs to store a single generating network and some shared parameters, thus significantly decreasing the storage space required. Hypernetworks can quickly generate new weights to adapt to new tasks as they arise. This rapid adaptation capability is particularly useful in applications that require frequent updates or expansions. In our experiments 4.5, we also found that using a hypernetwork can significantly enhance the generalization ability for tasks. This is because it not only retains knowledge within the domain-specific modules but also learns to generate question-targeted knowledge to be inserted into the model.

#### **Experimental Settings**

#### <span id="page-12-0"></span>**B.1** Background

Our task formulation follows retrieval augmented models for QA (Guu et al., 2020; Sachan et al., 2021). Let  $\mathcal{V}^*$  denote the infinite set, encompassing all potential strings over the tokens in vocabulary V, and this includes the empty string. An

instance within a QA dataset is defined as a triplet (q, a, c) comprising question q, answer a, and context c, where  $q, q, c \in \mathcal{V}^*$ . Conventionally, the context c is drawn from the knowledge corpus  $\mathcal{Z}$ , like Wikipedia, whereby  $\mathcal{Z} \subset \mathcal{V}^*$ .

The goal of QA is to learn a distribution function, represented as p(a|q), wherein the models decode a string a that serves as an abstractive answer to a given query q. In a closed-book setting, LLMs directly encode the given question and predict the answer (Roberts et al., 2020b). Specifically, considering the context c as the empty string, the reliance is solely on the model parameters, i.e.,  $\hat{a} = \arg \max_{a \in \mathcal{V}^*} p(a|q,\theta)$ , where  $\theta$  represents the LLMs' parameters. However, employing a direct approach of requesting models to output answers frequently results in subpar performance, primarily attributable to omitting a substantial amount of world knowledge during the process. Therefore, a popular approach is open domain setting, which marginalizes  $p(\boldsymbol{a}|\boldsymbol{q},\boldsymbol{c})$  over contexts c in the knowledge corpus (Lewis et al., 2020; Sachan et al., 2021) or generated from models (Yu et al., 2023). Given the computational infeasibility of calculating probabilities for all contexts, p(a|q,c) is approximated to the sum of probabilities for top k con- $\sum_{\boldsymbol{c} \in \text{Topk}(\boldsymbol{q})}^{\boldsymbol{c}_i \in \boldsymbol{c}} p(\boldsymbol{a} | \boldsymbol{q}, \boldsymbol{c}_i) p(\boldsymbol{c}_i | \boldsymbol{q}),$ texts, i.e.,  $p(\boldsymbol{a}|\boldsymbol{q},\boldsymbol{c}) =$ where Topk(q) denotes the set of resulting top k

passages after the retrieval or generated with a query q.

#### <span id="page-12-1"></span>**Prompts for Explicit Imagine with LLMs**

The prompt for explicit awakening of the context generator to imagine a short dummy useful document is:

Imagine contexts based on the question: \n input \n Contexts: \n

Table 14 shows the full prompts for zero-shot results on LLM that we use for open domain QA: NQ, TQA, WQ.

<span id="page-13-1"></span>

| Models    | Docu-<br>ments         | Steps | Lr   | Batch<br>Size |
|-----------|------------------------|-------|------|---------------|
| T5        | 0                      | 40000 | 1e-4 | 8             |
| LoRA-Base | 0                      | 40000 | 5e-4 | 8             |
| AAG       | 0                      | 50000 | 1e-3 | 8             |
| LoRA-l    | 0                      | 40000 | 1e-4 | 4             |
| AAG-l     | 0                      | 50000 | 5e-4 | 4             |
| FiD-3b    | 0                      | 40000 | 1e-4 | 2             |
| LoRA-3b   | 0                      | 40000 | 1e-4 | 4             |
| AAG       | 0                      | 50000 | 1e-4 | 1             |
| LoRA-Base | $0^{\dagger}$          | 40000 | 5e-4 | 8             |
| AAG       | ${\rm O}^{\dagger}$    | 50000 | 1e-3 | 8             |
| LoRA-l    | ${\rm O}^{\dagger}$    | 40000 | 1e-4 | 4             |
| AAG-l     | ${\rm O}^{\dagger}$    | 50000 | 5e-4 | 4             |
| LoRA-3b   | $\mathrm{O}^{\dagger}$ | 40000 | 1e-4 | 2             |
| AAG-3b    | $\mathrm{O}^{\dagger}$ | 50000 | 1e-4 | 1             |
| AAG       | 10                     | 50000 | 5e-4 | 1             |
| AAG-l     | 10                     | 50000 | 5e-4 | 1             |
| FiD-3b    | 10                     | 40000 | 1e-4 | 1             |
| AAG-3b    | 10                     | 50000 | 1e-4 | 1             |

Table 6: Hyperparameter Settings.

#### <span id="page-13-0"></span>**B.3** Implementations

In this section, we describe the implementation of our experiments in detail, including the baseline methods, backbone models, and hyperparameters. Our model is built based on the T5 (Roberts et al., 2020a). Differing from fine-tuning all model parameters  $\theta$  of the updated Pre-trained Language Model (LLM), LoRA (Hu et al., 2021) freezes all pre-trained Transformer parameters and optimizes only the parameters of each LoRA adapter. We employ LoRA to train a parameter-efficient fine-tuning baseline. Drawing from this, our approach updates only the parameters of the Hypernetwork to generate the weights for each LoRA adapter. This method is adopted based on LongLoRA's (Chen et al., 2023) recommendations and experimental findings, demonstrating improved performance when the normalization and FFN layers components are updated. This is because: 1) dynamically generating LoRA weights enhances generalization and parameter sharing, and 2) LoRA performs comparably to fine-tuning but mitigates the risk of catastrophic forgetting.

For the baseline, most of the hyperparameters are the default parameters of FiD (Izacard and Grave, 2021). For LoRA (Hu et al., 2021), add the LoRA module only to the  $\mathcal{QV}$  of the attention layers and

also release the normalization and FFN layers.

We consider conducting experiments using three different sizes of T5, namely T5-base, T5-large, T5-3b, and Llama2-7B, Llama2-13B (Touvron et al., 2023). Due to memory constraints and online distillation limitations, A100 supports processing 20 documents for T5-3b, while Llama2 does not support distillation. All experiments with T5-3b are conducted on 2 A100 GPUs, T5-large on 2 A6000 GPUs, and T5-Base on 2 RTX 3090 GPUs. However, experiments with Llama2-7b and 13b, except for AAG on 2 A100 GPUs, are tested on 8 RTX 3090 GPUs.

#### **B.3.1** Hyperparameters

The detailed hyperparameter setting is as shown in Table 6. For the LoRA modules, we set the  $\alpha$  32 and the *lora rank* 32.

#### **B.3.2** Baselines

**DPR** (Karpukhin et al., 2020) generates by searching for the most relevant documents through dense vector space representation.

**FiD** (Izacard and Grave, 2021) retrieve relevant documents and send them separately to the Encoder, then fuse the information in the Decoder.

**RFiD** (Wang et al., 2023a) uses the encoder of FiD to distinguish between causal and incidental features, and guides the decoder to generate answers based on this distinction.

EAR (Chuang et al., 2023) significantly enhances the traditional sparse retrieval method BM25 by connecting query expansion models and retrievers. FILCO (Wang et al., 2023d) identifies useful context based on lexical and information-theoretic methods.

**GENREAD** (Yu et al., 2023) prompt LLMs like InstructGPT (Ouyang et al., 2022) to generate a large number of relevant documents and let the reader process them.

**LoRA** We use LoRA (Hu et al., 2021) to obtain an efficiently fine-tuned baseline and compare it with our method.

#### **B.3.3** Evaluation

For QA datasets, we choose the exact match (EM) score (Rajpurkar et al., 2016) as the evaluation metric. An answer is deemed correct if it aligns with any of the responses in the list of acceptable answers after normalization. Normalization involves transforming the text into lowercase, omitting articles, punctuation, and eliminating redundant spaces.

<span id="page-14-4"></span>

|           |        |       | NQ    |       |       | TQA   |       |       | WQ    |       |
|-----------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Models    | # Docs | NQ    | TQA   | WQ    | NQ    | TQA   | WQ    | NQ    | TQA   | WQ    |
| T5        | 0      | 22.16 | 3.18  | 4.12  | 2.65  | 21.8  | 3.15  | 0.88  | 2.95  | 28.3  |
| LoRA-Base | 0      | 16.17 | 4.71  | 6.89  | 3.15  | 21.16 | 0.00  | 1.33  | 3.04  | 26.38 |
| AAG       | 0      | 23.89 | 6.21  | 10.94 | 5.31  | 22.69 | 6.30  | 3.23  | 5.10  | 30.31 |
| LoRA-Base | 1†     | 37.17 | 45.20 | 15.62 | 19.57 | 55.37 | 12.50 | 14.15 | 30.89 | 28.88 |
| AAG       | 1†     | 40.14 | 46.61 | 18.92 | 24.78 | 60.75 | 12.82 | 17.70 | 35.24 | 41.06 |
| FiD       | 10     | 46.81 | 53.93 | 24.02 | 28.57 | 63.32 | 17.83 | 18.81 | 41.88 | 41.78 |
| AAG       | 10     | 47.01 | 55.74 | 24.13 | 31.77 | 64.95 | 19.52 | 24.43 | 48.10 | 46.36 |
| T5-l      | 0      | 28.5* | 3.18  | 4.12  | 2.65  | 28.7* | 3.15  | 0.88  | 2.95  | 30.6* |
| LoRA-l    | 0      | 17.70 | 7.49  | 8.66  | 3.54  | 23.87 | 4.72  | 0.00  | 5.65  | 29.13 |
| AAG-l     | 0      | 29.32 | 10.17 | 14.06 | 7.02  | 30.11 | 7.81  | 2.65  | 7.06  | 32.68 |
| LoRA-l    | 1†     | 37.61 | 48.50 | 20.71 | 20.54 | 62.71 | 14.81 | 15.36 | 33.83 | 39.37 |
| AAG-l     | 1†     | 42.32 | 54.80 | 22.05 | 26.11 | 65.48 | 18.11 | 18.58 | 47.46 | 45.28 |
| FiD-l     | 10     | 46.7* | 57.93 | 25.12 | 34.29 | 61.9* | 19.64 | 27.65 | 53.87 | 48.1* |
| AAG-l     | 10     | 49.92 | 60.03 | 25.79 | 34.35 | 69.67 | 20.28 | 30.19 | 54.94 | 51.52 |

Table 7: OOD results. The primary row in the table header delineates the dataset trained, while the underscored secondary row demonstrates the in-distribution performance. AAG attains optimal performance both in-distribution and OOD under diverse document configurations.

<span id="page-14-3"></span>

| Dataset | Train  | Dev   | Test   |
|---------|--------|-------|--------|
| WebQ    | 3,417  | 361   | 2,032  |
| NQ      | 79,168 | 8,757 | 3,610  |
| TQA     | 78,785 | 8,837 | 11,313 |

Table 8: Open-Domain QA dataset statistics

