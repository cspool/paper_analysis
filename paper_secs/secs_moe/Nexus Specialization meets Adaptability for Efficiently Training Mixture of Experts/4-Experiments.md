# 4 Experiments

#### 4.1 Experimental setting

Our experimental setup includes 3 phases. Figure [1](#page-1-0) shows the architecture of Nexus and the corresponding experimental setting:

1. Training specialized expert LMs. For training the dense specialized experts, we use the sub-datasets from the SlimPajama dataset [\[Soboleva et al.,](#page-21-7) [2023\]](#page-21-7), a 627B token English-language corpus assembled from web data of various sources. We initialize four dense experts from the weights of the seed model and train them on the ArXiv, Books, C4, GitHub, StackExchange,

![](_page_6_Figure_0.jpeg)

Figure 3: Downstream performance at different scales: Nexus consistently outperforms upcycled baselines on both the 470M and 2.8B parameters scale, showing the robustness of our method. We report the average performance on Knowledge, Science, Reasoning and MMLU.

and Wikipedia domains.[2](#page-6-0) . As seed model, we use a 470M and 2.8B parameters decoder-only autoregressive Transformer models [\[Radford et al.,](#page-19-4) [2019\]](#page-19-4) that are trained with a standard language modeling objective for 750B tokens. We train dense experts for 25 and 40 billion tokens for 470M and 2.8B seed models respectively. We use parallel attention layers, [\[Anil et al.,](#page-15-1) [2023;](#page-15-1) [Wang,](#page-21-8) [2021\]](#page-21-8), SwiGLU activation [\[Shazeer,](#page-20-5) [2020\]](#page-20-5), no biases in dense layers, and a byte-pair-encoding (BPE) tokenizer with a vocabulary size of 256,000. During training, we use a linear warmup (10% of total steps) to a maximum learning rate of 1e-3 and a cosine decay schedule to 3e-4.

- 2. MoE training. After the training of dense expert models, we merge them into a unified MoE by appending their FFNs along a new dimension to create an MoE layer per Transformer block. For the shared expert in our MoE layer, we use the original FFN layer of the seed model to better preserve the previous capabilities in the MoE model. For all non-FFN parameters including the attention weights, we merge expert parameters using simple weight averaging, following [Sukhbaatar et al.](#page-21-2) [\[2024\]](#page-21-2). After the MoE model is created, we continually train it for an additional 25B and 40B tokens respectively for the 470M and 2.8B experiments, on a mix of all domain and original pre-training datasets, using the same training hyperparameters as in the single expert training. Finally, we train the MoE models using an additional 1B tokens by upweighting the original pre-training dataset as it includes high-quality data sources such as instruction-style datasets using a cosine learning rate decay to 3e-5 [\[Parmar et al.,](#page-19-5) [2024\]](#page-19-5).
- 3. Extending the MoE model with new experts. After adding a new expert as defined in Section [3,](#page-5-0) we finetune the extended MoE model for up to 1 billion tokens using a uniformly sampled data mix consisting of 50% the previous domains and pre-training data and 50% the new domain. For the new expert (Code), we train a dense model using code documents from StarCoder [\[Li et al.,](#page-18-4) [2023\]](#page-18-4) with the same settings as for the training of the initial experts. As the 470M scale MoE did not have sufficient instruction following capabilities to attempt the code benchmarks, we only tested extending the MoEs with a new expert on the 2.8B scale.

<span id="page-6-0"></span><sup>2</sup>We exclude the Github and StackExchange datasets from SlimPajama in order to ablate adding a new expert model using the Code domain

#### 4.2 Baselines

We compare our experiments against two baselines:

- 1. Dense Merging: We compare MoE variants against merging all separately pre-trained experts and the seed model into a dense Transformer via equal weight averaging similar to BTM [\[Li](#page-18-1) [et al.,](#page-18-1) [2022\]](#page-18-1). This allows us to ask What are the benefits of routing MoE over simple averaging?
- 2. MoE (Linear Router): To evaluate Nexus's novel router for upcycling, we compare it against an MoE with a standard linear router that is upcycled from dense experts. Here, we ask how does our specialized routing compare to conventional learned linear routing? For a fair comparison, we also train this MoE model on the same datasets and for the same number of tokens as our method, and use the same architectural modifications such as shared experts.

#### 4.3 Evaluation

For the downstream evaluation, we measure the performance of each model on 15 tasks[3](#page-7-0) from five evaluation categories that reflect different capabilities based on the tasks and the datasets used in the benchmarks:

- Knowledge: To measure question-answering capabilities based on world knowledge and web documents such as Wikipedia, we report the performance on OpenBookQA [\[Mihaylov et al.,](#page-19-6) [2018\]](#page-19-6), Natural Questions [\[Kwiatkowski et al.,](#page-18-5) [2019\]](#page-18-5), TriviaQA [\[Joshi et al.,](#page-18-6) [2017\]](#page-18-6), QUAC [\[Choi](#page-16-4) [et al.,](#page-16-4) [2018\]](#page-16-4) (all 0-shot) and SQuAD (4-shot) [\[Rajpurkar et al.,](#page-20-6) [2016\]](#page-20-6).
- Science: For measuring knowledge in science-oriented academic benchmarks, we use ARC-Easy, ARC-Challenge [\[Clark et al.,](#page-16-5) [2018\]](#page-16-5), SciQ [\[Welbl et al.,](#page-22-3) [2017\]](#page-22-3) (all 0-shot).
- Reasoning: For reasoning abilities, we use CommonSenseQA [\[Talmor et al.,](#page-21-9) [2019\]](#page-21-9), SIQA [\[Sap](#page-20-7) [et al.,](#page-20-7) [2019\]](#page-20-7), PIQA [\[Bisk et al.,](#page-15-2) [2020\]](#page-15-2), WinoGrande [\[Sakaguchi et al.,](#page-20-8) [2019\]](#page-20-8), and HellaSwag [\[Zellers et al.,](#page-22-4) [2019\]](#page-22-4) (all 0-shot).
- General Language Understanding: We use MMLU (5-shot) [\[Hendrycks et al.,](#page-17-8) [2021\]](#page-17-8) to test general language understanding.
- Code: For code generation, we evaluate models on MBPP [\[Austin et al.,](#page-15-3) [2021\]](#page-15-3), LBPP [\[Matton](#page-19-7) [et al.,](#page-19-7) [2024\]](#page-19-7) and HumanEval-Pack [\[Chen et al.,](#page-16-6) [2021\]](#page-16-6) that includes Cpp, Javascript, Java, Go, Python, and Rust (all 0-shot).

