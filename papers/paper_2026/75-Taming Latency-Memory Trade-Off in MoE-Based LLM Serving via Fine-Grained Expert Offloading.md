# **Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading** 

Xingqi Cui[*] Hong Zhang Rice University University of Waterloo Texas, USA Ontario, Canada xc66@rice.edu hongzhangblaze@gmail.com 

## Hanfei Yu 

Stevens Institute of Technology New Jersey, USA hyu42@stevens.edu 

## Hao Wang 

Hao Wang 

Rutgers University New Jersey, USA hw488@cs.rutgers.edu 

Stevens Institute of Technology New Jersey, USA hwang9@stevens.edu 

## **Abstract** 

_**Keywords:**_ Artificial Intelligence, Large Language Model, Mixture-of-Experts, Model Serving, Offloading 

Large Language Models (LLMs) have gained immense success in revolutionizing various applications, including content generation, search and recommendation, and AI-assisted operations. To reduce high training costs, Mixture-of-Experts (MoE) architecture has become a popular backbone for modern LLMs. However, despite the benefits, serving MoE-based LLMs experience severe memory inefficiency due to sparsely activated experts. Recent studies propose to offload inactive experts from GPU memory to CPU memory to improve the serving efficiency of MoE models. However, they either incur high inference latency or high model memory footprints due to coarse-grained designs. 

## **ACM Reference Format:** 

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang. 2026. Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading. In _21st European Conference on Computer Systems (EUROSYS ’26), April 27–30, 2026, Edinburgh, Scotland Uk._ ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3767295.3769319 

## **1 Introduction** 

Large Language Models (LLMs) have achieved remarkable success in advancing Natural Language Processing (NLP) research and transforming various applications, including content generation [2, 7, 12, 45], search and recommendation [34, 63], and AI-assisted operations [24, 33, 39]. Given the high training costs, modern LLMs have returned to Mixture-ofExperts (MoE) architectures [1, 11, 23, 50, 57, 60] as their backbone implementations. Inside MoE models, each MoE layer comprises a gating network and a collection of experts, with only a subset of experts being activated during computation. This sparse activation mechanism significantly reduces the number of floating point operations (FLOPs), enabling MoE-based LLMs to achieve substantially lower training costs compared to dense LLMs [11, 23, 60]. 

To tame the latency-memory trade-off in MoE serving, we present _FineMoE_ , a fine-grained expert offloading system for MoE serving that achieves low inference latency with memory efficiency. We design _FineMoE_ to extract fine-grained expert selection patterns from MoE models and semantic hints from input prompts to efficiently guide expert prefetching, caching, and offloading decisions. _FineMoE_ is prototyped on top of HuggingFace Transformers and deployed on a sixGPU testbed. Experiments with open-source MoE models and real-world workloads show that _FineMoE_ reduces inference latency by 47% and improves expert hit rate by 39% over state-of-the-art solutions. 

## _**CCS Concepts:**_ • **Computing methodologies** → **Distributed algorithms** ; **Artificial intelligence** ; **Machine learning** . 

Despite the computational efficiency, MoE models exhibit substantial memory inefficiency during the serving phase. Though certain model parameters remain inactive during inference, they must still reside in GPU memory to allow for potential future activation. Expert offloading [4, 16, 51, 58] has emerged as a promising strategy to address this issue, which predicts inactive experts and transfers them to CPU memory while retaining only the necessary experts in GPU memory, reducing the overall model memory footprint. 

> *This work was conducted while Xingqi Cui was a remote intern student, advised by Dr. Hao Wang at the IntelliSys Lab, Stevens Institute of Technology. 

This work is licensed under a Creative Commons Attribution 4.0 International License. 

However, existing expert offloading solutions struggle to effectively balance the _latency-memory trade-off_ in MoE serving. These approaches either suffer from high inference latency [4, 51] or incur substantial model memory footprints [16, 58]. The key reason is that existing works track expert patterns 

_EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk_ 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 

https://doi.org/10.1145/3767295.3769319 

176 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang 

**==> picture [488 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
Iteration 1 Iter. 2 Iter. 3 Iter. 4 Iter. 5<br>“Summarize this paper” “This” “This paper” “This paper is” … Latency<br>DeepSpeed<br>MoE LLM MoE LLM MoE LLM MoE LLM Input hidden states<br>Embedding layer Embed Embed Embed Self-attention layer Mixtral-offload<br>Transformer block (TB) 1 TB 1 TB 1 TB 1  MoE layer<br>…  … … … Gate Network ProMoE<br>Transformer block (TB) L TB L TB L TB L Probabilities MoE-Infinity<br>Language modeling head Head Head Head FineMoE<br>Expert 1 2 … J<br>No offload<br>“This” “paper” “is” “about”<br>Prefll Decode … Output hidden states Memory<br>…<br>**----- End of picture text -----**<br>


**(a)** MoE-based LLM serving workflows. 

**(b)** Trade-offs in MoE. 

**Figure 1.** Mixture-of-Experts (MoE) Large Language Model (LLM) serving. 

and manage experts in _coarse granularity_ . They fail to accurately identify and retain only the necessary experts in GPU memory during inference, resulting in frequent and costly on-demand expert loading [51] and performance degradation. 

In this paper, we propose _FineMoE_ , a _fine-grained_ expert offloading system that tames the latency-memory trade-off in MoE serving. To track and analyze MoE models’ expert selection behaviors in fine granularity, we propose a new data structure called _expert map_ , which records the iterationlevel probability distributions output by the gate network. _FineMoE_ uses historical expert maps for comparing expert trajectory similarity to guide offloading.[1] Apart from the expert map, _FineMoE_ is designed to track fine-grained input semantic embeddings from individual request prompts processed by the MoE model. Given the collected semantic-based and trajectory-based information, _FineMoE_ carefully searches the most accurate expert map for guiding expert prefetching, caching, and offloading through inference iterations. In summary, we make the following contributions: 

- We design _FineMoE_ , a _fine-grained_ expert offloading system that achieves low inference latency while reducing model memory footprints. 

- We propose a new data structure, expert map, that tracks fine-grained expert selection behaviors of MoE models. _FineMoE_ leverages input semantic embeddings to augment the expert map search to guide expert offloading. 

- We prototype _FineMoE_ on top of HuggingFace Transformers [55] and deploy it on a six-GPU testbed. Extensive experiments with open-source MoE models and real-world workloads show that _FineMoE_ reduces inference latency by 47% and improves expert hit rate by 39% compared to state-of-the-art solutions. 

> 1In this paper, “trajectory” is defined as the collection of probability distributions over experts observed through layers. 

## **2 Background and Motivation** 

## **2.1 LLM Serving** 

Unlike traditional Deep Learning (DL) model inference, Large Language Model (LLM) serving consists of two consecutive stages: _prefill_ and _decode_ . Figure 1a illustrates the two stages when an LLM performs inference for a request prompt. In the prefill stage, the LLM first computes the intermediate key-value (KV) states of the prompt tokens, prefills the KV cache [3, 28, 31, 37, 65], and then generates the first answer token. In the decode stage, the LLM sequentially generates the answer to the prompt token-by-token in an auto-regressive manner, where tokens generated previously are used for generating the next token. 

The two stages have their own unique characteristics. The prefill stage only requires one _iteration_[2] , processing all tokens in parallel and generating the first answer token. The decode stage spans several iterations, generating one token per iteration until the answer is completed. Due to the different characteristics of the two stages, recent studies [43, 65] have identified that the prefill stage is compute-bounded, while the decode stage is considered memory-bounded. Therefore, people typically quantify the serving performance of LLM two stages using different metrics. For the prefill stage, Time-ToFirst-Token (TTFT) is commonly employed, which measures the latency from receiving the user request until generating the first answer token. For the decode stage, Tokens-PerSecond (TPS) or Time-Per-Output-Token (TPOT) is used to measure the generation rate of LLM serving. 

## **2.2 MoE-based LLM Serving** 

By integrating MoE layers in Transformer blocks [54], MoE architectures [61] have emerged as a popular backbone for modern LLMs, such as Mixtral [23], Snowflake Arctic [50], and DeepSeek-MoE [11]. Figure 1a illustrates MoE-based LLMs’ typical structures, where feed-forward network (FFN) 

> 2An iteration refers to a single step in auto-regressive inference that generates one new token. The iteration time denotes the end-to-end latency of this step. 

177 

_FineMoE_ : Taming Latency-Memory Trade-Off in MoE-Based LLM Serving 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

|**MoE**|**Serving**|
|---|---|
|**Lossy**|**Lossless**|
|**Serving**|**Serving**|
|**Reroute**<br>**Compress**|**Coarse-grained**<br>**Fine-grained**|
|(He et al. [20],<br>Lynx [18])<br>(Samoyeds [56],<br>Hobbit [53])|(Mixtral-Ofoad [16],<br>MoE-Infnity [58])<br>(_FineMoE_)|



**Table 1.** Characteristics of three MoE models. 

|**MoE Models**|**Parameters**<br>**(active / total)**|**Experts Per Layer**<br>**(active / total)**|**Num. of**<br>**Layers**|
|---|---|---|---|
|Mixtral-8×7B [23]|12.9B / 46.7B|2 / 8|32|
|Qwen1.5-MoE [60]|2.7B / 14.3B|4 / 60|24|
|Phi-3.5-MoE [1]|6.6B / 42B|2 / 16|32|



**Figure 2.** The design space of MoE-based LLM serving. 

modules are replaced by MoE layers.[3] Each MoE layer consists of a gate network and a set of expert networks. Inside each Transformer block, the self-attention module first calculates the attentions [54] based on input hidden states, and then the gate network determines which expert(s) to activate for computing the output representations. Compared to traditional dense LLMs, MoE-based LLMs only activate a subset of parameters during training and inference, reducing computational overhead while delivering superior generation performance compared to dense LLMs with a comparable number of parameters [1, 11, 23, 50, 57, 60]. 

Despite the benefits of saving training computations, MoEbased LLM serving still suffers from GPU memory inefficiency as MoE inference requires loading all model parameters into GPU memory, including those inactive experts. Table 1 characterizes three popular MoE models: Mixtral8×7B [23], Qwen1.5-MoE [60], and Phi-3.5-MoE [1]. During inference, they exhibit 72%, 81%, and 84% inactive parameters, respectively, due to the sparsity of expert activation in MoE. This corresponds to 67, 23, and 70 GB of inactive GPU memory, resulting in low memory efficiency and serving throughput. Therefore, to efficiently serve large MoE models, we must seek a solution to the memory inefficiency inherited from MoE architecture. 

## **2.3 Latency-Memory Trade-Off** 

Recently, a few studies have been proposed to improve MoEbased LLM serving efficiency. Figure 2 describes the design space in MoE serving. Existing major studies can be categorized into two types: **Lossy serving** applies compression [44], pruning [30], and quantization [27] techniques to the original MoE models to reduce the serving memory requirements. However, this line of work achieves serving efficiency by sacrificing the generation quality. **Lossless serving** focuses on _offloading_ model weights (parameters [4, 41] or experts [16, 51, 58]) that are sparsely utilized in temporal or spatial patterns from GPU memory to CPU memory, aiming to preserve reasonable inference latency. Specifically, expert offloading seeks to predict the activation of experts in advance, prefetching or caching only the necessary experts 

> 3For simplicity, we only show the process of one single request prompt. 

in GPU memory during inference. We opt for lossless serving to design _FineMoE_ because this line of methods avoids modifying models, hence assuring generation quality. 

However, existing offloading solutions cannot achieve an optimal spot in the latency-memory trade-off when serving MoE-based LLMs. Figure 1b compares the performance ( _i.e._ , inference latency and memory footprint) of existing stateof-the-art (SOTA) offloading solutions, which either provide low inference latency but suffer from large memory footprint ( _e.g._ , No-offload and MoE-Infinity [58]), or vice versa ( _e.g._ , ProMoE [51], Mixtral-Offloading [16], and DeepSpeedInference [4]). 

The key reason behind this dilemma is that MoE-based decoder-only LLMs have balanced expert routing [51], leaving existing solutions hard to find effective patterns for guiding expert offloading. Existing research has identified two main reasons for this dilemma: First, most MoE-based LLMs are decoder-only architectures, which exhibit uniform expert activation patterns and low expert access skewness compared to encoder-decoder MoE LLMs [18, 51]. Second, recent MoEbased LLMs employ a load-balancing loss [1, 11, 23, 50, 57], which encourages the gate network to distribute tokens more uniformly across experts within each MoE layer, making expert usage more balanced during training. This balanced routing diminishes the predictability of expert patterns, thus making existing solutions ineffective. 

## **2.4 Existing MoE Offloading Solutions** 

Existing expert offloading approaches [16, 58] rely on **coarsegrained** expert patterns, which are inefficient for guiding offloading. We define coarse-grained information as the expert patterns collected at the request level, where information is aggregated over multiple iterations of a request prompt. For example, MoE-Infinity [58] tracks request-level expert activations. Fine-grained information is defined as the expert patterns observed separately during each inference iteration. Figure 3a shows examples of coarse-grained and finegrained expert activation heatmaps for Mixtral-8×7B [23]. The heatmap records the expert activations across 32 MoE layers, where each layer contains eight experts and activates two experts out of eight to compute representations. While fine-grained (iteration-level) heatmaps show clear expert activation patterns, the aggregated coarse-grained (request-level) heatmap diminishes predictability. 

178 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang 

**==> picture [495 x 78] intentionally omitted <==**

**----- Start of picture text -----**<br>
Coarse-grained expert heatmap<br>Coarse-grained Fine-grained Mixtral-8×7B Qwen1.5-MoE Phi-3.5-MoE<br>4 4<br>3<br>…  2 2<br>1 LMSYS-Chat-1M ShareGPT<br>Layers 0 Mixtral Qwen Phi Mixtral Qwen Phi 0 50 0 50<br>Fine-grained expert heatmap LMSYS-Chat-1M ShareGPT Inference iterations Inference iterations<br>Experts aggregate Mean entropy Mean entropy<br>**----- End of picture text -----**<br>


**(a)** Coarse-grained _vs._ fine-grained expert **(b)** Mean entropy per layer of three MoE mod- **(c)** Mean entropy per layer of three MoE modheatmaps for Mixtral-8×7B with LMSYSels and two datasets for coarse-grained and fineels and two datasets when aggregating expert Chat-1M. Heavier colors indicate more exgrained expert patterns. Higher entropy indipatterns through inference iterations, which dipert activations. cates lower predictability. minishes predictability. 

**Figure 3.** Expert pattern and predictability analysis in coarse granularity (request-level) and fine granularity (iteration-level). 

To demonstrate this point, we analyze the Shannon entropy [48] of expert activations per MoE layer for three popular MoE models. Entropy is an essential metric to quantify the uncertainty and unpredictability of variables in information theory. A balanced expert activation pattern ( _e.g._ , probability distribution [0 _._ 25 _,_ 0 _._ 25 _,_ 0 _._ 25 _,_ 0 _._ 25] of four experts) results in a high entropy, which indicates the pattern is less predictable and harder to select experts. Figure 3b presents the mean entropy computed per layer for three MoE models (Mixtral-8×7B [23], Qwen1.5-MoE [60], and Phi-3.5MoE [1]) across two realistic datasets LMSYS-Chat-1M [64] and ShareGPT [49]. Coarse-grained expert patterns have significantly higher entropy than fine-grained patterns, meaning that expert patterns in coarse granularity can be less effective for predictions. Figure 3c shows the mean entropy per layer when aggregating expert patterns across inference iterations, where expert selection becomes increasingly unpredictable as generation progresses. Qwen1.5-MoE reaches a higher entropy plateau due to its larger expert selection space (60 experts × 24 layers). Similarly, Phi-3.5-MoE (16 × 32) exhibits higher entropy than Mixtral-8×7B (8 × 32). After about ten iterations, expert patterns become blurred and the entropy plateaus, indicating that further iterations contribute only marginal additional unpredictability. While entropy is low at the beginning of inference, it gradually increases with iterations as more expert activation information is aggregated, thereby becoming more unpredictable. 

In contrast to coarse-grained expert offloading solutions, we argue that expert offloading should be carefully guided by **fine-grained** designs: analyzing iteration-level patterns, understanding models’ expert selection preferences, and leveraging semantic characteristics of request prompts. 

## **2.5 Problems of Coarse-Grained Offloading** 

Existing coarse-grained expert offloading solutions exhibit three problems: 

**1) Insufficient latency-memory trade-off.** Existing solutions prefetch and offload experts in coarse granularity, either heavily focusing on reducing inference latency but incurring 

**==> picture [203 x 109] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0<br>Mixtral-8×7B<br>0.5<br>1.0 Qwen1.5-MoE<br>0.5<br>1.0 Coarse-grained<br>Fine-grained Phi-3.5-MoE<br>0.5<br>0 5 10 15 20 25 30<br>Prefetch distance<br>Expert hit rate<br>**----- End of picture text -----**<br>


**Figure 4.** Expert hit rates of coarse-grained and finegrained expert offloading designs when serving Mixtral8×7B, Qwen1.5-MoE, and Phi-3.5-MoE with LMSYS-Chat1M at different prefetch distances, respectively. 

large memory footprint [58] or reducing memory footprint but severely increasing inference latency [4, 16]. 

**2) Low expert hit rates.** Existing solutions employ coarsegrained expert pattern tracking methods ( _e.g._ , Expert Activation Matrix in MoE-Infinity [58]), which produce ineffective expert patterns for guiding offloading decisions, leading to low expert hit rates and high inference latency. 

**3) Ignorance of MoE models’ and prompts’ heterogeneity.** 

Existing solutions largely ignore the unique characteristics of different MoE models and input prompts and serve them in a one-fits-all manner [4, 16, 51, 58], which omits opportunities for fine-grained optimizations adaptive to heterogeneous models and prompts in MoE serving. 

Figure 4 shows the expert hit rates of serving three popular MoE-based LLMs, Mixtral-8×7B [23], Qwen1.5-MoE [60], and Phi-3.5-MoE [1] using LMSYS-Chat-1M dataset [64] with coarse-grained and fine-grained expert offloading designs at different prefetch distances, respectively. Prefetch distance refers to the number of layers ahead that a prefetch instruction is issued before the target layer activates its experts. By leveraging fine-grained expert offloading, we can achieve 

179 

_FineMoE_ : Taming Latency-Memory Trade-Off in MoE-Based LLM Serving 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

significantly higher expert hit rates over coarse-grained methods and preserve better performance by adapting to varying prefetch distances. 

## **3** _**FineMoE**_ **’s Overview** 

## **3.1 Objectives and Challenges** 

_FineMoE_ is designed to achieve the following three goals: **Memory-efficient MoE serving with minimal inference latency.** We have demonstrated that existing expert offloading solutions [16, 51, 58] fail to tame the latency-memory trade-off in MoE serving (§2.3). We aim to achieve both low memory footprint and inference latency by proposing finegrained expert offloading. 

**Minimize expert miss due to mispredictions in expert prefetching.** Expert prefetching, involving future expert activation predictions, is an essential step in expert offloading solutions. However, a recent study [51] has shown that _expert miss_ due to mispredictions can cause high on-demand expert loading delay in inference. We should minimize expert miss and mitigate mispredictions in expert offloading. 

**Adapt to heterogeneous MoE models and prompts.** MoE inference can serve heterogeneous models [11, 23, 50, 57, 60] with varying prompts [49, 64] in real-world scenarios. While existing solutions handle different models and prompts with a one-fits-all design, we should design our expert offloading to adapt to the heterogeneity in MoE serving. 

We must address three critical challenges to realize the above objectives: 

**How to maximize expert hit rate when prefetching and offloading experts?** Expert hit rate directly relates to the inference latency. With more experts being hit, fewer experts need to be loaded on demand. We propose a fine-grained expert offloading solution to achieve a high expert hit rate. 

**How to adapt to different MoE models and prompts?** Heterogeneous MoE models and input prompts exhibit unique system and semantic characteristics. We should craft our solution with fine-grained optimizations to enable adaptivity. 

**How to avoid additional system overheads when managing experts?** Our design must not introduce additional system overheads when serving existing MoE LLMs. We apply a series of system optimizations in _FineMoE_ to ensure serving efficiency and minimize additional overheads. 

## **3.2 Architecture and Workflow** 

Figure 5 describes the architecture and workflow of _FineMoE_ , which consists of three main components: 

**Expert Map Store.** We record _expert maps_ , a new data structure defined in _FineMoE_ , to track _fine-grained_ expert activation patterns from historical request prompts. expert maps provide nuance expert selection preferences over existing coarse-grained expert tracking methods ( _e.g._ , Expert Activation Matrix in MoE-Infinity [58]). The Expert Map 

**==> picture [218 x 178] intentionally omitted <==**

**----- Start of picture text -----**<br>
Request prompt “Summarize this paper”<br>FineMoE Contexts 1<br>Inference<br>Expert Map  Expert Map Searcher<br>Store Prefill<br>Semantic<br>(         ,      )<br>(         ,      ) Trajectory Decode<br>(         ,      ) (         ,      ) “This”<br>2<br>“…”<br>5 Guide<br>Expert Cache 3<br>Prefetch 4<br>Offload …<br>CPU Mem. GPU Mem.<br>Generated answers “This paper is about…”<br>Search similarity<br>**----- End of picture text -----**<br>


**Figure 5.** _FineMoE_ ’s architecture and workflow. 

Store dynamically keeps the most useful and unique expert maps for real-time inferences. 

**Expert Map Searcher.** When a request prompt arrives, _FineMoE_ searches the Expert Map Store for appropriate expert maps to guide expert prefetching before inference. expert map search is guided by calculating similarity scores in two folds: _semantic_ and _trajectory_ similarity. 

**Expert Cache.** After receiving the searched expert maps, _FineMoE_ prefetches experts from CPU memory to GPU to perform computations in inference. _FineMoE_ evicts and offloads low-priority expert weights to CPU memory if exceeding Expert Cache capacity. 

_FineMoE_ follows the five steps below to enable memoryefficient MoE serving with minimal inference latency: 

**Step 1 : Inference context collection.** Before every inference iteration, _FineMoE_ collects necessary _contexts_ , such as semantic embeddings and previous expert activation trajectories (§4.1), and feeds them to the Expert Map Searcher for hybrid similarity searching. 

**Step 2 : Expert map similarity searching.** After receiving iteration-level contexts, the Expert Map Searcher identifies the most similar expert maps by comparing the input context data with historical context data in the Expert Map Store (§4.2). The retrieved expert maps are forwarded to the Expert Cache to guide expert prefetching and offloading decisions. **Step 3 : Guided expert prefetching and offloading.** We dynamically compute expert selection thresholds to determine which expert(s) to prefetch and offload in the MoE model guided by the searched expert maps (§4.3). Then, _FineMoE_ prefetches the expert weights from CPU to GPU memory and offloads cached experts from GPU to CPU when reaching the cache limit (§4.5). 

**Step 4 : Expert serving.** The whole inference process consists of one iteration in the Prefill stage and multiple iterations in the Decode stage. For each MoE layer in every iteration, 

180 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang 

_FineMoE_ directly serves the expert required by the gating network if the corresponding weights are available in the GPU memory (defined as an expert hit). Otherwise, _FineMoE_ on-demand loads the expert weights from CPU to GPU to perform lossless serving (defined as an expert miss). **Step 5 : Expert map update.** _FineMoE_ observes new expert maps produced after each iteration and updates them in the Expert Map Store (§4.4). When reaching the store capacity ( _e.g._ , 1K expert maps), _FineMoE_ deduplicates the Expert Map Store by identifying and dropping redundant expert maps to maintain diversity, maximizing the possibility of providing effective expert maps for any request prompts. 

**==> picture [217 x 98] intentionally omitted <==**

**----- Start of picture text -----**<br>
“This paper” Input hidden states Expert ID<br>1 2 … J<br>TB 1  MoE layer<br>Gate Network 0.1 0.2 … 0.5 1<br>TB 2<br>… Prob. Dist.   Pl [(] [i] [)] 0.4 0.2 … 0.1 0.4 0.2 … 0.1 2<br>TB L … … … … …<br>“is” Expert 1 2 … J 0.2 0.3 … 0.2 L<br>Iter. 3 Output hidden states Expert Map mapi<br>MoE LLM<br>Layer ID<br>**----- End of picture text -----**<br>


**Figure 6.** Expert selections tracked by an expert map. 

optimization problem: 

## **3.3 Problem Formulation** 

We consider serving an MoE-based LLM with _𝐿_ MoE layers on a GPU cluster, where each MoE layer has one gating network and _𝐽_ experts. The gating network of each layer selects top _𝐾_ ∈[1 _, 𝐽_ ] experts for computation. The MoE model processes and generates answers for a workload consisting of _𝑊_ unique request prompts. Let [ _𝑊_ ] := {1 _, . . . ,𝑤, . . . ,𝑊_ } denote the set of all requests, [ _𝐿_ ] := {1 _, . . . ,𝑙, . . . , 𝐿_ } denote the set of all layers in a MoE model, and [ _𝐽_ ] := {1 _, . . . , 𝑗, . . . , 𝐽_ } denote the set of all experts in a layer, respectively. Each request prompt _𝑤_ ∈[ _𝑊_ ] consists of multiple iterations processed during the prefill and decode stages. Let _𝐸_[(] _[𝑖]_[)] _[𝑗]_[-th] _𝑙,𝑗_[denote the] expert at the _𝑙_ -th layer in the _𝑖_ -th iteration, where _𝑙_ ∈[ _𝐿_ ], _𝑗_ ∈[ _𝐽_ ], and _𝑖_ ∈[ _𝑤_ ]. During each iteration _𝑖_ , we can make at most _𝐿_ · _𝐽_ prefetching decisions. Let _𝐸[𝑖] cache_[and] _[ 𝐸][𝑖] activate_[denote] the set of cached experts and the set of activated experts for Iteration _𝑖_ , respectively. Hence, we represent the result of whether an expert _𝐸𝑙,𝑗_[(] _[𝑖]_[)][∈] _[𝐸] activate_[(] _[𝑖]_[)][is missed by] _[ 𝐸] cache_[(] _[𝑖]_[)][:] 

**==> picture [194 x 31] intentionally omitted <==**

where _𝑅_[(] _[𝑖]_[)] _𝑙,𝑗_[=][ 1][ means] _[ 𝐸] 𝑙,𝑗_[(] _[𝑖]_[)][is a miss and requires on-demand] loading from CPU memory. Since all experts in an MoE model are typically designed to have the same weight size, we assume experts’ loading time _𝑇𝑒_ and memory footprint _𝑀𝑒_ are homogenous.[4] Therefore, the total on-demand loading latency _𝑇_ is summed across all iterations for each expert during the inference process: 

**==> picture [143 x 25] intentionally omitted <==**

Finally, employing the above definitions, we formulate the MoE expert offloading as an integer linear programming (ILP) 

> 4We only consider selective experts. Some MoE models, such as Qwen1.5MoE-A2.7B, have a few always-on experts that are not offloadable. 

**==> picture [215 x 45] intentionally omitted <==**

**==> picture [195 x 15] intentionally omitted <==**

**==> picture [195 x 15] intentionally omitted <==**

The objective is to minimize the on-demand loading latency (ideally _𝑇_ = 0 with perfect predictions) while limiting the total memory footprint of cached experts to satisfy the available GPU memory _𝑀_ . Constraint 1 denotes the total number of prefetched experts should not exceed the total number of all experts in the MoE model. Constraint 2 represents the total number of activated experts, which must be the same as the total number of top _𝐾_ experts summed across all _𝐿_ layers. Constraint 3 describes the total memory footprint of prefetched experts must be limited by the available GPU memory size. Note that solving the ILP problem is already NP-hard [10], while in reality, prefetching experts always have mispredictions that further complicate the problem. Therefore, we opt for a heuristic-based design for _FineMoE_ . 

## **4** _**FineMoE**_ **’s Design** 

## **4.1 Expert Maps** 

We propose a new data structure, _Expert Map_ , to track expert activation patterns with a fine granularity. Figure 6 depicts the structure of an expert map. During the _𝑖_ -th iteration, the _𝑙_ -th self-attention layer first calculates the attention states. The gate network receives attentions and computes a probability distribution P _𝑙_[(] _[𝑖]_[)] ∈ R _[𝐽]_ over all the experts at Layer _𝑙_ : 

**==> picture [236 x 25] intentionally omitted <==**

Then, top _𝐾_ ∈[1 _, 𝐽_ ] experts are selected from _𝑃𝑙_[(] _[𝑖]_[)] to compute representations for Layer _𝑙_ . We collect the probability distributions _𝑃𝑙_[(] _[𝑖]_[)] across all _𝐿_ layers to form the expert map of Iteration _𝑖_ : 

**==> picture [173 x 15] intentionally omitted <==**

181 

_FineMoE_ : Taming Latency-Memory Trade-Off in MoE-Based LLM Serving 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

**==> picture [229 x 195] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) a3 Semantic search<br>a2 score [sem] x,y Expert Map Store 1 … J Layers  2  [1 , d ]<br>0.9 Iter. 1 (         ,      ) 0.1 … 0.5 1 #J<br>Semantic  0.2 Iter. 2 (         ,      ) … … … … …<br>embeddings … 0.4 … 0.1 d #1<br>0.6 Iter. i (         ,      ) … … … … a4<br>a1 d<br>Expert ID b3 Trajectory<br>b1 …0.1 …… 0.5… Layer 1 b2 score 0.2 [traj] x,y Expert Map Store Iter. 1 (         ,      ) 1 search … J<br>… … … …<br>0.1 … 0.2 Layer l-d 0.4 Iter. 2… (         ,      ) 0.1 … 0.2 l<br>… … …<br>0.8 Iter. i (         ,      ) … … … …<br>0.3 … 0.1 Layer L-d<br>(b) Previous trajectory Layer  l 2  [ d  + 1 , L ] # 2 b4 Prefetch<br>Prefetch<br>… … …<br>“This” Embed Layer 1 Layer d Layer d+1 Layer l Layer L LM Head “paper”<br>…<br>…<br>**----- End of picture text -----**<br>


**Figure 7.** Workflow of _FineMoE_ ’s expert map search. 

By tracking expert maps, we guide _FineMoE_ to discover fine-grained expert patterns—the iteration-level expert selection preferences via probability distributions. Intuitively, analyzing probability distributions enables _FineMoE_ to not only identify which experts are binarily selected or omitted, but also to assess the confidence or preference assigned to each expert from the perspective of the gate networks. 

The design of expert maps has two key advantages over existing coarse-grained expert tracking methods ( _e.g._ , MoEInfinity [58] tracks the request-level expert hit counts). _First_ , existing works only focus on _aggregated_ request-level expert activations, whereas an expert map tracks individual iterations with detailed expert selections. _Second_ , existing works only record the expert hit counts, whereas we track detailed probability distributions. Note that expert maps can easily recover coarse-grained information by applying a top _𝐾_ selection operator to the probability distributions and aggregating expert counts over iterations, therefore generalizing to existing tracking methods. 

## **4.2 Expert Map Search** 

Given the historical expert maps defined in §4.1, _FineMoE_ searches expert maps that provide the most accurate expert activation predictions with two fine-grained metrics: semantic similarity (§4.2.1) and trajectory similarity (§4.2.2). We also show that they are both effective in searching accurate historical expert maps for prediction and offloading (§4.2.3). 

Existing solutions [16, 51, 58] _cannot_ observe previous expert patterns for prediction and prefetching before the target layer is ready to activate experts for the initial layers _𝑙_ ∈[1 _,𝑑_ ], where _𝑙_ represents the current layer index in an iteration and _𝑑_ is referred as the _prefetch distance_ . When predicting and prefetching experts for MoE models, _prefetch distance_ is used 

to avoid impacting inference latency [51, 62]. Prefetch distance is the number of layers ahead that a prefetch instruction is issued before the target layer activates its experts, similar to the same term in memory prefetching [29]. An ideal prefetch distance should perfectly overlap the prediction and prefetching operation overheads with the inference process. 

Therefore, existing approaches [16, 51, 58] typically employ coarse-grained rules to prefetch experts for initial layers _𝑙_ ∈[1 _,𝑑_ ]. For example, MoE-Infinity [58] prefetches the most popular experts across all historical data points. Even for layers _𝑙_ ∈[ _𝑑_ + 1 _, 𝐿_ ], existing approaches use coarse-grained (request-level) metrics for predicting and prefetching experts, leading to low offloading accuracy. 

In contrast, _FineMoE_ leverages fine-grained iteration-level metrics tailored to the prefetch distance _𝑑_ , employing semantic embeddings for layers prior to the prefetch distance and expert trajectories for layers subsequent to it. Figure 7 shows that _FineMoE_ employs two fine-grained search approaches to jointly search expert maps for guiding expert prefetching: _Semantic-based expert map search_ compares the input embeddings with historical embeddings to find expert maps with similar inputs, whereas _trajectory-based search_ observes previous expert trajectories ( _i.e._ , probability distributions) and searches for similar expert maps. We combine both semantic and trajectory features to improve _FineMoE_ ’s map-searching and expert offloading accuracy. 

**4.2.1 Semantic-based Expert Map Search.** Recent studies [25] demonstrate that semantic embeddings, _i.e._ , embedding layer’s output after processing raw tokens, can potentially indicate expert selection behaviors. When serving request prompts and recording their expert maps, we record the _semantic embeddings_ for each inference iteration. Existing MoE-based LLMs all contain an embedding layer for token semantic encoding, where words or subwords that appear in similar contexts will have similar embeddings [38]. It’s natural to extract the semantic embeddings using the output from the model’s original embedding layer. Figure 7(a) shows the semantic-based expert map search in four steps: a1) extract semantic embeddings from the embedding layer, a2) compute similarity scores using semantic embeddings with historical data points in the Expert Map Store, a3) search similar expert maps based on similarity scores, and a4) prefetch experts with high probabilities for layers _𝑙_ ∈[1 _,𝑑_ ]. 

For any input prompts, we compute pairwise cosine similarity _score[sem]_ ∈ R _[𝐵]_[×] _[𝐶]_ between the semantic embedding _sem[new]_ ∈ R _[𝐵]_[×] _[ℎ]_ and the collection of historical semantic embeddings _sem[old]_ ∈ R _[𝐶]_[×] _[ℎ]_ in the Expert Map Store: 

**==> picture [233 x 30] intentionally omitted <==**

where _𝐵_ is the batch size of input prompts, _𝐶_ is the Expert Map Store capacity, and _ℎ_ is the hidden dimension size. Then, for prompt _𝑥_ , the historical Iteration _𝑦_ with the highest score 

182 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang 

**==> picture [206 x 77] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0<br>Mixtral<br>Qwen<br>Phi<br>0.5<br>0<br>0 0.5 1.0 0 0.5 1.0<br>Semantic similarity score Trajectory similarity score<br>Mean expert hit rate<br>**----- End of picture text -----**<br>


**Figure 8.** Mean expert hit rates of different semantic and trajectory similarity scores with LMSYS-Chat-1M. 

is selected. We use partial expert maps from the selected iteration, {P1[(] _[𝑦]_[)] _, . . . ,_ P _𝑑_[(] _[𝑦]_[)] } ∈ _map[old] 𝑦_[, to guide layers] _[ 𝑙]_[∈[][1] _[,𝑑]_[]][.] 

**4.2.2 Trajectory-based Expert Map Search.** We leverage expert probability trajectories of previous ( _𝑙_ − _𝑑_ ) layers to search expert maps for layers _𝑙_ ∈[ _𝑑_ + 1 _, 𝐿_ ]. Specifically, when _𝑙_ = _𝑑_ + 1, we use the past expert trajectories from Layer 1 for prediction; when _𝑙_ = _𝑑_ + 2, we use the past trajectories from Layers 1 and 2; and so on. When _𝑙_ = _𝐿_ (last layer), we use the past trajectories from Layers 1 to _𝐿_ − _𝑑_ for prediction. Figure 7(b) shows the trajectory-based expert search for a layer _𝑙_ ∈[ _𝑑_ +1 _, 𝐿_ ] in four steps: b1) collect previous trajectory {P1 _, . . . ,_ P _𝑙_ − _𝑑_ } from Layers 1 to _𝑙_ − _𝑑_ , b2) compute similarity scores using collected trajectories with historical data points in the Expert Map Store, b3) search similar expert maps based on similarity scores, and b4) prefetch experts with high probabilities for the layer _𝑙_ ∈[ _𝑑_ +1 _, 𝐿_ ]. We repeat this process until the last layer (Layer _𝐿_ ) is completed. 

Similar to the semantic-based search, we compute pairwise cosine similarity _score[traj]_ ∈ R _[𝐵]_[×] _[𝐶]_ between the observed trajectories, _map[new]_ ∈ R _[𝐵]_[×(] _[𝑙]_[−] _[𝑑]_[)] _[𝐽]_ , and the collection of historical expert maps, _map[old]_ ∈ R _[𝐶]_[×(] _[𝑙]_[−] _[𝑑]_[)] _[𝐽]_ , in the Expert Map Store: 

**==> picture [234 x 30] intentionally omitted <==**

We select the historical iteration with the highest score. Then, we use P _𝑙_[(] _[𝑦]_[)] ∈ _map[old] 𝑦_ from the selected expert map to guide the expert prefetching for the target layer _𝑙_ ∈[ _𝑑_ + 1 _, 𝐿_ ]. 

By combining the two expert map search methods, we carefully customize the map that guides expert prefetching for every inference iteration in MoE serving. With this design, expert map search introduces negligible overhead to the endto-end inference latency, which we demonstrate in §6.8. 

**4.2.3 Effectiveness of Semantic and Trajectory Similar-** 

**ity.** To verify how semantic and trajectory similarity scores can guide expert offloading, we run three MoE models (Mixtral8×7B, Qwen1.5-MoE, and Phi-3.5-MoE) with two datasets (LMSYS-Chat-1M and ShareGPT). For each model and dataset, we first run prompts and record their semantic embeddings and expert trajectories, where each prompt generates one data point consisting of a semantic embedding and an expert map. Then, we exhaust all pairwise cases by calculating their 

**==> picture [204 x 70] intentionally omitted <==**

**----- Start of picture text -----**<br>
Semantic similarity Trajectory similarity<br>1<br>0<br>Mixtral Qwen Phi Mixtral Qwen Phi<br>LMSYS-Chat-1M ShareGPT<br>0.96 0.97 0.97 0.95 0.92 0.84 0.94 0.96 0.96 0.93 0.90 0.85<br>Pearson coefficient<br>**----- End of picture text -----**<br>


**Figure 9.** Pearson correlation coefficients between semantic and trajectory similarity scores and expert hit rates. 

semantic and trajectory similarity and expert hit rate ( _i.e._ , overlapped expert ratio). Figure 8 shows the mean expert hit rates of different semantic and trajectory similarity scores for three MoE models with LMSYS-Chat-1M. Both semantic and trajectory similarity can effectively indicate the accuracy of historical prompts or expert maps for offloading. 

To _statistically_ quantify the correlations between similarity score and expert hit rate, we calculate the Pearson correlation coefficients [9] using all paired semantic and trajectory similarity scores and corresponding expert hit rates in Figure 8. The Pearson coefficient is commonly used to measure correlations between variables, where a coefficient close to 1 indicates a strong positive correlation and a coefficient close to 0 means a weak correlation. Figure 9 shows the Pearson coefficients between similarity score and expert hit rate with three MoE models and two datasets. The results show that high similarity scores potentially relate to high expert hit rates. 

## **4.3 Expert Prefetching** 

Given the searched and customized expert map P _𝑙_[(] _[𝑖]_[)] for a layer _𝑙_ ∈[ _𝐿_ ] in Iteration _𝑖_ , we explain how it guides _FineMoE_ to dynamically prefetch experts in fine granularity. 

**Similarity-aware expert selection.** With the different contexts collected during iterations, expert maps searched by _FineMoE_ also have varying similarity scores.[5] Figures 8 and 9 demonstrated that similarity scores can effectively indicate the search confidence, where high searched similarity scores potentially mean high expert hit rates. Hence, we design _FineMoE_ ’s expert prefetching to be similarity-aware. For a layer _𝑙_ ∈[ _𝐿_ ] with a _score_ ∈[−1 _,_ 1] to prefetch, we first dynamically compute an expert selection threshold _𝛿𝑙_ ∈[0 _,_ 1]: 

**==> picture [227 x 10] intentionally omitted <==**

where _score_ is the cosine similarity score computed in Equations 4 and 5. Given searched P _𝑙_ , we find the set of experts to prefetch _𝐸prefetch_ by iteratively picking the expert with the highest probability from P _𝑙_ = { _𝑝𝑙,_ 1 _, . . . , 𝑝𝑙,𝑗, . . . , 𝑝𝑙,𝐽_ } until the 

> 5In the following paper, we use “similarity scores” in both search contexts for simplicity, _i.e._ , semantic similarity in semantic-based expert map search and trajectory similarity in trajectory-based search, respectively. 

183 

_FineMoE_ : Taming Latency-Memory Trade-Off in MoE-Based LLM Serving 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

summed probability of _𝐸prefetch_ exceeds _𝛿𝑙_ : 

**==> picture [206 x 46] intentionally omitted <==**

**==> picture [187 x 11] intentionally omitted <==**

where _𝐾_ is the number of experts needed to activate per layer ( _e.g._ , Mixtral-8×7B activates two experts per layer). Constraint 7 requires the total probability of selected experts to prefetch per layer to be greater than _𝛿𝑙_ . Constraint 8 represents the minimum number of selected experts must be larger than the number of experts to activate required by the MoE model. Intuitively, we assign a higher _𝛿_ to low-score expert maps so that more experts are prefetched to mitigate mispredictions and assign a lower _𝛿_ for high-score expert maps to reduce the memory footprint. Experts with higher probabilities are prioritized to be prefetched. 

**Asynchronous expert map searching and prefetching.** Existing studies [16, 58] predict and prefetch experts synchronously during inference, severely hindering the inference performance. For example, MoE-Infinity [58] cannot compute forward functions before finishing expert prediction and prefetching at every MoE layer [59]. To minimize the system overhead and inference latency, we decouple the map searching and expert prefetching from the inference process using an asynchronous Publisher-Subscriber architecture (Figure 7). The Expert Map Store is a message broker that keeps messages from both the inference process and the Expert Map Searcher. As the inference proceeds, _FineMoE_ ’s inference process continuously publishes and writes the inference contexts ( _i.e._ , semantic embeddings and expert probability distributions) to the Expert Map Store. At the same time, the Expert Map Searcher subscribes to the context data, searches expert maps based on new context data, and prefetches experts to the Expert Cache in an asynchronous manner. 

## **4.4 Expert Map Store Management** 

Practically, we design _FineMoE_ ’s Expert Map Store to maintain a capacity _𝐶_ for storing unique expert maps. To effectively guide inference across diverse prompts, it makes sense to identify and deduplicate redundant expert maps. 

**Expert map deduplication.** Since _FineMoE_ uses two approaches ( _i.e._ , semantic-based and trajectory-based) to compute similarity, we unify the two similarity scores to compute the pairwise redundancy scores between new iteration data and historical iteration data: 

**==> picture [240 x 22] intentionally omitted <==**

where _score[sem][score][traj]_[semantic-] _𝑥,𝑦_[∈][R] _[𝐵]_[×] _[𝐶]_[and] _𝑥,𝑦_[∈][R] _[𝐵]_[×] _[𝐶]_[are] based and trajectory-based pairwise similarity scores calculated from Equations 4 and 5, _𝑑_ is the prefetch distance, _𝐿_ is 

the total number of layers, _𝐵_ is the batch size of new interaction data, and _𝐶_ is the Expert Map Store capacity. Intuitively, as shown in Figure 7, the semantic-based and trajectory-based similarity scores contribute to the search expert map in proportion to _[𝑑] 𝐿_[and] _[𝐿]_[−] _𝐿[𝑑]_[, respectively. Therefore, we follow the] same ratio to unify and compute the redundancy score. Whenever new iterations’ context data arrive at the Expert Map Store, we compute the pairwise redundancy score _RDY𝑥,𝑦_ to determine which old iterations to drop. Hence, we update the old iterations _𝑦_ (columns in _RDY𝑥,𝑦_ ) with new iterations _𝑥_ (corresponding rows in _RDY𝑥,𝑦_ ) in the Expert Map Store. 

**Theoretical analysis.** The expert map deduplication can be formulated as a Minimum Sphere Covering problem [17]. Each expert map is a vectorized patch, and the full sphere represents all possible expert selections. The objective is to cover as much of the sphere as possible using a small number of maps, keeping storage overhead low. Studies [15, 46] have proved that maintaining at least 2 _𝐿𝐽_ expert maps guarantees a lower bound of 75% expert map similarity ( _i.e._ , we can find an expert map that is at least 75% similar to any new iterations), and keeping[1] 2 _[𝐿𝐽]_[ln][(] _[𝐿𝐽]_[)][expert maps provides a lower bound] of 98% similarity, where _𝐿_ and _𝐽_ are the numbers of layers and experts per layer in the MoE model, respectively. Given that modern MoE-based LLMs generally have _𝐿_ ∈[8 _,_ 128] and _𝐽_ ∈[24 _,_ 96], we can approximate the Expert Map Store’s maximal requirement to be less than 50K expert maps with 200 MB CPU memory [58]. 

## **4.5 Expert Caching and Eviction** 

Similar to existing expert offloading solutions [16, 51, 58], we design _FineMoE_ to maintain an Expert Cache on GPUs to reuse expert weights when serving different request prompts. Given searched expert maps from §4.2, we guide _FineMoE_ ’s Expert Cache to compute two priority scores for individual experts: 1) a _prefetching priority_ to decide the orders to prefetch experts in the searched maps, and 2) an _eviction priority_ to determine the orders to evict experts in the Expert Cache. 

**Expert prefetching priority.** Recall the set of experts to prefetch _𝐸prefetch_ is determined in Equation 6. For each expert _𝐸𝑙,𝑗_ ∈ _𝐸prefetch_ , we define the prefetching priority to be 

**==> picture [164 x 22] intentionally omitted <==**

where _𝑝𝑙,𝑗_ is the expert probability from the searched expert map, and _𝑙now_ is the current layer that the inference process stays at. Intuitively, experts with a higher probability _𝑝𝑙,𝑗_ to be activated should be prefetched sooner, and experts that sit closer to the current layer ( _i.e._ , smaller _𝑙_ − _𝑙now_ ) should also be prioritized. 

**Expert eviction priority.** Similar to MoE-Infinity [58], _FineMoE_ ’s expert caching is based on the least frequently used (LFU) caching algorithm. We integrate the searched map to jointly determine the eviction priority. For each expert 

184 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang 

_𝐸𝑙,𝑗_ ∈ _𝐸cache_ , we define the eviction priority to be 

**==> picture [168 x 24] intentionally omitted <==**

where _freq𝑙,𝑗_ is the cache visit frequency and _𝑝𝑙,𝑗_ is the probability from the searched map for an expert _𝐸𝑙,𝑗_ ∈ _𝐸cache_ . Intuitively, when reaching the Expert Cache limit, we want to first evict experts who are less frequently hit and have lower probabilities of being activated. Note that similar to existing works [51, 58], we do not consider the recent usage of experts as opposed to the classic least recently used (LRU) algorithm [16]. Since the expert usage is layer-wise sequential, _i.e._ , one layer following another, prioritizing recently used experts is against the nature of sequential forward computation. 

**On-demand expert loading.** Mispredictions of expert prefetching lead to expert miss in the Expert Cache, as the MoE model cannot find available experts designated by the gate networks. Whenever an expert miss occurs, _FineMoE_ pauses all expert prefetching tasks and immediately loads missed experts from CPU to GPU memory for fast serving. 

## **5** _**FineMoE**_ **’s Implementation** 

We prototype _FineMoE_ on top of Huggingface Transformers framework [55] using MoE-Infinity codebase [59]. The implementation of _FineMoE_ is described as follows. 

**Expert Map Store** is implemented in Python using PyTorch [42] and NumPy [19] libraries. We store all semantic embeddings and expert maps using ndarrays data structure for efficient array operations. The arrays are converted to tensors to compute similarity for expert map searching. 

**Expert Map Searcher** is implemented in Python using PyTorch [42]. We implement the pairwise computations, including similarity (§4.2) and redundancy (§ 4.4) scores, using PyTorch native operations. 

**Expert Cache** is implemented in C++ based on MoEInfinity codebase [59]. The expert management in GPUs is implemented with the CUDA Runtime APIs [40]. We implement prefetching and caching logic of _FineMoE_ in the MoE-Infinity codebase to enable expert offloading. Same with MoE-Infinity, _FineMoE_ supports multi-GPU inference with expert parallelism (EP), where the experts are mapped to different GPU devices for loading and offloading. We use a hash map to assign expert IDs to different GPUs and retrieve them during inference. The expert assignment follows a round-robin manner to balance the overall GPU load. Additionally, we use a task pool in the GPU space with asynchronous threads to schedule and execute expert prefetching and on-demand loading tasks. 

## **6 Evaluation** 

## **6.1 Experimental Setup** 

We introduce our evaluation methodology in this section. 

**Testbed.** We conduct all experiments on a six-GPU testbed, where each GPU is an NVIDIA GeForce RTX 3090 with 24 GB GPU memory. All GPUs are interconnected using pairwise NVLinks and connected to the CPU memory using PCIe 4.0 with 32GB/s bandwidth. Additionally, the testbed has an AMD Ryzen Threadripper PRO 3955WX CPU with 32 cores and 480 GB CPU memory. 

**Models.** We employ three popular MoE-based LLMs in our evaluation: Mixtral-8×7B [23], Qwen1.5-MoE [60], and Phi-3.5-MoE [1]. Table 1 describes the parameters, number of MoE layers, and number of experts per layer for the three models. Following the evaluation of existing works [51], we profile the models to set the optimal prefetch distance _𝑑_ to three before evaluation. 

**Datasets and traces.** We employ two real-world prompt datasets commonly used for LLM evaluation: LMSYS-Chat1M [64] and ShareGPT [49]. For most experiments, we split the sampled datasets in a standard 7:3 ratio, where 70% of the prompts’ context data ( _i.e._ , semantic embeddings and expert maps) are stored in _FineMoE_ ’s Expert Map Store, and 30% of the prompts are used for testing. For online serving experiments, we empty the Expert Map Store and use real-world LLM inference traces [43, 52] released by Microsoft Azure to set input and generation lengths and drive invocations. 

**Baselines.** We compare _FineMoE_ against four SOTA MoE serving baselines: 1) **MoE-Infinity** [58] uses coarse-grained request-level expert activation patterns and synchronous expert prediction and prefetching for MoE serving. We prepare the expert activation matrix collection for MoE-Infinity before evaluation for a fair comparison. 2) **ProMoE** [51] employs a stride-based speculative expert prefetching approach for MoE serving. Since the codebase of ProMoE is not open-sourced and requires training predictors for each MoE model, we reproduced a prototype of ProMoE on top of MoE-Infinity in our best effort. 3) **Mixtral-Offloading** [16] combines a layer-wise speculative expert prefetching and a LRU-based expert cache. 4) **DeepSpeend-Inference** [4] employs an expert-agnostic layer-wise parameter offloading approach, which uses pure on-demand loading and does not support prefetching. We implement the offloading logic of DeepSpeed-Inference in the MoE-Infinity codebase and add an expert cache for a fair comparison. We enable all baselines to serve MoE models from HuggingFace Transformer [55]. 

**Metrics.** Following the standard evaluation methodology of existing works [3, 51, 58, 65] on LLM serving, we report the performance of the prefill and decode stages separately. We measure Time-to-First-Token (TTFT) for the prefill stage and Time-Per-Output-Token (TPOT) for the decode stage. Additionally, we also report other system metrics, such as expert hit rate and overheads, for detailed evaluation. 

## **6.2 Offline Serving Performance** 

We first evaluate the offline serving performance of prefill and decode stages when running _FineMoE_ and other baselines 

185 

_FineMoE_ : Taming Latency-Memory Trade-Off in MoE-Based LLM Serving 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

**==> picture [203 x 152] intentionally omitted <==**

**----- Start of picture text -----**<br>
10<br>0<br>FineMoE ProMoE DeepSpeed<br>MoE-Infinity Mixtral-Offload<br>2<br>0<br>1.0<br>0.5<br>0<br>Mixtral Qwen Phi Mixtral Qwen Phi<br>LMSYS-Chat-1M ShareGPT<br>TTFT (s)<br>TPOT (s)<br>Expert hit rate<br>**----- End of picture text -----**<br>


**Figure 10.** Overall performance of prefill and decode stages. 

**==> picture [210 x 79] intentionally omitted <==**

**----- Start of picture text -----**<br>
DeepSpeed ProMoE FineMoE<br>Mixtral-Offload MoE-Infinity<br>1.0<br>0.5 Mixtral- Qwen1.5- Phi-3.5<br>8×7B MoE MoE<br>0<br>150 200 40 60 80 60 80 100 120<br>End-to-end request latency (s)<br>CDF<br>**----- End of picture text -----**<br>


**Figure 11.** CDF of request latency for MoE online serving. 

with the three MoE models, where we report Time-To-FirstToken (TTFT) and Time-Per-Output-Token (TPOT). Similar to existing works [3, 65], we measure TTFT and TPOT for individual prompts for each combination of model and dataset. For evaluation with LMSYS-Chat-1M and ShareGPT datasets, the input lengths are set to 37 and 43 tokens, and generation lengths to 127 and 122 tokens, which are the mean values calculated across datasets, respectively. For each dataset, we randomly sample 64 prompts and report average results. 

Figure 10 shows the TTFT, TPOT, and expert hit rate of _FineMoE_ and four baselines when serving three MoE models with LMSYS-Chat-1M and ShareGPT datasets, respectively. DeepSpeed-Inference has both the worst TTFT and TPOT due to expert-agnostic offloading and lacking expert prefetching. While Mixtral-Offloading, ProMoE, and MoEInfinity perform better than DeepSpeed-Inference, they are underperformed by _FineMoE_ because of coarse-grained offloading designs. Compared to DeepSpeed-Inference, MixtralOffloading, ProMoE, and MoE-Infinity, _FineMoE_ reduces the average TTFT by 74%, 67%, 56%, and 53%, and reduces the average TPOT by 46%, 38%, 27%, and 22%, respectively. 

For expert hit rate, DeepSpeed-Inference has no expert misses because it fetches whole layers with full experts, but with the worst latency due to pure on-demand loading. Mixtral-Offloading achieves a higher hit rate than ProMoE and MoE-Infinity because of its synchronous speculative 

**==> picture [193 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
6 Mixtral-8x7B<br>4<br>2<br>3 Qwen1.5-MoE<br>2<br>1 FineMoE MoE-Infinity<br>ProMoE Mixtral-Offload<br>DeepSpeed<br>4 Phi-3.5-MoE<br>3<br>2<br>1<br>6 12 24 48 96<br>Expert cache limit (GB)<br>TPOT (s)<br>**----- End of picture text -----**<br>


**Figure 12.** Performance under varying expert cache limits. 

prefetching with a prefetch distance of 1. However, due to synchronous prefetching, its TTFT and TPOT are worse than others except DeepSpeed-Inference. Overall, _FineMoE_ improves the average expert hit rate by 14%, 37%, and 68% over Mixtral-Offloading, ProMoE, and MoE-Infinity, respectively. 

## **6.3 Online Serving Performance** 

Except for the offline evaluation ( _i.e._ , Expert Map Store in full capacity before serving), we also evaluate _FineMoE_ against other baselines in online serving settings. We empty the Expert Map Store of _FineMoE_ and the expert activation matrix collection of MoE-Infinity for the online serving experiment. The request traces are derived from Azure LLM inference traces [43, 52], with randomly sampled 256 requests (2.91 requests per second), to drive LMSYS-Chat-1M prompts for each MoE model serving. To ensure consistency, _FineMoE_ and all baselines input and generate the exact number of tokens specified in the traces. Figure 11 illustrates the CDF of end-to-end request latency across three MoE models. The results demonstrate that _FineMoE_ significantly reduces overall request latency compared to other baselines in online serving. 

## **6.4 Impact of Expert Cache Limits** 

We measure the TPOT of _FineMoE_ and other baselines by limiting the expert cache memory budget to investigate their performance in the latency-memory trade-off (§2.3). We mainly focus on TPOT to show the end-to-end performance impacted by varying cache limits. Figure 12 shows the TPOT of _FineMoE_ and four baselines when serving three MoE models under different expert cache limits. We gradually increase the GPU memory allocated for caching experts from 6 GB to 96 GB while employing the same experimental setting in §6.2. Similarly, DeepSpeed-Inference has the worst TPOT due to being expert-agnostic. _FineMoE_ consistently outperforms Mixtral-Offloading, ProMoE, and MoE-Infinity under varying expert cache limits. As the cache limit increases, the performance gap between all baselines narrows due to the 

186 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang 

**==> picture [193 x 148] intentionally omitted <==**

**----- Start of picture text -----**<br>
6<br>4<br>2<br>0<br>1.0 DeepSpeed MoE-Infinity<br>Mixtral-Offload FineMoE<br>ProMoE<br>0.5<br>0<br>1.0<br>0.5<br>0<br>Mixtral Qwen Phi<br>TTFT (s)<br>TPOT (s)<br>Expert hit rate<br>**----- End of picture text -----**<br>


**Figure 13.** Performance on high-end GPU testbed. 

**==> picture [240 x 86] intentionally omitted <==**

**----- Start of picture text -----**<br>
Speculate Map (T) Map (T+S+δ) LRU FineMoE<br>Hit count Map (T+S) LFU<br>1.0 1.0<br>0.5 0.5<br>0 0<br>(a)  Expert pattern tracking approaches. (b)  Prefetch and caching.<br>Mixtral Qwen Phi MixtralQwen Phi<br>Expert hit rate Expert hit rate<br>**----- End of picture text -----**<br>


**Figure 14.** Ablation study of _FineMoE_ . 

increased availability of cached experts. Nevertheless, for limited GPU memory sizes ( _e.g._ , 6GB), _FineMoE_ reduces the TPOT by 36%, 25%, 16%, and 29%, compared to DeepSpeedInference, Mixtral-Offloading, ProMoE, and MoE-Infinity, across three MoE models, respectively. With fine-grained expert offloading, _FineMoE_ significantly reduces the expert on-demand loading latency while maintaining a lower GPU memory footprint, therefore achieving a better spot in the latency-memory trade-off of MoE serving. 

## **6.5 Impact of GPU Performance** 

To evaluate the impact of GPU performance on offloading methods, we repeat the experiments using LMSYS-Chat-1M on an NVIDIA A100 testbed equipped with 80 GB of HBM2e memory and a peak memory bandwidth of 2 TB/s. Figure 13 presents the serving performance of _FineMoE_ and the baselines across the three MoE models. _FineMoE_ achieves smaller performance gains on the A100 than on the 6×3090 testbed, since high-end GPUs and the lack of EP yield faster inference and lower offloading overhead. Nevertheless, _FineMoE_ consistently outperforms all baselines. The expert hit rate remains largely unaffected, as GPU performance has less impact on expert predictions. 

**==> picture [204 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
10<br>5<br>0<br>4 Mixtral-8×7B Phi-3.5-MoE<br>Qwen1.5-MoE<br>2<br>0<br>1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8<br>Prefetch distance<br>TTFT (s)<br>TPOT (s)<br>**----- End of picture text -----**<br>


**Figure 15.** Performance with different prefetch distances. 

**==> picture [225 x 90] intentionally omitted <==**

**----- Start of picture text -----**<br>
Mixtral-8×7B Qwen1.5-MoE Mixtral-Off. MoE-Inf.<br>Phi-3.5-MoE ProMoE FineMoE<br>1.0<br>10<br>0.8<br>0<br>1.0 (a) 5 (b)<br>0.8<br>0<br>500 1000 1500 1 2 4 8<br>Expert Map Store capacity Inference batch size<br>TTFT (s)<br>   sem. scores<br>TPOT (s)<br>Mean traj. /<br>**----- End of picture text -----**<br>


**Figure 16.** Sensitivity analysis of _FineMoE_ . 

preferences in fine granularity. We evaluate the effectiveness of the expert map against five expert pattern-tracking approaches as follows. 1) **Speculate** : speculative prediction used by Mixtral-Offloading [16] and ProMoE [51], 2) **Hit count** : request-level expert hit count used by MoE-Infinity [58], 3) **Map (T)** : expert map with only trajectory similarity search, 4) **Map (T+S)** : expert map with both trajectory and semantic similarity search but statically select top-K experts to prefetch, and 5) **Map (T+S+** _𝛿_ **)** : expert map with full features enabled, including trajectory and semantic similarity search (§4.2) and dynamically selecting experts to prefetch (§4.3). We implement the above methods in _FineMoE_ ’s Expert Map Searcher for a fair comparison. Figure 14a shows the expert hit rate of the above expert pattern tracking methods. Speculative prediction is effective due to the widespread presence of residual connections in Transformer blocks. However, its effectiveness decreases drastically as prefetch distance increases [51]. The request-level expert activation count has the worst performance due to coarse granularity. As features are incrementally restored to _FineMoE_ ’s expert map, the expert hit rate gradually increases, demonstrating its effectiveness. 

**Effectiveness of expert prefetching and caching.** We evaluate _FineMoE_ ’s expert prefetching and caching against two caching algorithms: 1) **LRU** used by Mixtral-Offloading [16] and 2) **LFU** used by MoE-Infinity [58]. Figure 14b depicts the expert hit rate of _FineMoE_ and two baselines. The results show that LRU performs poorly in expert offloading scenarios. Though LFU achieves a higher hit rate than LRU, _FineMoE_ surpasses both, achieving the highest expert hit rate. 

## **6.6 Ablation Study** 

We present the ablation study of _FineMoE_ ’s design. 

**Effectiveness of expert map search.** One of _FineMoE_ ’s key designs is the expert map, which tracks expert selection 

## **6.7 Sensitivity Analysis** 

We analyze the sensitivity of prefetch distance of MoE models, Expert Map Store capacity, and inference batch size. 

187 

_FineMoE_ : Taming Latency-Memory Trade-Off in MoE-Based LLM Serving 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

**==> picture [215 x 85] intentionally omitted <==**

**----- Start of picture text -----**<br>
Collect context Map match Expert load<br>Inference Expert prefetch Map update<br>Mixtral-8×7B<br>Qwen1.5-MoE<br>Phi-3.5-MoE<br>0 500 1000<br>Latency (ms)<br>**----- End of picture text -----**<br>


**Figure 17.** Latency breakdown of _FineMoE_ ’s one iteration. 

**Prefetch distance of MoE models.** Figure 15 shows the TTFT and TPOT of _FineMoE_ when serving three MoE models with different prefetch distances. We have demonstrated that the expert hit rate decreases when gradually increasing the prefetch distance (Figure 4). When the prefetch distance is small, _FineMoE_ cannot perfectly hide its system delay from the inference process, such as the map searching and expert prefetching, leading to an increase in inference latency. With larger prefetch distances, _FineMoE_ has worse expert hit rates that also degrade performance. Therefore, we set the prefetch distance _𝑑_ to 3, 6, and 4 for Mixtral-8×7B, Qwen1.5-MoE, and Phi-3.5-MoE, respectively. 

**Capacity of Expert Map Store.** We measure the mean semantic and trajectory similarity scores searched in _FineMoE_ ’s expert map searching for MoE model serving. Figure 16(a) presents the mean semantic and trajectory similarity scores of _FineMoE_ with different Expert Map Store capacity sizes. Both semantic and trajectory similarity scores improve as the store capacity increases. While the similarity scores exhibit a significant increase with capacities below 1K, further capacity expansion yields diminishing similarity gains. To minimize _FineMoE_ ’s memory overhead, we set _FineMoE_ ’s Expert Map Store capacity to 1K in evaluation. 

**Inference batch size.** We investigate the impact of inference batch size on _FineMoE_ and three baselines using Mixtral-8×7B with LMSYS-Chat-1M. Figure 16(b) presents the performance of _FineMoE_ , Mixtral-Offloading, ProMoE, and MoE-Infinity as the batch size increases from one to eight. _FineMoE_ achieves the lowest TTFT and TPOT in most cases. 

## **6.8 System Overheads** 

We measure and report the system overheads of _FineMoE_ . 

**Latency overheads of** _**FineMoE**_ **’s operations.** Figure 17 shows the latency breakdown of one inference iteration in _FineMoE_ when serving the three MoE models. We report operation overheads of _FineMoE_ , including context collection, map searching, expert on-demand loading, expert prefetching, and map update after the iteration completes. Qwen1.5-MoE has lower end-to-end iteration latency than Mixtral-8×7B and Phi-3.5-MoE because of significantly fewer parameters. Note that expert prefetching, map searching, and map update tasks are executed asynchronously, aside from the inference process. Hence, they do not contribute to the end-to-end iteration latency. Excluding three asynchronous tasks, the total 

**==> picture [204 x 87] intentionally omitted <==**

**----- Start of picture text -----**<br>
Qwen1.5-MoE<br>100 Phi-3.5-MoE<br>Mixtral-8×7B<br>10<br>1K 2K 4K 8K 16K 32K<br>Expert Map Store capacity<br>Memory footprint (MB)<br>**----- End of picture text -----**<br>


**Figure 18.** CPU memory footprint of _FineMoE_ ’s Expert Map Store with different capacity. 

delay incurred by other operations is consistently less than 50ms (1% of the iteration) across three MoE models, which is negligible compared to the inference latency. 

**Memory overheads of** _**FineMoE**_ **’s Expert Map Store.** Figure 18 shows the CPU memory footprint of _FineMoE_ ’s Expert Map Store when varying the store capacity from 1K to 32K maps. The memory needed to store expert maps for Qwen1.5-MoE is more than Mixtral-8×7B and Phi-3.5-MoE because it has more experts per layer over the other two models, which increases the map shape. Even for the largest capacity (32K), the Expert Map Store requires less than 200MB of memory to store the maps, which is trivial since modern GPU servers usually have abundant CPU memory ( _e.g._ , p4d.24xlarge on AWS EC2 [5] has over 1100 GB of CPU memory). In evaluation, _FineMoE_ ’s map store capacity with 1K maps is sufficient for maintaining performance (§6.7), resulting in minimal memory overhead. 

## **7 Discussion** 

In this section, we compare the heuristic-based _FineMoE_ with Neural Network (NN)-based predictors, analyze the impact of model parallelism on _FineMoE_ ’s performance, and discuss how _FineMoE_ can be extended to other MoE architectures. 

**NN-based predictors.** NN-based predictors for expert offloading are impractical due to multiple sources of overhead. First, they often introduce sub-second inference latency, comparable to MoE inference latency itself. Second, they require extensive data collection, hour-long per-layer training, and frequent retraining to adapt to workload shifts. Third, they consume substantial GPU memory, as prior work [51] reports millions of parameters per MoE layer. Moreover, they are incompatible with _FineMoE_ ’s fine-grained design: training on fine-grained data hinders convergence, while storing iteration-level probabilities generates large volumes that further prolong training and limit feasibility. Therefore, we adopt a heuristic-based design rather than NN-based approaches. 

**Impact of EP and tensor parallelism (TP).** Higher EP distributes experts across more devices and enables greater expert replication, which can increase _FineMoE_ ’s offloading opportunities and memory savings. In contrast, higher TP raises the overhead of offloading operations, since dense model components are split across devices and require coordinated offloading and reloading. As noted in prior work [11, 35], 

188 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang 

production MoE systems generally avoid high TP because its communication costs outweigh performance benefits. In large-scale deployments ( _e.g._ , DeepSeek [35]), MoE systems usually use low EP during prefill to maximize throughput, while adopting high EP during decode to enable higher expert redundancy. Though high-EP decode reduces per-GPU expert occupancy, the larger number of expert replicas ( _e.g._ , 2× more than prefill in DeepSeek [35]) creates additional offloading opportunities by allowing experts to be compacted onto fewer devices. 

**Adaptation to other MoE architectures..** _FineMoE_ can be easily integrated with different MoE architectures. For shared experts, we treat them as always-hit during expert prediction. One of our evaluated models, Qwen1.5-MoE, includes shared experts that are used by all tokens. For multigating MoE, we can extend the expert map search by recording each gate’s probability distribution and flattening the outputs into a single vector for efficient similarity computation. This enables unified handling across diverse routing schemes. 

He et al. [20] drops and reroutes tokens from overloaded experts to others to reduce the straggler effect. Samoyeds [56] serves MoE models with sparsity computing. Lynx [18] selects experts based on batch-level expert importance instead of gate network outputs. DAOP [62] performs computations with predicted experts directly and cannot guarantee full generation quality. FLoE [66] compresses experts on-the-fly during inference. However, lossy serving impacts the model quality and is orthogonal to _FineMoE_ . 

**MoE refactorization.** Some works propose to redesign and refactor the current MoE architecture, such as decoupling gate networks from inference process [14, 22] or building activation-efficient MoE models [8, 25]. This line of work requires model training or fine-tuning before serving. Pregated MoE [22] trains pre-gate functions to eliminate the sequential dependencies between expert selection and execution. SiDA [14] proposes a sparsity-inspired data-aware inference system that decouples the expert routing from inference. READ-ME [8] refactors pre-trained dense LLMs into specialized MoE models. MoLE [25] replaces inputs of all MoE layers with embedding tokens to avoid sparse expert activation. In contrast, _FineMoE_ requires zero training to serve MoE models. 

## **8 Related Work** 

In this section, we provide a brief overview of recent studies and related works on MoE serving. 

**Lossless MoE serving.** Recent studies on lossless MoE serving have been widely proposed. DeepSpeed Inference [4] offloads layer-wise parameters without considering expert awareness and does not provide expert prefetching or caching capabilities. Mixtral-Offloading [16] employs LRU expert caching and introduces speculative prediction to enable expert prefetching. MoE-Infinity [58] proposes the request-level expert activation matrix to guide offloading in coarse granularity. SwapMoE [47] maintains a set of critical experts in GPU memory and adjusts them based on workload changes to minimize offloading overhead. ProMoE [51] trains predictors per MoE layer to achieve high speculative prediction accuracy and low inference latency. Lina [32] exports unpopular experts to host memory while focusing on MoE training. Liu et al. [36] partitions and serves MoE models on serverless computing. Fiddler [26] serves MoE inference on CPU and GPU collaboratively. MoEShard [6] shards experts to achieve balanced expert loads. Unlike existing coarse-grained offloading solutions, _FineMoE_ tracks fine-grained expert patterns from both trajectory and semantic aspects and outperforms SOTA baselines. 

**Lossy MoE serving.** Expert pruning [13] reduces memory usage by removing under-utilized experts. Expert compression [21, 44, 53, 56, 66] compresses less-popular experts to reduce models’ memory footprint. Expert load rerouting [18, 20, 62] balances tokens to under-loaded experts instead of following the outputs of gate networks. Specifically, Hobbit [53] uses low precision to serve less-critical experts. 

## **9 Conclusion** 

This paper proposes _FineMoE_ , a fine-grained expert offloading system for MoE serving that achieves low inference latency without incurring significant model memory footprints. _FineMoE_ tracks iteration-level expert probability distributions from the MoE model using expert map and analyzes input semantic embeddings from individual request prompts. Based on the input semantic and expert trajectory information, _FineMoE_ searches the most accurate expert map to carefully guide the expert prefetching, caching, and offloading decisions tailored to every inference iteration. _FineMoE_ is prototyped on top of HugginFace Transformers and deployed to a six-GPU testbed. Extensive experiments with open-source MoE models and real-world workloads show that _FineMoE_ reduces inference latency by 47% and improves expert hit rate by 39% compared to state-of-the-art solutions. 

## **Acknowledgments** 

We thank anonymous reviewers and our shepherd, Dr. Yaniv David, for their valuable feedback. The work of Hanfei Yu and Hao Wang was supported in part by NSF 2527416, 2534241, and 2523997, and the AWS Cloud Credit for Research program. The work of Hao Wang (Rutgers CS) was supported in part by Amazon Faculty Research Award, Microsoft AI & Society Fellowship, NSF CAREER Award IIS-2340125, NIH grant R01CA297832, and NSF grant IIS-2127918. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the funding agencies. 

189 

_FineMoE_ : Taming Latency-Memory Trade-Off in MoE-Based LLM Serving 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

## **References** 

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. 2024. Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. _arXiv preprint arXiv:2404.14219_ (2024). 

- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. GPT-4 Technical Report. _arXiv preprint arXiv:2303.08774_ (2023). 

- [3] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. In _18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)_ . 

- [4] Reza Yazdani Aminabadi, Samyam Rajbhandari, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Olatunji Ruwase, Shaden Smith, Minjia Zhang, Jeff Rasley, et al. 2022. DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale. In _SC22: International Conference for High Performance Computing, Networking, Storage and Analysis_ . 

- [5] AWS. 2006. AWS EC2: Secure and Resizable Compute Capacity in the Cloud. https://aws.amazon.com/ec2/. 

- [6] Oana Balmau, Anne-Marie Kermarrec, Rafael Pires, André Loureiro Espírito Santo, Martijn de Vos, and Milos Vujasinovic. 2025. Accelerating MoE Model Inference with Expert Sharding. In _Proceedings of the 5th Workshop on Machine Learning and Systems (EuroMLSys)_ . 

- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language Models are Few-Shot Learners. _Advances in neural information processing systems_ (2020). 

- [8] Ruisi Cai, Yeonju Ro, Geon-Woo Kim, Peihao Wang, Babak Ehteshami Bejnordi, Aditya Akella, and Zhangyang Wang. 2024. Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)_ . 

- [9] Israel Cohen, Yiteng Huang, Jingdong Chen, Jacob Benesty, Jacob Benesty, Jingdong Chen, Yiteng Huang, and Israel Cohen. 2009. Pearson Correlation Coefficient. _Noise Reduction in Speech Processing_ (2009). 

- [10] Thomas H Cormen, Charles E Leiserson, Ronald L Rivest, and Clifford Stein. 2022. _Introduction to Algorithms_ . MIT press. 

- [11] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. 2024. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-ofExperts Language Models. _arXiv preprint arXiv:2401.06066_ (2024). 

- [12] Sunhao Dai, Yuqi Zhou, Liang Pang, Weihao Liu, Xiaolin Hu, Yong Liu, Xiao Zhang, Gang Wang, and Jun Xu. 2024. Neural Retrievers are Biased Towards LLM-Generated Content. In _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ . 

- [13] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen. 2021. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. In _International Conference on Learning Representations (ICLR)_ . 

- [14] Zhixu Du, Shiyu Li, Yuhao Wu, Xiangyu Jiang, Jingwei Sun, Qilin Zheng, Yongkai Wu, Ang Li, Hai Li, and Yiran Chen. 2024. SiDA: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models. _Proceedings of Machine Learning and Systems (MLSys)_ (2024). 

- [15] Ilya Dumer. 2007. Covering Spheres with Spheres. _Discrete & Computational Geometry_ (2007). 

- [16] Artyom Eliseev and Denis Mazur. 2023. Fast Inference of Mixtureof-Experts Language Models with Offloading. _arXiv preprint arXiv:2312.17238_ (2023). 

- [17] D Jack Elzinga and Donald W Hearn. 1972. The Minimum Covering Sphere Problem. _Management Science_ (1972). 

- [18] Vima Gupta, Kartik Sinha, Ada Gavrilovska, and Anand Padmanabha Iyer. 2024. Lynx: Enabling Efficient MoE Inference through Dynamic Batch-Aware Expert Selection. _arXiv preprint arXiv:2411.08982_ (2024). 

- [19] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. 2020. Array Programming with NumPy. _Nature_ (2020). 

- [20] Shwai He, Weilin Cai, Jiayi Huang, and Ang Li. 2025. Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts. _arXiv preprint arXiv:2503.05066_ (2025). 

- [21] Shwai He, Daize Dong, Liang Ding, and Ang Li. 2024. Towards Efficient Mixture of Experts: A Holistic Study of Compression Techniques. _arXiv preprint arXiv:2406.02500_ (2024). 

- [22] Ranggi Hwang, Jianyu Wei, Shijie Cao, Changho Hwang, Xiaohu Tang, Ting Cao, and Mao Yang. 2024. Pre-gated MoE: An AlgorithmSystem Co-Design for Fast and Scalable Mixture-of-Expert Inference. In _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . 

- [23] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of Experts. _arXiv preprint arXiv:2401.04088_ (2024). 

- [24] Zhihan Jiang, Jinyang Liu, Zhuangbin Chen, Yichen Li, Junjie Huang, Yintong Huo, Pinjia He, Jiazhen Gu, and Michael R Lyu. 2024. LILAC: Log Parsing using LLMs with Adaptive Parsing Cache. _Proceedings of the ACM on Software Engineering_ (2024). 

- [25] Shibo Jie, Yehui Tang, Kai Han, Yitong Li, Duyu Tang, Zhi-Hong Deng, and Yunhe Wang. 2025. Mixture of Lookup Experts. In _International Conference on Machine Learning (ICML)_ . 

- [26] Keisuke Kamahori, Tian Tang, Yile Gu, Kan Zhu, and Baris Kasikci. 2025. Fiddler: CPU-GPU Orchestration for Fast Inference of Mixtureof-Experts Models. In _International Conference on Learning Representations (ICLR)_ . 

- [27] Young Jin Kim, Raffy Fahim, and Hany Hassan Awadalla. 2023. Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness. _arXiv preprint arXiv:2310.02410_ (2023). 

- [28] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In _Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)_ . 

- [29] Jaekyu Lee, Hyesoon Kim, and Richard Vuduc. 2012. When Prefetching Works, When It Doesn’t, and Why. _ACM Transactions on Architecture and Code Optimization (TACO)_ (2012). 

- [30] Jaeseong Lee, Aurick Qiao, Daniel F Campos, Zhewei Yao, Yuxiong He, et al. 2024. STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning. _arXiv preprint arXiv:2409.06211_ (2024). 

- [31] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. 2024. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. In _18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)_ . 

- [32] Jiamin Li, Yimin Jiang, Yibo Zhu, Cong Wang, and Hong Xu. 2023. Accelerating Distributed MoE training and inference with Lina. In _2023 USENIX Annual Technical Conference (USENIX ATC 23)_ . 

- [33] Yichen Li, Yintong Huo, Renyi Zhong, Zhihan Jiang, Jinyang Liu, Junjie Huang, Jiazhen Gu, Pinjia He, and Michael R Lyu. 2024. Go Static: Contextualized Logging Statement Generation. _Proceedings of_ 

190 

EUROSYS ’26, April 27–30, 2026, Edinburgh, Scotland Uk 

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang 

_the ACM on Software Engineering_ (2024). 

- [34] Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, and Tat-Seng Chua. 2024. Data-efficient Fine-tuning for LLM-based Recommendation. In _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval_ . 

- [35] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. DeepSeek-V3 Technical Report. _arXiv preprint arXiv:2412.19437_ (2024). 

- [36] Mengfan Liu, Wei Wang, and Chuan Wu. 2025. Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing. In _IEEE Conference on Computer Communications (INFOCOM)_ . 

- [37] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, et al. 2024. CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving. In _Proceedings of the ACM SIGCOMM 2024 Conference_ . 

- [38] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient Estimation of Word Representations in Vector Space. (2013). 

- [39] Daye Nam, Andrew Macvean, Vincent Hellendoorn, Bogdan Vasilescu, and Brad Myers. 2024. Using an LLM to Help with Code Understanding. In _Proceedings of the IEEE/ACM 46th International Conference on Software Engineering_ . 

- [40] NVIDIA. 2024. CUDA Runtime API :: CUDA Toolkit Documentation. https://docs.nvidia.com/cuda/cuda-runtime-api/index.html. 

- [41] Ollama. 2024. Get Up and Running with Large Language Models. https://ollama.com/. 

- [42] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. PyTorch: An Imperative Style, HighPerformance Deep Learning Library. _Advances in Neural Information Processing Systems (NIPS)_ (2019). 

- [43] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. In _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . 

- [44] Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, Tianlong Chen. 2024. Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy. In _International Conference on Learning Representations (ICLR)_ . 

- [45] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language Models are Unsupervised Multitask Learners. _OpenAI blog_ (2019). 

- [46] Robert Alexander Rankin. 1947. On the Closest Packing of Spheres in N Dimensions. _Annals of Mathematics_ (1947). 

- [47] Rui Kong, Yuanchun Li, Qingtian Feng, Weijun Wang, Xiaozhou Ye, Ye Ouyang, Linghe Kong, Yunxin Liu. 2023. SwapMoE: Serving Offthe-shelf MoE-based Large Language Models with Tunable Memory Budget. _arXiv preprint arXiv:2308.15030_ (2023). 

- [48] Claude Elwood Shannon. 1948. A Mathematical Theory of Communication. _The Bell System Technical Journal_ (1948). 

- [49] ShareGPT. 2022. ShareGPT: Share Your Wildest ChatGPT Conversations. https://sharegpt.com/. 

   - [53] Peng Tang, Jiacheng Liu, Xiaofeng Hou, Yifei Pu, Jing Wang, PhengAnn Heng, Chao Li, and Minyi Guo. 2024. Hobbit: A Mixed Precision Expert Offloading System for Fast MoE Inference. _arXiv preprint arXiv:2411.01433_ (2024). 

   - [54] A Vaswani. 2017. Attention is all you need. _Advances in Neural Information Processing Systems_ (2017). 

   - [55] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. HuggingFace’s Transformers: State-of-the-Art Natural Language Processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ . 

   - [56] Chenpeng Wu, Qiqi Gu, Heng Shi, Jianguo Yao, and Haibing Guan. 2025. Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores. In _Proceedings of the Twentieth European Conference on Computer Systems (EuroSys)_ . 

   - [57] xAI. 2023. Announcing Grok. https://x.ai/blog/grok. 

   - [58] Leyang Xue, Yao Fu, Zhan Lu, Luo Mai, and Mahesh Marina. 2024. MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache. _arXiv preprint arXiv:2401.14361_ (2024). 

   - [59] Xue, Leyang and Fu, Yao and Lu, Zhan and Mai, Luo and Marina, Mahesh. [n. d.]. MoE-Infinity Codebase. https://github.com/TorchMoE/ MoE-Infinity. 

   - [60] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. 2024. Qwen2 Technical Report. _arXiv preprint arXiv:2407.10671_ (2024). 

   - [61] Seniha Esen Yuksel, Joseph N Wilson, and Paul D Gader. 2012. Twenty Years of Mixture of Experts. _IEEE Transactions on Neural Networks and Learning Systems (TNNLS)_ (2012). 

   - [62] Yujie Zhang, Shivam Aggarwal, and Tulika Mitra. 2025. DAOP: DataAware Offloading and Predictive Pre-Calculation for Efficient MoE Inference. In _Design Automation and Test in Europe (DATE)_ . 

   - [63] Yuyue Zhao, Jiancan Wu, Xiang Wang, Wei Tang, Dingxian Wang, and Maarten de Rijke. 2024. Let Me Do It for You: Towards LLM Empowered Recommendation via Tool Learning. In _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval_ . 

   - [64] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric P Xing, et al. 2023. LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset. _arXiv preprint arXiv:2309.11998_ (2023). 

   - [65] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. In _18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)_ . 

   - [66] Yuxin Zhou, Zheng Li, Jun Zhang, Jue Wang, Yiping Wang, Zhongle Xie, Ke Chen, and Lidan Shou. 2025. FloE: On-the-Fly MoE Inference on Memory-constrained GPU. In _International Conference on Machine Learning (ICML)_ . 

- [50] Snowflake. 2024. Snowflake Arctic: The Best LLM for Enterprise AI. https://www.snowflake.com/en/data-cloud/arctic/. 

- [51] Xiaoniu Song, Zihang Zhong, and Rong Chen. 2024. ProMoE: Fast MoE-based LLM Serving using Proactive Caching. _arXiv preprint arXiv:2410.22134_ (2024). 

- [52] Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri, Josep Torrellas, and Esha Choukse. 2025. DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency. In _International Symposium on High-Performance Computer Architecture (HPCA)_ . 

191 

