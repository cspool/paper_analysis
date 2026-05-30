# Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

Haiyang Huang\*† , Newsha Ardalani\*, Anna Sun\*, Liu Ke\*‡, Hsien-Hsin S. Lee\*, Anjali Sridhar\*, Shruti Bhosale\*, Carole-Jean Wu\*, Benjamin Lee\*\$

<sup>∗</sup>Meta AI †Duke University \$University of Pennsylvania ‡Washington University in St. Louis

*Abstract*—Mixture-of-Experts (MoE) models have gained popularity in achieving state-of-the-art performance in a wide range of tasks in computer vision and natural language processing. They effectively expand the model capacity while incurring a minimal increase in computation cost during training. However, deploying such models for inference is difficult due to their large size and complex communication pattern. In this work, we provide a characterization of two MoE workloads, namely Language Modeling (LM) and Machine Translation (MT) and identify their sources of inefficiencies at deployment.

We propose three optimization techniques to mitigate sources of inefficiencies, namely (1) Dynamic gating, (2) Expert Buffering, and (3) Expert load balancing. We show that dynamic gating improves maximum throughput by 6.21-11.23× for LM, 5.75- 10.98× for MT Encoder and 2.58-5.71× for MT Decoder. It also reduces memory usage by up to 1.36× for LM and up to 1.1× for MT. We further propose Expert Buffering, a new caching mechanism that only keeps hot, active experts in GPU memory while buffering the rest in CPU memory. This reduces static memory allocation by up to 1.47×. We finally propose a load balancing methodology that provides additional scalability to the workload.

# I. INTRODUCTION

The prediction capability of a machine learning model is strongly correlated with the model capacity, (i.e., the number of parameters in the network). In pursuit of accuracy, capacity has grown at an exponential pace of 10 times per year [\[28\]](#page-11-0), accompanied by higher demand for computational resources and extortionate training costs. Sparsely activated neural networks, such as Mixture of Experts (MoE), are attractive model architectures that decouple the requirement for many parameters from the computational costs. In a sparsely activated model, parts of the network are conditionally activated, which reduces training costs. Results from previous works [\[2\]](#page-10-0), [\[7\]](#page-10-1), [\[21\]](#page-11-1), [\[22\]](#page-11-2), [\[30\]](#page-11-3), [\[34\]](#page-11-4) show that MoE models reduce training cost yet improve model prediction performance in tasks such as language modeling [\[2\]](#page-10-0), [\[5\]](#page-10-2), [\[7\]](#page-10-1), [\[27\]](#page-11-5), machine translation [\[22\]](#page-11-2) and image recognition [\[26\]](#page-11-6), [\[33\]](#page-11-7). While training has been relatively well studied, MoE deployment and inference has received much less attention.

Characterizing and optimizing inference is increasingly important as large language models, like ChatGPT, are deployed for production services. Figures [1](#page-0-0) and [2](#page-0-1) highlight model

![](_page_0_Figure_10.jpeg)

<span id="page-0-0"></span>Fig. 1. Comparison of MoE and Dense Language Models on training cost and perplexity (the lower perplexity the better in model quality). MoE models can achieve better performance than their dense counterparts at lower training cost (Source: Artetxe et. al. [\[2\]](#page-10-0)).

![](_page_0_Figure_12.jpeg)

<span id="page-0-1"></span>Fig. 2. Comparison of MoE and Dense models on single node inference latency. While theoretically MoE models should be able to infer on a similar latency as their flop-equivalent dense counterparts, we find that in practice they are 15× slower for Language Modeling (LM), 22× slower for Machine Translation (MT) encoder and 3× slower for Machine Translation decoder.

prediction capabilities as well as associated training and inference costs between the state-of-the-art MoE and dense model architectures. In Figure [1,](#page-0-0) MoE models achieve the same level of performance and quality (*i.e.*, perplexity) with half of the training cost (GPU-days) compared to their dense counterparts. However, when deployed for inference, MoE models are 15× slower for language models (LM) and more than 3× slower for machine translation (MT) compared to their FLOP-equivalent dense counterpart, as shown in Figure [2.](#page-0-1)

A few strategies have been proposed to reduce MoE inference latency. We might distill MoE models into much smaller dense models with a similar number of FLOPs [\[2\]](#page-10-0), [\[7\]](#page-10-1). Although distillation reduces model size and inference latency, it also reduces model quality. Lepikhin *et. al.* show

<sup>†</sup> ‡ Work done while interning at Meta

that a 14.7 billion parameter Switch Transformer model retains only 29% of its perplexity gain on language modeling after distillation [\[21\]](#page-11-1). DeepSpeed-MoE and Tutel [\[16\]](#page-11-8), [\[24\]](#page-11-9) focus on increasing parallelism and optimizing pipelines to increase hardware utilization when deploying MoE models on hundreds of GPUs. These optimizations are scoped narrowly and mitigate inefficiencies in specific kernels for communication collectives and GPU computation. However, these studies lack a comprehensive analysis of inference latency and neglect inefficiencies in the MoE algorithms themselves.

*In this paper, we provide optimization strategies for efficient MoE deployment, reducing inference costs with minimal impact on model quality.* First, we characterize MoE Transformer deployment on three important axes: inference latency, memory usage, and expert activation. Our detailed characterization establishes significant correlations between expert activation patterns and deployment efficiency. Latency and memory usage is high because expert activations are highly sparse and query load is highly imbalanced across experts,

Second, we analyze unique expert activation patterns to propose a new, optimized gating policy—called Dynamic Gating—and implement it on an open-source, state-of-theart MoE-based Transformer [\[23\]](#page-11-10). For Language Modeling (LM) and Machine Translation (MT) across various datasets and subtasks [\[8\]](#page-10-3), [\[22\]](#page-11-2), our system prototype for dynamic gating improves inference throughput by 6.21-11.23× for LM, 5.75-10.98× for MT Encoder and 2.58-5.71× for MT Decoder by enabling larger batch sizes and smaller latencies. Our optimization strategies complement previously proposed optimizations on distillation, communication collectives, and GPU kernels. When integrated with other optimizations, our gating policy could achieve even greater benefits.

Finally, we take a closer look into expert activation patterns, discovering significant imbalance in load distribution across experts but high temporal locality. Based on these two key observations, we propose Expert Buffering, which improves memory efficiency by allocating a fixed, but limited, amount of GPU memory for hot and active experts and relies on CPU memory to buffer all other experts. The less frequently accessed experts are brought into GPU memory as needed, reducing demand for GPU memory significantly. Expert buffering is orthogonal to existing memory management techniques, such as offloading. Our experiments show that expert buffering reduces static memory usage by up to 1.47× on tasks that demonstrate significant expert sparsity. To balance load, we further propose a priori load balancing based on historical expert activation data, and analyze its benefits for throughput.

To summarize, our contributions in this paper are as follows:

- We provide a thorough characterization of MoE deployment, identifying sources of inefficiencies by breaking down inference latency and memory usage across different components of the model architecture.
- We identify the gating function as a major contributing factor to the high latency and large memory footprint of MoE models. We propose a novel gating policy which

- significantly reduces latency and memory consumption while also enabling inference with larger batch sizes and a smaller number of GPUs.
- We analyze expert activation patterns during inference and discover a significant imbalance in load distribution across experts but high temporal locality.
- We propose Expert Buffering, a new caching mechanism that keeps only hot or active experts in GPU memory and buffers the rest in CPU memory. The less frequently accessed experts are brought into GPU memory as needed. This optimization can reduce static memory allocation in GPU by 1.47×.
- We propose techniques to balance load across experts to further improve memory usage and system robustness.

# II. BACKGROUND

# *A. Mixture-of-Experts Module*

Using different models for different inputs has long been discussed as a way to improve model versatility and robustness. Mixture-of-Experts (MoE) module [\[29\]](#page-11-11) is a practical application of this idea for neural networks. An MoE module (Figure [3\)](#page-2-0) consists of multiple independent models (called experts), and a gating function that assigns inputs to each of the experts. Each input only activates its assigned expert network, which theoretically allows the model capacity (*i.e.*, the number of parameters in the model) to expand "outrageously" with minimal computation efficiency loss.

# *B. Transformer Model Architecture*

The Transformer architecture has gained popularity in computer vision and natural language processing by defining the state-of the-art on multiple tasks in these domains [\[4\]](#page-10-4), [\[32\]](#page-11-12). From the top down, a Transformer consists of a tokenizer that parses the input into tokens, and an encoder-decoder architecture consisting of dense transformer layers. The encoder structure has N dense transformer layers, where N varies from single digits to dozens across different model architectures. Each dense transformer layer is composed of two blocks: a multi-head attention (MHA) block, and a Feed-Forward Network (FFN) block connected by a residual connection, as shown in Figure [3.](#page-2-0) The decoder's structure is very similar to the encoder's, except for an optional MHA layer that attends to encoder output.

# *C. MoE Transformer Model Architecture*

The MoE Transformer combines the MoE idea with the Transformer architecture. In addition to the normal dense Transformer layer, it introduces a new kind of layer with sparse MoEs. Sparse MoE layers replace the FFN block with an MoE block that consists of multiple different expert FFNs. Instead of applying a single FFN to all the input tokens, it first uses a gating function to decide which expert(s) is most suitable for each token, and then routes the tokens to their corresponding expert. Typically, a token is routed to one or two experts in a policy that is referred to as top-1 or top-2 gating. Sparse MoE

![](_page_2_Figure_0.jpeg)

<span id="page-2-0"></span>Fig. 3. Visualization of MoE module, dense transformer encoder layer, MoE transformer encoder layer and MoE transformer encoder layer deployed with expert parallelism. MHA stands for Multi-head attention block, whereas FFN stands for Feed-forward Network block. (a) MoE module introduced in [\[29\]](#page-11-11) (b) Dense Transformer Encoder Layer. A typical dense transformer layer consists of Multi-head Attention (MHA) followed by an FFN layer. (c) Naive MoE Transformer layer. The single FFN block in dense transformer is replaced by a set of FFNs, called experts, that operate in parallel. Not all tokens are processed by all experts. The gating function decides which experts will receive which tokens.. (d) MoE Transformer with expert parallelism. Each device only holds a subset of all experts. Tokens assigned to non-local expert FFNs are dispatched to their assigned expert via an all-to-all communication collective. .

layers replace the dense transformer layers intermittently in the multi-layer model architecture.

These modifications grant greater degrees of freedom to the model and effectively expand the model size. Compared to traditional Transformer models, where the FLOP count per batch scales linearly with the number of parameters, MoE networks require much less computation, thus allowing large models to be trained efficiently. MoE Transformers have been successful in reducing the training cost of large transformer models [\[2\]](#page-10-0), [\[5\]](#page-10-2), [\[7\]](#page-10-1), [\[21\]](#page-11-1) and achieving high accuracy in vision, text, speech and multitask learning area [\[11\]](#page-10-5), [\[12\]](#page-11-13), [\[20\]](#page-11-14), [\[27\]](#page-11-5), [\[35\]](#page-11-15).

# *D. Expert Parallelism*

Compared to traditional Transformer models of the same model capacity, MoE models offer an interesting trade-off. MoE requires much less computation but much more memory usage. Expert layers deploy many additional FFNs, which increase model size and associated demands for memory capacity near the compute device (*e.g.*, the GPU). To handle this problem, GShard [\[21\]](#page-11-1) proposes expert parallelism, which distributes the workload across multiple devices to reduce memory and computation per device.

With expert parallelism, MoE layers are distributed across multiple devices. Each device holds only a subset of expert FFNs and a copy of all the other parameters. When a token is assigned to experts that reside on other devices, an all-toall communication collective sends the token to corresponding devices. The tokens are processed by the expert and then sent back by another all-to-all communication.

At maximum expert parallelism, which allocates one expert per device, memory usage and FLOP count per device are comparable to that from a dense transformer model. Since the gating function is a lightweight linear layer, the overall computational complexity of a batch is about the same as that of a dense transformer with much fewer parameters. Nevertheless, the enormous size, the sparse activation of experts, and the complex communication pattern between devices hosting different experts poses severe challenges during model deployment and inference.

# III. CHARACTERIZATION OF THE MOE MODEL

To characterize the workload of MoE Transformer models, we study two major use cases: Language Modeling and Machine Translation. Language modeling generates the probability an input sequence appears in natural text whereas machine translation maps the input from one language to another. Both tasks are core problems to natural language processing, and are currently major applications of MoE Transformers. We choose models in recent publications that achieved state-of-the-art as our testbed. The details of the datasets and models can be found in Table [I.](#page-3-0)

The MoE model's dense counterparts are selected to be FLOP-equivalent, so they share most of the hyperparameters with the MoE Transformers of interest including hidden dimensions, number of layers and attention heads. The only difference is that the MoE Transformer replaces the FFN layer with an MoE layer every MF layers. Capacity factor C, a parameter unique to the MoE Transformer, controls how many tokens can be processed by a single expert. Under the original design, no matter how many tokens are assigned to an expert, the expert will always process a number of tokens equal to C times the sequence length. When too many tokens are assigned to a single expert, excess tokens are dropped and not

| Task     | Type                                                                                                        | Size          | E            | MF           | CF               |
|----------|-------------------------------------------------------------------------------------------------------------|---------------|--------------|--------------|------------------|
| LM       | Dense<br>MoE                                                                                                | 355M<br>52B   | –<br>512     | –<br>2       | –<br>0.05        |
| MT       | Dense<br>MoE                                                                                                | 3.3B<br>54.5B | –<br>128     | –<br>4       | –<br>1           |
| Task     | Type                                                                                                        | Layers        | TD           | HD           | Vocab            |
| LM       | Dense<br>MoE                                                                                                | 24<br>24      | 1024<br>1024 | 4096<br>4096 | 51200<br>51200   |
| MT       | Dense<br>MoE                                                                                                | 48<br>48      | 2048<br>2048 | 8192<br>8192 | 256206<br>256206 |
| Platform | Specification                                                                                               |               |              |              |                  |
| CPU      | 2×Intel Xeon E5-2698 v4 at 2.2GHz<br>with 700GB memory                                                      |               |              |              |                  |
| CPU-GPU  | 16GB/s via PCIe 3.0                                                                                         |               |              |              |                  |
| GPU      | 8×NVIDIA Tesla V100, with 5120 CUDA<br>cores, 32GB HBM2 memory at 900GB/s<br>connected by NVLink at 300GB/s |               |              |              |                  |
|          |                                                                                                             |               |              |              |                  |

#### TABLE I

<span id="page-3-0"></span>EXPERIMENTAL SETUP. LM: LANGUAGE MODELING. MT: MACHINE TRANSLATION. E: NUMBER OF EXPERTS. MF: MOE LAYER FREQUENCY. CF: CAPACITY FACTOR. TD: TOKEN DIMENSION. HD: HIDDEN DIMENSION. VOCAB: VOCABULARY SIZE. E, MF AND CF DO NOT APPLY TO DENSE MODELS.

processed by any expert. When too few tokens are assigned, unused capacity will be filled by zeros. We utilize the capacity factor settings recommended by [\[2\]](#page-10-0), [\[22\]](#page-11-2). Table [I](#page-3-0) details the experimental setup.

# *A. Latency and Memory Consumption*

The most important metrics in machine learning model deployment are execution time (i.e., latency) and memory consumption. A shorter latency ensures a more timely response from the service, whereas lower memory consumption indicates lower resource usage and potential to accommodate larger batch sizes. In this subsection, we compare MoE inference performance along these two axes. The mini batch size is set to 8 for language modeling and 48 for machine translation. We use a dense model of similar FLOPs as the baseline for comparison.

Latency. Figures [2](#page-0-1) shows the latency and memory consumption of the MoE Transformers of interest and that of their dense counterparts. Although in theory, MoE Transformers exeucte a similar number of FLOPs compared to the baseline dense models, in practice they are significantly slower. For the Language Model, the dense model requires 74.2ms, whereas the MoE Transformer requires more than 1.09s. For Machine Translation, the dense model executes the encoder and decoder in 101ms and 32ms, respectively, but the MoE Transformer requires 2.26s and 90ms.

Figure [5](#page-3-1) breaks down latency under different scenarios. The latency gap has been previously attributed to the frequent allto-all communication collective in MoE models. [\[21\]](#page-11-1) While all-to-all collectives does increase latency under multi-node deployment, we note that this is not the only source of latency.

![](_page_3_Figure_8.jpeg)

<span id="page-3-3"></span>Fig. 4. MoE vs Dense model memory footprint comparison during inference. The MoE models require significantly more memory usage when deployed on GPUs. Besides the large memory consumption due to the expanded model capacity (introduced by expert parameters), it also requires more memory for activation. (results for batchsize=48 for MT, and batchsize=8 for LM. Note that these are the largest batch sizes that are feasible to run under the baseline implementation.)

![](_page_3_Figure_10.jpeg)

<span id="page-3-1"></span>Fig. 5. MoE Model latency breakdown. Besides all-to-all communication, other components of the model, such as gating function and expert execution, are also inefficient. Communication overhead increases significantly when more than one node is involved.(Results for batch size=8 for LM and batch size=48 for MT).

In Section [III-B,](#page-3-2) we will discuss these extra sources of latency.

Memory. We also observe a large increase in memory consumption for MoE models (see Figure [4\)](#page-3-3). For LM, the dense model only requires 2.2GB on each GPU whereas the MoE model requires 18.88GB at its peak, an increase of 8.58×. For MT, the dense and MoE models use 7.02GB and 21.16GB, respectively, an increase of 3.01×.

We perform a detailed analysis by separating static and dynamic memory usage. Static memory consumption refers to memory allocated to model parameters, whereas dynamic memory consumption refers to memory allocated on demand, usually by network activations. Due to the fact that each GPU accommodates more than one expert during inference, the increase in static memory is expected. However, we observe that the peak dynamic memory consumption also increases significantly in both cases, which is surprising.

# <span id="page-3-2"></span>*B. Inefficiency of Static Gating*

What is behind the major overhead in latency and memory consumption? A detailed examination of the latency breakdown sheds light on this matter. While the all-to-all communication collective plays a significant role in multinode scenarios, other components, such as the gating function and expert execution, also contribute significant inefficiencies. Furthermore, a close analysis of the memory trace indicates that memory allocation occurs during the gating and reordering phases. The source of inefficiency in these components warrants further investigation.

The root cause of performance and resource overheads lies in the static gating policy. Recent implementations [\[2\]](#page-10-0), [\[18\]](#page-11-16), [\[24\]](#page-11-9), [\[27\]](#page-11-5) of MoE Transformer models usually assume the number of tokens assigned to each expert is roughly the same because the loss function during training accounts for load balance. As a result, the token distribution process is simplified to an all-to-all collective that distributes the same number of tokens (see Figure [8\(](#page-5-0)a)).

The Capacity Factor C defines the number of tokens processed by each expert in one batch. If the gating function assigns fewer tokens than an expert's capacity, the rest of the capacity will be filled by placeholders (*i.e.*, zero vectors). If the gating function assigns more tokens than an expert's capacity, excess tokens are dropped by the expert and their information will be retained only by the residual connection. Token drop is undesirable as it harms accuracy. To avoid information loss and accuracy fluctuations, capacity factor C is usually set at high values during inference. While this safeguards accuracy, it increases latency and memory costs.

Waste Factors. For Language Model, where the number of experts E = 512 and the Capacity Factor C = 0.05, the number of tokens processed by an expert in a sequence S is ECS = 512×0.05×S = 25.6S. The amount of computation the device must actually perform, instead, is only 2S since the model implements top-2 gating and each token would be processed by two experts. Therefore, the waste factor is 25.6S/2S = 12.8.

For Machine Translation, the analysis is similar. The number of tokens processed by each expert is ECS = 128 × 1 × S = 128S. However, the amount of computation the device must actually perform is 2S as well, which leads to a waste factor of 128S/2S = 64. The huge waste factor suggests that typical MoE models perform a large amount of excess computation and communication as well as consume a large amount of extra memory.

Our question is whether this over-provisioned resource usage is avoidable. If the workload is well balanced such that token allocations across experts are comparable, resource waste can be reduced by simply scaling down the Capacity Factor. On the other hand, if the expert activation is sparse, scaling down the capacity factor is not an option because doing so increases the chance of dropping tokens and harming model accuracy.

# IV. ANALYSIS OF EXPERT ACTIVATION PATTERNS

<span id="page-4-0"></span>To understand whether such a huge waste factor is necessary for service stability, we will study expert activation patterns across two applications of MoE models, namely language modeling and machine translation. Moreover, we propose two optimizations (dynamic gating and expert caching) to reduce the waste factor and improve latency and memory consumption.

# *A. Language Modeling Case*

For Language Modeling, we use the PILE dataset [\[8\]](#page-10-3) as the input, which is the validation set used in prior work [\[2\]](#page-10-0). We select three domains (Wikipedia, PubMed and Github) from the PILE dataset to study the effect of different input data on the expert activation patterns across time (*i.e.*, consecutive batches).

We visualize the results in Figure [6.](#page-5-1) Each row represents a batch and each column represents the load of a particular expert. A more intense color indicates the expert receiving a higher portion of all tokens in a batch. As shown in Figure [6\(](#page-5-1)a), load distribution across experts is highly imbalanced. There exists multiple hot experts that always get a large share of tokens (multiple lines of intense color), and the other experts consistently receive a small amount of tokens (lines of lighter colors).

In the most extreme cases, Figure [7](#page-5-2) indicates there exist experts that never get any tokens. Due to the static gating policy, these experts still receive and process empty token placeholders, introducing a huge waste of computational resources. As shown in Figures [6\(](#page-5-1)a) and [7,](#page-5-2) the set of hot experts and their hotness level varies across domains even though all domains consistently exhibit a high-degree of sparse expert activation.

# *B. Machine Translation Case*

For Machine Translation (MT), we use the original validation dataset NLLB-200 [\[22\]](#page-11-2). We use English as the source language, and select three different target languages (French, Japanese and Asturian). Expert activation on MT for randomly selected layers is visualized in Figure [6\(](#page-5-1)b).

Machine Translation models also exhibit load imbalance and a small fraction of experts that are more hot than others, and the load imbalance is even more pronounced. Certain experts on both encoder and decoder has received a large share of all tokens that is almost half of the full batch, whereas many experts maintain a low degree of activation.

We further inspect whether expert sparsity exists on the encoder and the decoder of the model. Figure [7](#page-5-2) demonstrates the expert sparsity level on the encoder and decoder on all three tasks. We find that the encoder activation is mostly dense, that most of the experts are activated at all times. The decoder activation is extremely sparse (about 75%).

We visualize the selected activation pattern of the encoder and decoder in Figure [6\(](#page-5-1)b). The activation is normalized within a batch, and the color intensity is a measure of load intensity, representing the percentage of tokens assigned to each expert within a batch. The detailed activation shows that the expert activation pattern in machine translation is similar across different languages. The encoder architecture captures the source language properties which is the same across all three tasks (English). To our surprise, we found

![](_page_5_Figure_1.jpeg)

<span id="page-5-1"></span>Fig. 6. Visualization of the expert activation pattern on selected layer of (a) language modeling and (b) machine translation. Activation is normalized. The expert activation pattern exhibits strong imbalance on all the tasks, and the imbalance is consistent. Specifically, on machine translation decoder the sparseness is enormous, and the expert also demonstrates strong temporal correlation.

![](_page_5_Figure_3.jpeg)

<span id="page-5-2"></span>Fig. 7. Average number of inactive experts on Language Modeling and Machine Translation. Most, if not all experts are activated throughout the LM and MT encoder. However, activation on MT decoder is extremely sparse, even if we utilize a batch size of 96 under dynamic gating policy.

that expert activation is more or less similar across different target languages as well as decoder architectures.

A closer inspection on the expert activation on the decoder shows that the expert sparsity has a strong temporal locality. The intense color representing high load of expert usually appears as lines, suggesting that an expert is active across consecutive batches. This implies temporal locality for hot experts. This observation is a key motivation for expert caching discussed in Section [VI.](#page-6-0)

# V. DYNAMIC GATING OPTIMIZATION

<span id="page-5-3"></span>The observed activation patterns demonstrate a distinct gap between assumptions in system design and inference performance. Naively increasing expert capacity may still not prevent token overflow for some experts, but will create extra redundancy and waste for other experts. While previous studies also notice the imbalanced activation across experts [\[16\]](#page-11-8), [\[24\]](#page-11-9), existing solutions retain a static gating policy, which increases CF when severe imbalance appears [\[16\]](#page-11-8). Our conclusion is that static gating increases resource waste and fixed expert capacity is not the optimal solution for the distribution of

![](_page_5_Figure_9.jpeg)

<span id="page-5-0"></span>Fig. 8. Comparison between the static gating in [\[2\]](#page-10-0), [\[21\]](#page-11-1) and our implementation of dynamic gating. For simplicity, we assume E=3, S=6, C=0.5 and top-1 gating in this example. Shapes of tensors are recorded in parentheses. (a) Static Gating. Under a static gating policy, each expert always processes a predefined amount of tokens, which may lead to token overflow or empty tokens. See Section [III-B](#page-3-2) for details. (b) Dynamic Gating. Under a dynamic gating policy, each expert only processes the tokens that are assigned to it. The token distribution mechanism is simplified with less complexity, and the communication and computation are reduced.

tokens to experts. The constraints imposed by static capacity should be removed and the gating function should be dynamic.

Nevertheless, changing the gating policy to allow dynamic sizes for experts is non-trivial. Major, existing implementations [\[2\]](#page-10-0), [\[16\]](#page-11-8), [\[26\]](#page-11-6) do not support dynamic. They rely on static capacity to guarantee that message sizes of all-to-all collectives are the same, which simplifies the communication.

# *A. Case for Dynamic Gating*

Figure [8\(](#page-5-0)a) visualizes the static gating policy. In this example, we assume a sequence length (total number of tokens in a batch of sentences) S = 6, number of experts E = 3, capacity factor C = 0.5 such that static capacity is S × C = 3. We assume top-1 gating such that each token is assigned to only one expert.

After the gating function generates the gating decision, the static gating policy translates it into E dispatch masks, each of size (S,(S × C)). The dispatch mask is generated as follows. If token i is assigned to expert e, then the process will check if the e-th mask still has capacity. If so, the i-th column of the first empty row will be marked as 1, whereas the other numbers are kept to be 0. This process gives us a sparse dispatch mask that is a tensor of the size (E, S,(S ×C)), in which at most S entries are 1's due to potential token dropping. Note that this matrix is highly sparse. The input tokens will be multiplied with the dispatch mask to reorder inputs into E sets of inputs, each with S × C tokens. Each set of inputs will be sent to its assigned expert's device.

Figure [8\(](#page-5-0)b) shows how the gating function must be redesigned when the number of tokens transferred between devices is variable. Our implementation simplifies the distribution process. We find an array of indices that can sort the array by performing an argsort. This set of optimal index prepares the order of tokens for dispatch to devices. By counting the number of occurrences of each expert, we know the exact size of the dispatched input to each device.

Because the sizes of dispatched input are variable [\[13\]](#page-11-17), we adopt a two-step approach. First, we use an all-to-all collective to inform each device the size of the incoming tokens. Second, we use another all-to-all collective to perform the real token transfer. The size all-to-all collective is launched as soon as sizes are known, maximizing overlap with other kernels. Meanwhile, inputs are reordered based on the optimal index for each token and the reordered input will be split based on the size of input.

Dispatch requires a sort of O(S log S), a bin-count of O(S), and an indexing operation of O(SD). The overall complexity O(SD + S log S) is much smaller than the batch matrix multiplication of size O(S <sup>2</sup>EDC). The additional cost is modest and only an extra all-to-all collective whose message size is minimal. Our dynamic gating ensures token dropping will not happen, improving the model's robustness. Our dynamic gating also ensures that no empty placeholders will be transferred between devices, removing the waste in memory allocation of the reordered input and communication volume.

After all tokens are processed by their assigned experts, they are collected through another all-to-all communication collective, sent to their original device, and restored to their original order. This is typically implemented using batch matrix multiplication (BMM) but, as in the first stage, BMM can be replaced with an indexing operation that reduces complexity.

# *B. Reduced Latency and Memory Usage*

We study the impact of the dynamic gating policy on execution time and memory usage on the LM and MT MoE use cases. We keep the machine configuration same as before for a fair comparison. As the dynamic gating policy introduces workload imbalance between different GPUs, for each case, we study the impact of different datasets and tasks on the performance of the model. The experiment is executed by forwarding ten independent epochs on each subset/subtask and recording the average throughput and memory consumption for each batch across the experiments.

Figure [9](#page-7-0) compares the impact of different gating policies on throughput. Results on Expert Buffering and Load Balancing will be explained in Section [VI](#page-6-0) and [VII,](#page-9-0) respectively. Our dynamic gating policy significantly increases throughput across all batch sizes and tasks, compared against the baseline and Tutel gating [\[16\]](#page-11-8). By removing the large dispatch mask, dynamic gating also enables larger batchsizes under the same amount of resources. This improves throughput by up to 6.21×/3.32× when compared against static/Tutel gating under single-node LM, by up to 5.75×/5.33× under MT Encoder, and by up to 2.58×/1.88× under MT decoder.

Since dynamic gating removes the waste factor by reducing the volume of communication, it also reduces communication overheads and improves throughput. Therefore, the benefit of the dynamic gating policy widens when the model is deployed on multiple nodes, where communication overheads are more prominent. Dynamic gating improves throughput by up to 11.55×, 10.98× , 5.71× in LM, MT Encoder, and MT decoder, respectively, when compared against static gating.

Figure [10](#page-8-0) summarizes the effect of different gating policies on memory consumption, using single node cases as an example. Dynamic gating enables larger batch sizes, which runs even faster, when compared against the static gating policy with smaller batch sizes. Dynamic gating reduces the memory footprint by removing the dispatch mask, and also reduces wasted memory allocations for empty paddings and placeholders. As a result, the memory allocated for the activation (bright colors in the figure) for LM with batch size of 8 falls from 6.29GB to 1.28 GB, which is a 79.6% decrease. For MT with batchsize of 8, memory allocations fall from 1.89GB to 1.05 GB, which is a 44.2% decrease. Reducing memory consumption also allows a larger batch size to be allocated under the same machine configuration. The dynamic gating version permits a batch size of 64 for LM and 96 for MT, which is 8× and 2× larger than batch size permitted on the static counterparts.

# VI. EXPERT BUFFERING

<span id="page-6-0"></span>Although dynamic gating reduces waste in computation and dynamic memory allocation associated with the gating function, static memory usage associated with the large number of MoE parameters still puts a huge burden on GPU memory at deployment. The high sparsity in expert activation pattern prompts us to investigate whether there is a way to reduce the memory usage by pruning out the idle experts.

![](_page_7_Figure_0.jpeg)

<span id="page-7-0"></span>Fig. 9. Throughput comparison of different gating policies in MoE models, including static gating (baseline), Tutel gating [\[16\]](#page-11-8), our dynamic gating policy (Sec. [V\)](#page-5-3), dynamic gating with Load Balancing (LB, Sec. [VII\)](#page-9-0), dynamic gating with Expert Buffering (EB, Sec. [VI\)](#page-6-0), and all optimizations combined. Missing bars represent infeasible cases under the corresponding policy and batch size. Eg. Tutel cannot support beyond batch size=32 for LM-1 Node. Dynamic gating reduces memory usage and message sizes in communication, enables larger batch sizes and substantially faster processing times than static gating. Expert buffering trades latency for smaller memory usage while still achieving higher throughput on the MT Decoder. Load balancing further improves latency when combined with dynamic gating and expert buffering. Note that load balancing only makes sense in the context of dynamic gating where each expert gets different number of tokens. Load balancing particularly shines under multi-node setting or combined with expert buffering as it can improve cache miss rate (See Fig [14\)](#page-9-1).

# *A. Sparse Expert Activation*

Our investigation in the expert activation pattern shows that although, in every batch, there exists some experts that are inactive, all experts have been activated a few times across time and batches. Pruning out experts that are not frequently active can potentially hurt model accuracy. However, we can offload the less frequently accessed experts to CPU memory and use the GPU memory for hot and active experts.

We propose the expert buffering mechanism to exploit expert sparsity and reduce static memory allocation. Figure [11](#page-8-1) illustrates the mechanism, which reduces static memory consumption by offloading expert parameters to CPU memory. Since CPU is much slower than GPU for matrix multiplication, we only use CPU memory to hold the experts but do not offload the computation. We use GPU memory to cache active experts and perform computation.

# *B. Cache Management*

During inference, under dynamic gating, once the gating decision is made by the gating function, each GPU receives the number of tokens assigned to its experts. If an expert receives a positive token count, it is considered active for the current batch. The process then checks if the active expert is already cached in GPU memory. If not, then the process will launch a Memcopy to transfer the required expert parameters into the cache. Copying expert parameters from CPU memory to GPU DRAM will be launched in parallel with all-to-all communication, to allow for overlap of data transfers and latency hiding.

In cases where the cache is already full but more experts are needed, eviction will be triggered to make space for the new experts. The eviction policy is designed as follows. First, we will first evict experts that are not active in this batch since they are also less likely to be used in the future due to temporal locality. Next, we will evict expert parameters under a Last In, First Out (LIFO) policy.

The reason for adopting a LIFO policy is rooted in the implementation of recent MoE Transformers. If multiple experts are allocated to a single GPU, MoE Transformer will execute the experts serially in the increasing order of their ids. Consider a small example of E = 4 experts and cache size of 2 experts, and assume expert (1, 2, 3) are needed. After stage 1, expert 1 and 2 will be pushed into the cache, and we need to evict one of them to load expert 3. By evicting expert 2 instead of 1, we ensure the expert with the shortest reuse distance is kept in the cache.

![](_page_8_Figure_0.jpeg)

<span id="page-8-0"></span>Fig. 10. Comparison of memory consumption between MoE models under static and dynamic gating policy. Light shade represents dynamic memory allocation (activation memory). Dark shade represents static memory allocation (model parameters). Missing bars in each plot capture the infeasible cases under the corresponding policy. Compared to Static and Tutel [\[16\]](#page-11-8), Dynamic Gating reduces the memory usage, thus enabling larger batch sizes. Expert Buffering further reduces the memory consumption of model parameters.

![](_page_8_Figure_2.jpeg)

<span id="page-8-1"></span>Fig. 11. Illustration of the Expert Buffering mechanism. We move the expert parameters to CPU memory to reduce burden on GPU memory. On GPU memory, we allocate space only for a few expert entries to buffer active or hot experts. (1) During inference, the all-to-all size message sent in stage 1 as shown in Figure [8\(](#page-5-0)b) signals which experts located in the current device are active. (2) Then the expert cache will check whether the active experts currently reside in the buffer. (3a) If found (cache hit), parameters in the expert buffer will be used to process the tokens. (3b) If not found (cache miss), then the expert parameters will be requested from the CPU memory. The number of cache entries on GPU memory is a tunable parameter to adjust for desirable GPU memory usage and latency (See Section [VI\)](#page-6-0).

# *C. Cache Miss Rates*

To estimate the technical feasibility of Expert Buffering, we calculate cache miss rate for machine translation use case on layers that exhibit enormous expert sparsity. We note that the cache is deployed per device, and each device caches experts that have been assigned to it. As a result, we may vary cache

![](_page_8_Figure_6.jpeg)

<span id="page-8-2"></span>Fig. 12. Worst-Case Cache Miss Rate obtained from traces of expert activations from MT decoders. We tune the cache size per GPU from 1 to 16, and examine the impact of cache size on the worst case Cache Miss Rate on different layers of MT decoders over different tasks. (a) caching performance with/without any reassignment. (b) caching performance against the theoretical optimal Belady's MIN. The miss rate is further reduced by load balancing, and is very close to Belady's MIN (See Sec. [VII\)](#page-9-0).

size from 1 to 16 experts, generating a saving of 0-32.2% on total static memory allocation. We calculate the global cache miss rate under each circumstance. The worst-case cache miss rates are shown in Figure [12.](#page-8-2) We notice the Cache Miss Rate starts to decrease faster when the cache size is larger than 5 per GPU, which is a cache size of 40 in total. This result is consistent with our previous observation that there will be more than 90 experts being empty inside the decoder. We further compare our caching policy with Belady's MIN, the theoretical optimal policy that requires information from the future. Fig. [12\(](#page-8-2)b) shows that the LIFO policy is better than FIFO, and with the load balancing introduced in Sec. [VII,](#page-9-0) our policy can obtain a cache miss rate very close to Belady's MIN.

To evaluate the impact of the proposed mechanism, we perform experiments on Expert Buffering on the MT Decoder. The cache size is selected to be around 80 experts in total, which fits 10 experts per GPU under single node case. This is the point where the cache miss rate starts showing saturation behavior in Figure [12\(](#page-8-2)a).

Figure [9](#page-7-0) and Figure [10](#page-8-0) show the impact of expert buffering on throughput and the static memory allocations for MT. Expert buffering has successfully reduced the static memory consumption by 2.25GB. This memory reduction is particularly useful for users with limited number of GPUs. Although the throughput becomes smaller compared to memory-intensive dynamic gating, the throughput obtained by expert caching is comparable to baselines under single-node, and 2.21×/4.30× under 2/4 nodes.

Furthermore, we study the latency-memory tradeoff incurred by the expert caching mechanism, and estimate the pareto frontier. Figure [13](#page-9-2) shows the latency and memory consumption under a series of cache configurations. We vary the cache per GPU and measure the decoder latency and peak memory consumption. The result shows that the pareto frontier is similar

![](_page_9_Figure_0.jpeg)

<span id="page-9-2"></span>Fig. 13. Tradeoff between memory and latency under different cache configuration on MT Decoder. Corresponding cache size per GPU is marked on the plot.

to the outliers on the cache miss rate plot in Figure 12(a).

Our findings also indicate that the primary contributor to increased latency is the constrained CPU-GPU bandwidth, which we observed to saturate at 12GB/s during our experiments. The findings indicate that layers with a high cache miss rate may impede performance. Adopting technologies that enhance CPU-GPU bandwidth, such as the NVIDIA Grace Hopper superchip and PCIe v5 can mitigate the latency issues in situations where memory is constrained.

We note that no prior work exploits the unique characteristics of MoE Transformers to optimize memory usage. As a caching strategy that is specifically tailored for MoE models, expert buffering is orthogonal to prior memory saving mechanisms such as offloading [25], [30] and can be seamlessly integrated for greater memory savings.

#### VII. LOAD BALANCING

<span id="page-9-0"></span>As we saw in Section IV, token assignments to experts are highly imbalanced, hence the load assigned to each device is also highly imbalanced. Those devices hosting hot experts can become bottlenecks and become more vulnerable to out of memory error. Moreover, devices hosting cold experts may sit idle while waiting for devices that are hosting hot experts to finish their load. As a result, load balancing is critical for having a robust and stable model.

We propose a simple load balancing scheme during the model deployment. We optimize the allocation of experts by leveraging historical load data. Specifically, we encode historical expert allocation into a matrix, and balance the load on each device accordingly. We combine higher-loaded experts with lower-loaded experts, so that the load can be distributed to different devices. We denote the expert placement with  $P_{mn}$ , where m is the expert id  $(m=1\ldots E)$ , n is the device id  $(n=1\ldots D)$  and  $P_{mn}=1$  indicates that the m-th expert is allocated on the n-th device. We also denote the expert activation with  $A_{mb}$ , where m is the expert id and b is the batch id  $(b=1\ldots B)$ , and  $A_{mb}$  represents the fraction of tokens assigned to expert e at batch id b. The problem can be thus formalized as follows:

$$\min \max_{m,b} |\sum_n P_{mn} A_{mb} - \frac{1}{D}| \text{ subject to } \sum_m P_{mn} = \frac{E}{D} \ \forall n$$

This problem can be reduced to the multi-way number partitioning problem [10], which is NP-hard. To balance the

![](_page_9_Figure_10.jpeg)

<span id="page-9-1"></span>Fig. 14. The effect of our proposed load balancing mechanism: Based on historical activation data, our algorithm is able to significantly reduce the load imbalance problem on LM tasks and improve the robustness by limiting the maximum workload on a single device.

memory usage and simplify the communication process, each GPU should be assigned the same number of experts.

#### A. Greedy Balancing for Independent Activation

We utilize a greedy algorithm to generate approximations to the optimal assignment. We sort the experts by their average work load in historical data  $\tilde{A}_m$ , and assign the experts to GPUs on a descending order. At each step, an expert is assigned to the GPU with the smallest load, calculated by  $\sum_m P_{mn}\tilde{A}_m$ . Once a GPU reaches the designated capacity, it will be removed from the list of candidates.

Figure 14 summarizes the balance of the load under the original order and the new order. The balance of load is estimated using existing activation data introduced in Section III-B. To perform the experiment, we separate the data into two halves. We use the first half of the activation data to generate a device assignment for each expert, then estimate the work load under generated assignment using the second half of the activation data. Results are normalized by the total batch size, which means the numbers represent the share of the total number of tokens each device will handle in a certain batch.

We record the Max Load, which is the maximum share of the load that has ever appeared in all batches on the test set, and the Avg Max Load, which is the maximum share of the load averaged over all batches. The Max Load estimates the worst case scenario that relates to out-of-memory error, and the Avg Max Load estimates the average case where imbalance on work load can lead to bottlenecks and harm inference speed.

Results show that Greedy is able to balance the load for LM use case by improving both metrics (Max Load and Avg Max load per device) significantly, reducing from more than 0.6 to less than 0.4. Similarly for Machine Translation use case, Greedy can balance the expert load assignment for the MT encoder. Figure 9 shows the benefit of the Greedy Rebalancing on the LM and MT Encoder: Rebalancing increases the throughput by a maximum of 10.1% and 19.5% compared against pure dynamic gating. Furthermore, rebalancing allows the LM to achieve a batchsize of 64 and 128 when it is deployed on four nodes, making the model more robust.

#### B. Anti-correlation Balancing for Correlated Activation

Greedy is less effective for the Machine Translation Decoder. We found that expert activation level becomes a less effective indicator in this case due to correlation between experts. To handle this problem, we propose Anti-correlation Balancing, which takes correlation into consideration. Denoting the Pearson correlation between the current expert a to expert b in the historical data as  $S_{ab}$ , the current work load can be modified from  $\sum_{m} P_{mn} \tilde{A}_{m}$  to  $\sum_{m} P_{mn} (\tilde{A}_{m} + 0.5 * S_{am})$ . This algorithm successfully reduces the Avg Max Load and the Max Load on most cases. We notice that a more balanced work load also has a positive impact on the cache miss rate. As shown in Figure 12 and 9, the worst-case cache miss rate decreases for all cache sizes over MT Decoders, which leads to a maximum increase of 1.9% on their throughputs.

#### VIII. RELATED WORK

While the MoE Transformer substantially reduces the training cost and FLOPS for large models, the outrageous size of MoE Transformers and the complex expert parallelism [21] poses obstacles for its deployment, including the high GPU memory requirement and the excessive communication overhead of expert assignment. Various approaches have been invented to relieve these obstacles. Switch Transformer [7] and ELSLM [2] use knowledge distillation to distill a large MoE Transformer into a dense model. While distillation reduces the number of parameters, only a small portion (about 30%) of the accuracy gain can be retained. The MoS strategy proposed in DeepSpeed-MoE [24] distills the knowledge to a smaller MoE Transformer with less layers and shared experts. SE-MoE [30] uses pruning to reduce the number of experts in the model. WideNet [?] and MPoE [9] reduce the number of parameters by enforcing parameter sharing. Beyond reducing the parameters, other methods directly reduce computation and communication. The BASE Layer and Switch Transformer also reduce the number of experts each token is assigned to reduce the communication volume and computation. V-MoE [26] further reduces the number by dropping out a large portion of tokens. Hash Layer [27] replaces the gating layer with a precomputed hash function, which reduces the computation cost, but doesn't alleviate the communication overhead. As the MoE Transformer is a type of Transformer, techniques and optimized architectures that enhance Transformer inference speed may apply. Relevant examples include Reformer [19], Longformer [3], and Terraformer [17]. However, there is scant discussion of their application to MoE Transformers.

Offloading and swapping strategies such as [15] swaps unused tensors form the GPU memory to the main memory to reduce the resource requirement. However, existing strategies can only be applied on dense models. Applying these strategies efficiently on conditional neural networks such as MoE is non-trivial, since the data flow graph cannot be constructed in advance due to the conditional computation. FastMoE [13], [14] designed customized communication primitives and gating kernels for token assignment to reduce the communication overhead, but it has not been tested on outrageously

large neural networks. Tutel [16] and DeepSpeed-MoE [24] improve MoE model performance on datacenter-scale systems by combining system and architecture methods with tailored kernels for both Transformer and MoE layers, and specialized communication primitives. The approach combines expert parallelism, model parallelism, and tensor parallelism to significantly boost throughput and reduce latency. However, DeepSpeed-MoE is not designed to conserve GPU resources and therefore may be impractical for many academic users. SE-MoE [30] utilizes Ring Memory offloading to reduce GPU usage, achieving better throughput than DeepSpeed-MoE in low-resource scenarios. However, this approach does not leverage expert parallelism from MoE Transformers.

### IX. CONCLUSION

While at training time, mixtures of expert (MoE) models show superior performance to their flop-equivalent dense counterpart models, they are notoriously large, need a large number of GPUs to deploy and hard to democratize. Researchers outside large industry labs do not have access to hundreds or thousands of GPUs to afford exploring such large models. Moreover, they are much slower than their dense counterparts at inference. To overcome these challenges, we propose three optimization techniques (Dynamic Gating, Expert Buffering, and Expert Load Balancing) to improve memory and latency profile of such models for deployment.

#### REFERENCES

- "FasterTransformer," https://github.com/NVIDIA/FasterTransformer, accessed: 2023-03.
- <span id="page-10-0"></span>[2] M. Artetxe, S. Bhosale, N. Goyal, T. Mihaylov, M. Ott, S. Shleifer, X. V. Lin, J. Du, S. Iyer, R. Pasunuru et al., "Efficient large scale language modeling with mixtures of experts," arXiv preprint arXiv:2112.10684, 2021.
- <span id="page-10-8"></span>[3] I. Beltagy, M. E. Peters, and A. Cohan, "Longformer: The long-document transformer," arXiv preprint arXiv:2004.05150, 2020.
- <span id="page-10-4"></span>[4] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., "An image is worth 16x16 words: Transformers for image recognition at scale," arXiv preprint arXiv:2010.11929, 2020.
- <span id="page-10-2"></span>[5] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat et al., "Glam: Efficient scaling of language models with mixture-of-experts," arXiv preprint arXiv:2112.06905, 2021.
- [6] W. Fedus, J. Dean, and B. Zoph, "A review of sparse expert models in deep learning," arXiv preprint arXiv:2209.01667, 2022.
- <span id="page-10-1"></span>[7] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *Journal* of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022. [Online]. Available: http://jmlr.org/papers/v23/21-0998.html
- <span id="page-10-3"></span>[8] L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima *et al.*, "The pile: An 800gb dataset of diverse text for language modeling," *arXiv preprint* arXiv:2101.00027, 2020.
- <span id="page-10-7"></span>[9] Z.-F. Gao, P. Liu, W. X. Zhao, Z.-Y. Lu, and J.-R. Wen, "Parameter-efficient mixture-of-experts architecture for pre-trained language models," arXiv preprint arXiv:2203.01104, 2022.
- <span id="page-10-6"></span>[10] R. L. Graham, "Bounds on multiprocessing timing anomalies," SIAM journal on Applied Mathematics, vol. 17, no. 2, pp. 416–429, 1969.
- <span id="page-10-5"></span>[11] S. Gupta, S. Mukherjee, K. Subudhi, E. Gonzalez, D. Jose, A. H. Awadallah, and J. Gao, "Sparsely activated mixture-of-experts are robust multi-task learners," arXiv preprint arXiv:2204.07689, 2022.

- <span id="page-11-13"></span>[12] H. Hazimeh, Z. Zhao, A. Chowdhery, M. Sathiamoorthy, Y. Chen, R. Mazumder, L. Hong, and E. Chi, "Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning," *Advances in Neural Information Processing Systems*, vol. 34, pp. 29 335– 29 347, 2021.
- <span id="page-11-17"></span>[13] J. He, J. Qiu, A. Zeng, Z. Yang, J. Zhai, and J. Tang, "Fastmoe: A fast mixture-of-expert training system," *arXiv preprint [arXiv:2103.13262](http://arxiv.org/abs/2103.13262)*, 2021.
- <span id="page-11-22"></span>[14] J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, and Q. Li, "Fastermoe: Modeling and optimizing training of largescale dynamic pre-trained models," in *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, ser. PPoPP '22. New York, NY, USA: Association for Computing Machinery, 2022, p. 120–134. [Online]. Available: <https://doi.org/10.1145/3503221.3508418>
- <span id="page-11-21"></span>[15] C.-C. Huang, G. Jin, and J. Li, "Swapadvisor: Pushing deep learning beyond the gpu memory limit via smart swapping," in *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2020, pp. 1341–1355.
- <span id="page-11-8"></span>[16] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram, J. Chau, P. Cheng, F. Yang, M. Yang, and Y. Xiong, "Tutel: Adaptive mixture-of-experts at scale," *CoRR*, vol. abs/2206.03382, Jun. 2022. [Online]. Available: [https://arxiv.org/pdf/](https://arxiv.org/pdf/2206.03382.pdf) [2206.03382.pdf](https://arxiv.org/pdf/2206.03382.pdf)
- <span id="page-11-20"></span>[17] S. Jaszczur, A. Chowdhery, A. Mohiuddin, L. Kaiser, W. Gajewski, H. Michalewski, and J. Kanerva, "Sparse is enough in scaling transformers," *Advances in Neural Information Processing Systems*, vol. 34, pp. 9895–9907, 2021.
- <span id="page-11-16"></span>[18] Y. J. Kim, A. A. Awan, A. Muzio, A. F. C. Salinas, L. Lu, A. Hendy, S. Rajbhandari, Y. He, and H. H. Awadalla, "Scalable and efficient moe training for multitask multilingual models," *arXiv preprint [arXiv:2109.10465](http://arxiv.org/abs/2109.10465)*, 2021.
- <span id="page-11-19"></span>[19] N. Kitaev, Ł. Kaiser, and A. Levskaya, "Reformer: The efficient transformer," *arXiv preprint [arXiv:2001.04451](http://arxiv.org/abs/2001.04451)*, 2020.
- <span id="page-11-14"></span>[20] S. Kudugunta, Y. Huang, A. Bapna, M. Krikun, D. Lepikhin, M.-T. Luong, and O. Firat, "Beyond distillation: Task-level mixture-of-experts for efficient inference," *arXiv preprint [arXiv:2110.03742](http://arxiv.org/abs/2110.03742)*, 2021.
- <span id="page-11-1"></span>[21] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," *arXiv preprint [arXiv:2006.16668](http://arxiv.org/abs/2006.16668)*, 2020.
- <span id="page-11-2"></span>[22] NLLB Team, M. R. Costa-jussa, J. Cross, O. C¸ elebi, M. Elbayad, ` K. Heafield, K. Heffernan, E. Kalbassi, J. Lam, D. Licht, J. Maillard, A. Sun, S. Wang, G. Wenzek, A. Youngblood, B. Akula, L. Barrault, G. Mejia-Gonzalez, P. Hansanti, J. Hoffman, S. Jarrett, K. R. Sadagopan, D. Rowe, S. Spruit, C. Tran, P. Andrews, N. F. Ayan, S. Bhosale, S. Edunov, A. Fan, C. Gao, V. Goswami, F. Guzman, P. Koehn, ´ A. Mourachko, C. Ropers, S. Saleem, H. Schwenk, and J. Wang, "No language left behind: Scaling human-centered machine translation," 2022.
- <span id="page-11-10"></span>[23] M. Ott, S. Edunov, A. Baevski, A. Fan, S. Gross, N. Ng, D. Grangier, and M. Auli, "fairseq: A fast, extensible toolkit for sequence modeling," in *Proceedings of NAACL-HLT 2019: Demonstrations*, 2019.
- <span id="page-11-9"></span>[24] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale," *arXiv preprint [arXiv:2201.05596](http://arxiv.org/abs/2201.05596)*, 2022.
- <span id="page-11-18"></span>[25] J. Ren, S. Rajbhandari, R. Y. Aminabadi, O. Ruwase, S. Yang, M. Zhang, D. Li, and Y. He, "{ZeRO-Offload}: Democratizing {Billion-Scale} model training," in *2021 USENIX Annual Technical Conference (USENIX ATC 21)*, 2021, pp. 551–564.
- <span id="page-11-6"></span>[26] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, and N. Houlsby, "Scaling vision with sparse mixture of experts," *Advances in Neural Information Processing Systems*, vol. 34, 2021.
- <span id="page-11-5"></span>[27] S. Roller, S. Sukhbaatar, J. Weston *et al.*, "Hash layers for large sparse models," *Advances in Neural Information Processing Systems*, vol. 34, 2021.
- <span id="page-11-0"></span>[28] J. Sevilla, L. Heim, A. Ho, T. Besiroglu, M. Hobbhahn, and P. Villalobos, "Compute trends across three eras of machine learning," *arXiv preprint [arXiv:2202.05924](http://arxiv.org/abs/2202.05924)*, 2022.
- <span id="page-11-11"></span>[29] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," *arXiv preprint [arXiv:1701.06538](http://arxiv.org/abs/1701.06538)*, 2017.

- <span id="page-11-3"></span>[30] L. Shen, Z. Wu, W. Gong, H. Hao, Y. Bai, H. Wu, X. Wu, H. Xiong, D. Yu, and Y. Ma, "Se-moe: A scalable and efficient mixture-ofexperts distributed training and inference system," *arXiv preprint [arXiv:2205.10034](http://arxiv.org/abs/2205.10034)*, 2022.
- [31] Y. Tay, M. Dehghani, D. Bahri, and D. Metzler, "Efficient transformers: A survey," *arXiv preprint [arXiv:2009.06732](http://arxiv.org/abs/2009.06732)*, 2020.
- <span id="page-11-12"></span>[32] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," *Advances in neural information processing systems*, vol. 30, 2017.
- <span id="page-11-7"></span>[33] F. Xue, Z. Shi, F. Wei, Y. Lou, Y. Liu, and Y. You, "Go wider instead of deeper," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 36, no. 8, 2022, pp. 8779–8787.
- <span id="page-11-4"></span>[34] A. Yang, J. Lin, R. Men, C. Zhou, L. Jiang, X. Jia, A. Wang, J. Zhang, J. Wang, Y. Li *et al.*, "Exploring sparse expert models and beyond," *arXiv preprint [arXiv:2105.15082](http://arxiv.org/abs/2105.15082)*, 2021.
- <span id="page-11-15"></span>[35] Z. You, S. Feng, D. Su, and D. Yu, "Speechmoe: Scaling to large acoustic models with dynamic routing mixture of experts," *arXiv preprint [arXiv:2105.03036](http://arxiv.org/abs/2105.03036)*, 2021.