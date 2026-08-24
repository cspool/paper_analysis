# <span id="page-2-5"></span>2 Preliminary

## 2.1 Parallelisms in Distributed Training

As model sizes and data volumes grow rapidly, various distributed parallelism techniques are employed in LLM training to distribute the workload across multiple devices.

**2.1.1 Data Parallelism.** Data parallelism (DP) [28] splits the data along sample dimension. Each device is responsible for a part of the input samples, and the gradients need to be synchronized across the devices. DP requires each device to maintain a complete copy of the model, which is redundant. To address this, sharded data parallelism (SDP) has been proposed, such as DeepSpeed-ZeRO [38] and PyTorch FSDP [52]. These methods not only split the data but also the

model states across devices, allowing each device to store only a fraction of the model states and introducing additional communication to synchronize the model states.

**2.1.2 Sequence Parallelism.** Sequence parallelism (SP) also splits data, and can be considered a special form of DP. Unlike DP, which splits data across the sample dimension, sequence parallelism splits data across the sequence dimension. SP is designed to mitigate the memory shortage caused by increasingly longer context lengths of LLMs. DeepSpeed-Ulysses [19] proposes Ulysses-style SP, which splits the sequence dimension of linear projection in MLP and attention module, dropout modules, and normalization modules, and employs All-to-All primitives to collect and distribute sequences.

$$O_s, K_s, V_s = X_s W_O, X_s W_K, X_s W_V \in \mathbb{R}^{\frac{N}{P} \times d}$$
 (1)

$$\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h = \text{AlltoAll}(\mathbf{Q}_s, \mathbf{K}_s, \mathbf{V}_s) \in \mathbb{R}^{N \times \frac{d}{P}}$$
 (2)

<span id="page-2-2"></span><span id="page-2-1"></span><span id="page-2-0"></span>
$$\mathbf{P}_h = \operatorname{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^{\top}}{\sqrt{d}}\right) \mathbf{V}_h \in \mathbb{R}^{N \times \frac{d}{P}}$$
 (3)

<span id="page-2-3"></span>
$$O_s = AlltoAll(P_h W_O) \in \mathbb{R}^{\frac{N}{P} \times d}$$
 (4)

In the attention module of Ulysses-style SP, each device holds a portion of sequences  $\mathbf{X}_s \in \mathbb{R}^{\frac{N}{P} \times d}$  and the complete model parameters  $\mathbf{W}_Q$ ,  $\mathbf{W}_K$ ,  $\mathbf{W}_V$ ,  $\mathbf{W}_O \in \mathbb{R}^{d \times d}$ , where N denotes the total sequence length, P denotes the SP degree, and d denotes the hidden size. After the linear projection of query, key and value (Eq. 1), three rounds of All-to-All communication are employed to collect the complete sequence on each device (Eq. 2). The multi-head attention operation is then calculated on the complete sentence (Eq. 3), and an another round of All-to-All communication is introduced to distribute the sequence across the devices (Eq. 4).

Megatron-LM also proposes Megatron-style SP [22], which splits only the dropout and normalization modules, and requires All-Gather and Reduce-Scatter communication. It can be treated a supplementary scheme proposed to be used in conjunction with Megatron-TP (§2.1.3), aimed at addressing the redundant activation memory usage of Megatron-TP, and they must have the same parallelism degree.

## <span id="page-2-4"></span>2.1.3 Other parallelisms.

Model Parallelism. Model parallelism distributes the model parameters across the cluster, which can be divided into two categories: tensor parallelism (TP) and pipeline parallelism (PP). TP splits the model vertically. Megatron-TP proposed by Megatron-LM [33] is the most widely used, which splits the tensor multiplication computations in each attention layer and feed-forward layer across multiple devices, incorporating communication to synchronize the computation results. PP [16, 18, 31, 32] partitions the model horizontally, with

GPipe [\[18\]](#page-13-11), PipeDream-Flush [\[32\]](#page-14-15) being notable implementations. These approaches divide the model layers into multiple stages placed across different devices, pass intermediate computation results with point-to-point communication, and orchestrate the model execution into a pipeline.

Context Parallelism. Context parallelism (CP) [\[6,](#page-12-3) [24,](#page-13-4) [27,](#page-13-5) [29\]](#page-13-6) also splits the sequence dimension. Compared to sequence parallelism (SP) which splits dropout and normalization module activations but necessitates complete sentence for attention operation, CP further distributes the attention operation. Specifically, CP distributes sequence dimension of the query, key, and value across multiple devices, and involves additional ring communication to collect key and value for completing attention computations. Such extra communication volume is substantial, thus CP allows the computation to overlap the extra communication overhead by conducting the ring communication and computation of attention operation chunk by chunk.

