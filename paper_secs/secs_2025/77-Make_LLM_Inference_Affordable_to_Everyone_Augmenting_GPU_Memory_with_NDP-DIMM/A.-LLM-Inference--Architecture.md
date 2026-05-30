# *A. LLM Inference & Architecture*

As shown in Figure 2, the inference procedure of transformer-based LLMs comprises two stages: prompting and token generation. During the prompting stage, an input sequence is used to produce keys and values (KV cache)

<sup>1</sup>This paper defines a neuron as a specific row/column in a weight matrix, and neurons will not be activated when associated with zero activations.

![](_page_2_Figure_0.jpeg)

Fig. 2. The LLM inference procedure and architecture

for each transformer layer in the LLM, and this is done just once per inference. In the token generation stage, previously generated tokens are used to update the KV cache and generate new tokens incrementally. This stage is executed multiple times, depending on the length of the output sequence. Since token generation accounts for more than 90% of the total runtime [32], this paper primarily focuses on optimizing inference efficiency in token generation.

An LLM has multiple transformer layers, each containing a self-attention and an MLP block. In the self-attention block, input x is projected linearly to produce Q, K and V, processed by the attention operator to yield the attention result, and then computed by the projection layer for the MLP input. The MLP block includes fully connected (FC) layers and nonlinear functions. For example, the OPT model uses two FC layers which are connected by one ReLU activation function.

## B. Activation Sparsity in LLMs

The activation function such as ReLU in the MLP block introduces the intrinsic activation sparsity to LLMs [34], [38], [52]. As shown in Figure 3a, the ReLU function in the MLP block, can turn many activation values to zero, eliminating the need to load and compute these inactive neurons. As the red dashed box shows, a neuron in this paper represents a specific row or column within a weight matrix. For example, due to the ReLU function zeros out the 1st, 4th and 5th input values of the FC2 layer, the corresponding columns and rows in FC1 and FC2 weight matrix will not be activated.

To further achieve activation sparsity on self-attention blocks, programmers insert ReLU functions before QKV generation [38], as illustrated in Figure 3b. For LLMs that do not use ReLU as their activation function, such as LLaMA (SiLU) and Falcon (GELU) [4], [57], recent work has demonstrated that they can also be replaced by ReLU functions [38], [52], as demonstrated in Figure 3c. Previous studies [52], [54], [66] also demonstrated that the activation sparsity within LLMs provides significant sparsity (ranging from 70% to 90%) with negligible accuracy degradation (less than 1%).

## C. Offloading-based LLM Inference Systems

Most existing LLM inference systems [28], [41], [45], [61] require the use of expensive server-grade GPUs, which provide

![](_page_2_Figure_9.jpeg)

Fig. 3. The inherent activation sparsity within certain LLMs is further enhanced to achieve higher sparsity across various LLMs.

high-capacity HBM to store the large-scale LLM parameters. This limits their deployment to easily accessible and affordable hardware. Offloading is a viable technique to enable LLM inference on such commodity hardware [23]. For instance, a single consumer-grade GPU can leverage the host memory resources to perform inference of LLaMA2-70B [22], [45].

Existing offloading-based inference systems utilize host memory to extend the storage capacity of the GPU to accommodate LLMs. As long as there is sufficient host memory, this strategy can be used to perform inference on LLMs of various sizes. HuggingFace Accelerate [23] integrates offloading techniques from training systems by automatically mapping and partitioning weights into GPU and host memory respectively, only transferring the necessary parameters during inference. However, the characteristics of LLM inference are quite different from training [10], making it inefficient. To address this issue, FlexGen [50] provides a novel zigzag offloading strategy to maximize the inference throughput within a limited PCIe bandwidth. This zig-zag scheduling strategy integrates multiple tokens into a block and overlaps the weight-loading cost during token processing within one block. For instance, it computes all the tokens in one block (e.g., more than 100 tokens) with the weights in layer i, while prefetching the weights in layer i + 1 simultaneously. The burdensome block computation in one layer effectively overlaps the weight prefetching cost for the next layer, especially for the prefill phase which occupies multiple tokens even with a single batch. However, this method is unsuitable for local deployment scenarios, which only occupy limited batch sizes [21] during token generation. Deja Vu [34] further exploits activation sparsity to perform LLM inference by predicting and loading only the activated neurons, thereby reducing data access and computation overhead. However, since the activated neurons are dynamic and cannot be preloaded into the limited consumer-grade GPU memory, data still need to be loaded from host memory, resulting in inference efficiency being bounded by PCIe bandwidth. Overall, while existing offloading solutions can effectively extend the storage capacity of inference systems to support larger LLMs, the low bandwidth data transfer of PCIe results in poor inference performance.

## III. MOTIVATIONS & CHALLENGES

## A. Why NDP-DIMM Enhanced GPU?

Offloading is essential for LLM inference on low-budget systems with a single consumer-grade GPU. However, as noted in Section II-C, even utilizing activation sparsity to reduce weight parameter access, the PCIe bandwidth remains the bottleneck. Thus, costly data transfers between extended memory and GPU must be minimized. However, simply offloading the corresponding computation of cold neurons on the host CPU [17], [53] can only achieve a limited performance improvement, as the host CPU can only access DRAM with limited improved bandwidth than PCIe (e.g., 89.6 GB/s vs. 64 GB/s). To this end, we choose to employ multiple NDP-DIMMs as the extended memory, as they offer comparable bandwidth and larger storage capacity than a single consumergrade GPU. Need to mention that as a budget-friendly host memory solution, we do not consider high-performance but expensive HBM-PIM and AiM [11], [43] in this study. Given the limited computation capability, only utilizing the processing units in NDP-DIMMs cannot boost the inference efficiency [58]. Consequently, we are motivated to use NDP-DIMMs to enhance GPU for efficient LLM inference.

Our observation indicates that the activation sparsity within LLMs effectively partitions weight parameters into two distinct regions, which are ideally suited to consumer-grade GPU and NDP-DIMMs, respectively. Specifically, activation sparsity in LLMs follows a power-law distribution [53], [59]. About 20% of neurons (hot neurons) account for 80% of computations, while 80% (cold neurons) handle only 20%. Hot neurons, with  $16 \times$  higher computation intensity, fit GPU memory, while cold neurons suit NDP-DIMMs. During inference, GPU can provide high computation capability for hot neurons and NDP-DIMMs enable the cold neurons computation in memory.

## B. Necessity of Hot/Cold Neuron Partition

Hot/cold neuron partition impacts the computational load on GPU/NDP-DIMMs, affecting the inference performance of the heterogeneous system. Due to the input-specific nature of activation sparsity, solely relying on the offline partition is insufficient. Our evaluation on LLaMA2-70B reveals significant dynamics in when the neuron will be activated (hereafter, neuron activity patterns) during inference. Approximately 52% of the initialized hot neurons exhibit varied activity during inference. This variability in neuron behavior results in suboptimal performance with a fixed hot/cold partition, causing a  $1.63 \times$  degradation compared to an oracle (the theoretically optimal partition) scheme. Thus, we must dynamically predict and adjust the hot/cold neuron partition.

However, typical MLP-based predictors [34], [38], [52], [64] for activation sparsity in LLMs are costly. For example, predicting the activated neurons in LLaMA-7B needs perlayer MLP-based predictors, requiring an extra 2GB storage and inducing 10%-25% inference runtime. Fortunately, the inherent locality of activation sparsity leads us to design a lightweight and accurate predictor for efficient online partition

![](_page_3_Figure_7.jpeg)

Fig. 4. Distribution patterns for activation sparsity. (a) The adjacent tokens enjoy high similarity on activated neurons for various models and datasets. (b) The activated neurons between consecutive layers are highly correlated.

adjustments. To be specific, we found that activation sparsity in LLM inference shows considerable token-wise similarity and layer-wise correlation, worth exploiting.

- 1) Token-wise Similarity: We analyzed the similarity between tokens to explore the distribution characteristics of activation sparsity. As shown in Figure 4a, we evaluate the tokenwise similarity for LLaMA-13B and Falcon-40B with multiple widely adopted datasets, including COPA [46], Wikitext2 [37] and PIQA [7]. As one can notice, the adjacent tokens have a higher distribution similarity than distant tokens. Specifically, the similarity between adjacent tokens exceeds 90% (95% for Falcon-40B), but drops to 70% once the tokens' distance exceeds 10. This indicates that in context, adjacent tokens often express similar meanings, leading to high similarity in their activity distribution. Additionally, we observe that when the distance between tokens exceeds 25, the distribution similarity almost no longer decreases, indicating that beyond a certain window size, the semantic correlation becomes weak and has less impact on the overall distribution.
- 2) Layer-wise Correlation: We further observed that the distribution of activated neurons in two consecutive layers is highly correlated. As shown in Figure 4b, when the 6th neuron in layer-30 of LLaMA-13B is activated, the probability of neurons 0 and 5 being activated in layer-31 exceeds 90%. This suggests that we can use the results of the preceding layer to predict the distribution of activated neurons in the current layer.

Overall, the token-wise similarity and layer-wise correlation motivate us to design a lightweight online predictor based on historical activation information. According to the prediction results, we can online adjust the hot/cold neurons partition to effectively exploit the processing advantages of the consumergrade GPU and NDP-DIMMs, respectively.

## C. Load Imbalance across Multiple NDP-DIMMs

Due to the storage limitation of a single DIMM, multiple DIMMs are required to store all the neurons (weight parameters) in LLM. Specifically, one DIMM only stores portions of the neurons and the corresponding processing unit can only directly assess neurons in the DIMM with high internal bandwidth. However, due to the input-specific nature of activation sparsity, the computational load on each NDP-DIMM can be diverse. For example, when fixing the cold neuron distribution on multiple DIMMs for LLaMA-13B,

the most overloaded NDP-DIMM will have 1.2-2.5× more computational load than others.

Therefore, we need an online scheduling strategy to remap the cold neuron across DIMMs to achieve load balance. Meanwhile, an efficient data transmission pathway among DIMMs is essential to help adjust the neuron placement. By optimizing neuron computation scheduling, we can minimize data transfers across NDP-DIMMs while ensuring balanced computational loads across DIMMs. This ensures that all parts of the system can maximize their performance.

In summary, the NDP-DIMM enhanced GPU approach effectively addresses the substantial data transfer overhead in offloading processes, providing a promising solution to improve LLM inference efficiency by leveraging the activation sparsity patterns inherent in LLMs.

