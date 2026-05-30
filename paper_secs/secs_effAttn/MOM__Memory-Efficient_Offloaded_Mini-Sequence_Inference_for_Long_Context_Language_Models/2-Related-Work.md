# **2 Related Work**

**KV Cache Offloading** KV cache offloading is a technique used in LLM inference to manage memory constraints when processing long sequences. Since the key-value (KV) cache stores past attention states, its size scales linearly with sequence length and can quickly exceed GPU memory capacity. Offloading moves inactive or less frequently accessed KV cache tensors to CPU memory, NVMe storage, or lower-bandwidth GPU memory, freeing up highspeed HBM (High Bandwidth Memory) for active computation. This is particularly useful for batched inference and long-context models, where keeping the entire KV cache in GPU memory would be impractical. Advanced implementations, such as PagedAttention [\(Kwon](#page-11-6) [et al.,](#page-11-6) [2023a\)](#page-11-6), further optimize this by dynamically swapping only necessary KV blocks back to GPU when needed, reducing data transfer latency. Efficient KV cache offloading allows scalable long-sequence inference without excessive memory overhead, improving overall throughput and system efficiency.

**MLP-Dominated Prefill Memory** In large language model (LLM) inference, the prefill stage dominates peak GPU memory consumption, primarily due to the MLP (feed-forward) layers rather than the attention layers (Kalra, 2023). During the prefill stage, the entire input sequence is processed in parallel, requiring  $\mathcal{O}(\text{sequence-length} \times d_{\text{model}}^2)$  memory for the MLP layers. While self-attention contributes to memory usage—particularly due to key-value (KV) cache growth in long sequences—it scales  $\mathcal{O}(\text{sequence-length}^2 \times d_{\text{model}})$  in standard attention, which can be optimized using FlashAttention and Grouped-Query Attention (GQA). In contrast, MLP layers involve large matrix multiplications with weights that cannot be easily pruned or quantized without affecting accuracy, making them the dominant factor in peak GPU memory usage. A detailed illustration is provided in Fig 1. As shown by Figure 9 in Appendix A, during the decode stage, memory usage is dominated by the KV cache rather than the MLP layers. Since only one token is processed per step, it grows linearly with sequence length and does not exceed the peak seen in prefill (Lienhart, 2024).

**Chunked Prefill** To address the MLP bottleneck in prefill stage, Chunked Prefill and its variants are widely used in academia (Agrawal et al., 2023) (Agrawal et al., 2024) and industry (by NVIDIA in TensorRT-LLM (NVIDIA, 2024)) to mitigate the peak memory usage in the prefill stage by splitting the input sequence into smaller chunks (see Algorithm 2 in Appendix B). This allows GPUs to process smaller sections of the input, reducing intermediate memory requirements while keeping high parallelism in matrix multiplications. Chunked prefill is particularly useful for optimizing batch inference workloads, reducing VRAM spikes, and preventing out-of-memory (OOM) errors while maintaining high throughput. Similar to Mini-sequence, it reduces theoretical peak intermediate memory to  $\frac{1}{C}$  of its original size, for a chunk size of C. However, unlike Mini-sequence, which only partitions the MLP layer, chunked prefill splits the entire prefill process and computes each chunk sequentially. As a result, it can result in higher latency compared to full-sequence prefill, as the overhead from multiple kernel launches and data movement may outweigh the benefits for shorter sequences.

Mini-Sequence Transformer The Mini-Sequence Transformer (MST) optimizes LLM training by internally partitioning input sequences into mini-sequences before each MLP layer, reducing intermediate memory usage in MLP and LM-Head layers. This method minimizes peak memory consumption while maintaining full-sequence accuracy and throughput. MST enables 12× longer sequence training without degradation, extending models like Llama3 (Grattafiori et al., 2024), Qwen (Bai et al., 2023), Mistral (Jiang et al., 2023) and Gemma (Riviere et al., 2024) by 12-24×. Applying MST-like chunking to inference offers key advantages over Chunked Prefill, which splits input sequences dynamically at runtime, causing memory fragmentation, synchronization overhead, and inefficient GPU utilization. MST, by contrast, naively partitions sequences within the model architecture, reducing activation memory without extra inference-time computation.

#### 3 Method

In this work, we propose Memory-efficient Offloaded Mini-Sequence Inference (MOM) for long context. Let  $A \in \mathbb{R}^{B \times S \times d}$  denote the input sequence's representation to the MLP layer, where B is the batch size (we assume B=1 in this paper), S is the sequence length, and d is the hidden dimension. The core idea of Mini-sequence is to partition A into M shorter sequences  $(A_1, A_2, \ldots, A_M)$ , where each sequence  $A_i \in \mathbb{R}^{B \times N \times d}$  with  $N \approx S/M$ . In our inference setting, we apply Mini-sequence exclusively to the MLPs and only take the last token to feed the last MLP layer and LM-head block, leaving the attention layers unchanged so that existing optimizations such as FlashAttention and Grouped-Query Attention can continue to operate. Crucially, by decoupling from gradient computations, we can integrate offloading to move KV caches to CPU memory (or disk) when they are not actively used, thereby further reducing the GPU memory footprint.

### 3.1 Mini-Sequence Processing for Inference

During the prefill stage, where the entire prompt is processed to initialize the KV cache, we employ internal chunking within the MLP blocks as shown in Figure 3. When performing autoregressive decoding, we only project the final token's hidden state to the last MLP layer and LMHead layer to obtain the next-token logits. This is formalized in Algorithm 1.

![](_page_4_Figure_3.jpeg)

<span id="page-4-0"></span>Figure 3: MOM Architecture Overview.

#### Algorithm 1 Memory-efficient Offloaded Mini-Sequence Inference

```
Require: Input sequence X \in \mathbb{R}^{B \times S \times d}, Mini-sequence size C, offloaded KV cache K, feed-
forward layer MLP, batch size B, sequence length S, and hidden dimension d.
Compute attention layer output A = Attention(X)
Update and offload KV cache to CPU: \mathcal{K} \leftarrow \text{offload}(\mathcal{K}, A)
if last MLP layer then
    Extract last token representation: A_{\text{last}} = A[:, -1, :] \triangleright \text{Select last token's representation}
    Compute MLP output O_{last} = MLP(A_{last})
    Compute logits: L = LLM\_Head(O_{last})
    Transfer offloaded cache back to GPU for decode stage.
                          ▶ Return logits for the last token to start autoregressive decoding
    return L
else
    Partition A into M = \lceil S/C \rceil mini-sequences \{A_i\}_{i=1}^M, where each A_i \in \mathbb{R}^{B \times N \times d} and
N \approx C.
    for i = 1, \ldots, M do
        Compute O_i = MLP(A_i)
                                             ▶ Mini-sequence processing through MLP layers
    Concatenate outputs: O = concat(O_1, ..., O_M).
    return O.
                                         ▷ Continue processing in the next transformer block
end if
```

## **3.2 KV Cache Offloading Integration**

During inference, the Transformer relies on a KV cache to store intermediate attention states. Our method leverages existing offloading mechanisms (e.g., via Hugging Face's transformers.cache utils.OffloadedCache class) to move inactive KV cache tensors to CPU memory, as shown in Figure [4.](#page-5-0) The offloading integration is dynamic: before processing mini-sequences, the corresponding KV caches are updated and offloaded automatically, ensuring that only the minimal set of tensors required for the current computation resides on GPU when the token's representations are processed by MLP layers. During the decode stage, the KV cache is reloaded back to GPU to prevent frequent cache transfer overheads in autoregressive decoding. This is detailed in Algorithm [1.](#page-4-1)

![](_page_5_Picture_3.jpeg)

Figure 4: Dynamic KV Cache Transfer Between GPU and CPU in Prefill and Decode Stages.

## <span id="page-5-0"></span>**3.3 Analysis: Memory Efficiency of MOM**

**Llama MLP Layer** The Llama MLP layer utilizes a SwiGLU (Swish-Gated Linear Unit) architecture, enhancing efficiency and expressivity compared to standard Transformer feedforward networks [\(Shazeer,](#page-12-10) [2020;](#page-12-10) [Touvron et al.,](#page-12-8) [2023\)](#page-12-8). It employs three key matrices: *W*gate (gating), *W*up (expansion), and *W*down (compression). Input *X* is first projected through *W*gate and *W*up; the gating projection uses a Swish activation Swish(*XW*gate), adaptively modulating feature importance [\(Ramachandran et al.,](#page-11-11) [2017\)](#page-11-11). Its output is then multiplied element-wise with the expanded features from *W*up, which increases hidden dimension from *d* to 4*d*. Finally, *W*down compresses features back to dimension *d*. This SwiGLU design improves information flow and parameter efficiency over traditional GELU-based Transformer MLPs [\(Hendrycks & Gimpel,](#page-10-14) [2016\)](#page-10-14).

**Standard Transformers Without Optimization** Let *X* ∈ **R***S*×*<sup>d</sup>* be the input sequence of length *S*, and hidden dimension *d*, number of transformer block layers *L*. In a standard (full-sequence) forward pass, the peak intermediate activation memory required for MLP blocks is *A*full. The memory used during inference consists of model weights *W*model, the KV cache of size 2 · *S* · *d* · *L*, and the intermediate computation results of each layer:

- For the attention mechanism, the theoretical peak intermediate memory is *S* · *S*, but optimized attention mechanisms such as FlashAttention [\(Dao et al.,](#page-10-1) [2022;](#page-10-1) [Dao,](#page-10-2) [2023\)](#page-10-2) and Memory Efficient Attention [\(Rabe & Staats,](#page-11-12) [2021\)](#page-11-12) significantly reduce this. The peak memory usage is instead determined by the output size, which is *S* · *d*.
- In the MLP layers, intermediate tensors *I*up, *I*gate ∈ **R***S*×*<sup>I</sup>* are generated, where *I* ≈ 4*d*. Memory usage peaks at the Up-Projection hidden layer output, size *S* · *I*.
- During inference, the LM head generates only one token at a time, requiring intermediate memory equal to the vocabulary size *V*.

Since intermediate memory does not persist throughout inference, the peak intermediate memory consumption is the maximum of these components. In models like Llama 3, *I* is typically much larger than *d*, so this is dominated by the MLP layer:

$$\mathcal{M}_{\text{intermediate}} = \max(S \cdot d, S \cdot I, V) = S \cdot I \tag{1}$$

Thus, the total peak memory consumption for inference is:

<span id="page-5-1"></span>
$$\mathcal{M}_{\text{total}} = W_{\text{model}} + \mathcal{M}_{\text{KV}} + \mathcal{M}_{\text{intermediate}} = W_{\text{model}} + 2 \cdot S \cdot d \cdot L + S \cdot I \tag{2}$$

**Mini-sequence Partitioning.** To reduce intermediate memory, X gets split into M mini-sequences, each of length  $N \approx \frac{S}{M}$ . Processing each mini-sequence independently lowers the peak intermediate memory to approximately

<span id="page-6-1"></span>
$$\mathcal{M}_{\text{intermediate\_mini}} \approx \frac{\mathcal{M}_{\text{intermediate\_full}}}{M} = \frac{S \cdot I}{M}$$
 (3)

Assuming intermediate buffers are freed between mini-sequences. In practice, the memory required for each mini-sequence will be less than  $\mathcal{M}_{intermediate\_full}$  but more than  $\mathcal{M}_{intermediate\_mini}$  due to overlapping buffers and computational overhead.

**Offloading** During inference, key/value (KV) caches and other data can be offloaded to CPU/disk. Let  $W_{\rm model}$  be the model weights in GPU memory, and  $O_{\rm offload}$  the overhead for data transfers and buffers. Then, out of total GPU memory  $M_{\rm max}$ , the effective memory available for MLP and LM head during prefill stage is

$$\mathcal{M}_{\text{avail}} = \mathcal{M}_{\text{max}} - W_{\text{model}} - O_{\text{offload}}.$$
 (4)

**Maximum Sequence Length** Define  $S_{\text{max}}$  as the maximum sequence length fitting into GPU memory. As Mini-sequence reduces peak intermediate to  $\mathcal{M}_{\text{intermediate.mini}}$ , we get

<span id="page-6-0"></span>
$$S_{\text{max}} \propto \frac{\mathcal{M}_{\text{avail}}}{\mathcal{M}_{\text{intermediate\_mini}}} = \frac{\mathcal{M}_{\text{max}} - W_{\text{model}} - O_{\text{offload}}}{\mathcal{M}_{\text{intermediate\_mini}}}.$$
 (5)

As M grows,  $\mathcal{M}_{\text{intermediate\_mini}}$  decreases, allowing for larger  $S_{\text{max}}$ . Equation (5) shows that even with non-trivial offloading overhead, Mini-sequence reduces the intermediate memory per sequence sufficiently to handle much larger lengths without exhausting GPU resources. Hence, by lowering intermediate demands (via Mini-sequence) and storing much of the KV cache off-GPU (via offloading), we can substantially extend  $S_{\text{max}}$  under a given memory budget. Therefore, MOM can process longer sequences without exceeding GPU limits, effectively removes the prefill memory as the primary memory constraint and shifts the new peak memory bottleneck to the decode stage, dominated by the GPU-resident KV cache.

#### 4 Experiments

We evaluate MOM on Llama 3.2 (Meta AI, 2024), a state-of-the-art large language model designed for high-quality text generation. We use the 8B size version with bfloat16 datatype on single A100 80G GPU. In the Appendix D and E, we expand our tests to include other models. The evaluation examines the combination of Mini-sequence inference and offloading, comparing it with alternative techniques such as chunked prefill. It covers input context lengths of [48000, 80000, 112000, 144000] tokens to ensure the results remain consistent.

#### 4.1 GPU Memory for Analysis

We evaluated the peak VRAM usage of various models under different configurations, including with and without Mini-sequence and with and without offloading, and plotted the results across different context lengths.

![](_page_7_Figure_1.jpeg)

<span id="page-7-0"></span>Figure 5: VRAM Comparison for Mini-sequence Inference and Offloads.

The results in Figure [5](#page-7-0) align with the calculations in Equation [\(2\)](#page-5-1) and Equation [\(3\)](#page-6-1), that memory use increases linearly over context length both with or without Mini-sequence.

Mini-sequence inference has a significant impact on memory savings. And when minisequence inference is applied, offloading further reduces VRAM usage. However, without it, offloading alone does not lead to a substantial reduction in VRAM consumption.

As context length increases, the proportion of intermediate memory in total memory grows, since the model weight size remains unchanged. Consequently, we observe a higher percentage of memory savings with mini-sequence inference as total memory usage increases.

<span id="page-7-1"></span>Table 1: Memory Usage WITH Mini-sequence Divided by NO Mini-sequence on Different Offloading Schemes (%, lower is more memory efficient)

| Context Length (#Tokens): | 48000  | 80000  | 112000 | 144000 |
|---------------------------|--------|--------|--------|--------|
| No offload                | 71.682 | 65.333 | 61.643 | 59.235 |
| (prefill only) Offload    | 64.834 | 56.804 | 52.146 | 49.065 |

Table [1](#page-7-1) also indicates that the percentage of memory savings from enabling Mini-sequence is higher when combined with offloading. This is because offloading primarily reduces KV cache or weight size, making intermediate memory—which Mini-sequence optimizes—a larger proportion of the total memory.

#### **4.2 Maximium Input Context Length Extension for Different Methods**

We tested the maximum context length that fits into an A100-80GB GPU using different methods before encountering an Out of Memory (OOM) error. Overall, MOM outperforms all other methods, expanding the maximum context length from 155,000 tokens in the unoptimized standard model to 455,000 tokens—nearly a threefold improvement as shown in Figure [6.](#page-8-0)

![](_page_8_Figure_1.jpeg)

Figure 6: Maximium Number of Context Tokens Extended from Standard Llama3.2.

## <span id="page-8-1"></span>**4.3 Inference Speed Comparison**

<span id="page-8-0"></span>Table 2: Total Inference Latency (s, lower is faster)

| Context Length (#Tokens):     | 48000  | 80000  | 112000 | 144000 |
|-------------------------------|--------|--------|--------|--------|
| Standard                      | 13.971 | 23.837 | 36.592 | 52.274 |
| (prefill only) Offload        | 14.693 | 25.542 | 38.538 | 56.559 |
| Mini-sequence                 | 13.536 | 23.556 | 36.160 | 51.249 |
| MOM (Mini-sequence + offload) | 14.520 | 25.020 | 38.180 | 53.417 |
| Chunked Prefill size=8192     | 14.057 | 24.515 | 37.666 | 53.247 |

All the methods discussed in this section, including mini-sequence inference, offloading and chunked prefill (with 8192 chunck size), have minimal impact on speed. In Table [2,](#page-8-1) we tested them across different context lengths, measuring speed by forcing the model to generate a fixed output of 200 tokens at a time and recording the total runtime for both the prefill and decoding stages. A more detailed breakdown of prefill and decoding rate could be found in appendix [C.](#page-13-3)

#### **4.4 Memory Speed Trade-off**

To evaluate how each optimization technique balances memory usage and speed, we measure their average memory consumption (as a percentage of the unoptimized Standard model) and throughput across multiple trials with context lengths. These results are then plotted on a scatter graph. Methods positioned closer to the bottom-right corner are generally more optimized, indicating greater memory savings with higher inference throughput. Notably, Figure [2](#page-2-0) shows that MOM appears closer to the lower-right corner, suggesting that it achieves better memory efficiency with minimal trade-offs in speed.

## **4.5 Accuracy**

**Logit Equivalence Test** To validate that MOM has no effect on accuracy, we first tested random input sequences on both MOM and the standard model, comparing the output logits, which were identical.

**Needle Test** We evaluated the model's ability to retrieve a specific detail ("Mary's favorite number is 43251") embedded within a long, unrelated text at varying depths (*needle depth*). Accuracy was binary (100 if correct, 0 otherwise). As shown in Figure [8,](#page-9-2) the standard model failed when *needle depth* × *context length* > 150000 due to GPU memory constraints causing text truncation. In contrast, MOM (Figure [7\)](#page-9-3) handled extended contexts without

truncation. Occasional incorrect responses appeared similarly in both models, indicating no accuracy degradation from MOM.

![](_page_9_Figure_2.jpeg)

<span id="page-9-3"></span>Figure 7: Needle Test Accuracy Scores for MOM

<span id="page-9-2"></span>Figure 8: Needle Test Accuracy Scores for Standard Llama3.2-8B

#### **5 Future Works**

**Optimizing Integration with Other Inference Frameworks** Beyond Hugging Face, large language model inference for individuals and small businesses is often performed using frameworks like vLLM (Kwon et al., 2023b) or sglang (Sgl-Project, 2025). While the MOM mechanism is compatible with these frameworks, not all inference processes may be fully optimized or seamlessly integrated. A deeper investigation into their inference mechanisms is needed to ensure optimal performance and compatibility across different implementations.

**Optimizing KV Cache During Inference** Our method has significantly optimized memory usage during the prefill stage , bringing it close to optimal (Figure 1). Memory consumption is now dominated by the KV cache during decoding stage, presenting an opportunity for further improvement. Future research on KV cache compression techniques for the decoding stage could complement our method, allowing for even greater memory efficiency.

## Acknowledgment

We thank Caltech CS165 support. A. Anandkumar is supported by the Bren named chair professorship, Schmidt AI 2050 senior fellowship, ONR (MURI grant N00014-18-12624).

#### **Ethics Statement**

Our Memory-efficient Offloaded Mini-Sequence Transformer (MOM) addresses GPU memory efficiency and computational performance for inference tasks. While MOM itself does not inherently introduce ethical concerns, the increased accessibility and efficiency of large language models enabled by our approach could amplify societal impacts, including existing biases present in the underlying datasets. We encourage practitioners adopting MOM to follow responsible AI practices, such as bias monitoring, fairness evaluations, transparency, and privacy preservation, particularly when deploying models in sensitive contexts. All experimental procedures in this work adhere strictly to ethical standards, without involving human subjects or private data.

#### References

<span id="page-9-0"></span>Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming {Throughput-Latency} tradeoff in {LLM} inference with {Sarathi-Serve}. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pp. 117–134, 2024.

<span id="page-9-1"></span>Anshul Agrawal, Akshat Panwar, Janarthanan Mohan, Nipun Kwatra, Bhargav S. Gulavani, and Ramachandran Ramjee. Sarathi: Efficient llm inference by piggybacking decodes

- with chunked prefills. *arXiv preprint arXiv:2308.16369*, 2023. URL [https://arxiv.org/](https://arxiv.org/abs/2308.16369) [abs/2308.16369](https://arxiv.org/abs/2308.16369).
- <span id="page-10-9"></span>Mistral AI and NVIDIA. Mistral nemo: A state-of-the-art 12b model, 2024. URL [https:](https://mistral.ai/news/mistral-nemo) [//mistral.ai/news/mistral-nemo](https://mistral.ai/news/mistral-nemo). Accessed: 2025-03-25.
- <span id="page-10-6"></span>Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebron, ´ and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. *arXiv preprint arXiv:2305.13245*, 2023.
- <span id="page-10-8"></span>Alibaba. Qwen2.5 language models, 2024. URL [https://huggingface.co/collections/](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) [Qwen/qwen25-66e81a666513e518adb90d9e](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e). Accessed: 2025-03-25.
- <span id="page-10-12"></span>Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. *arXiv preprint arXiv:2309.16609*, 2023.
- <span id="page-10-0"></span>Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. *arXiv preprint*, arXiv:2004.05150, 2020.
- <span id="page-10-5"></span>Zhenni Bi, Kai Han, Chuanjian Liu, Yehui Tang, and Yunhe Wang. Forest-of-thought: Scaling test-time compute for enhancing llm reasoning. *arXiv preprint arXiv:2412.09078*, 2024.
- <span id="page-10-4"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.
- <span id="page-10-7"></span>Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. *arXiv preprint arXiv:1604.06174*, 2016.
- <span id="page-10-2"></span>Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. *arXiv preprint arXiv:2307.08691*, 2023.
- <span id="page-10-1"></span>Tri Dao, Daniel Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast ´ and memory-efficient exact attention with io-awareness. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 35, pp. 16344–16359, 2022.
- <span id="page-10-15"></span>Tim Dettmers. bitsandbytes: A lightweight cuda-based library for 8-bit and 4-bit quantization in pytorch, 2022. URL [https://github.com/bitsandbytes-foundation/](https://github.com/bitsandbytes-foundation/bitsandbytes) [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes). Accessed: 2025-03-25.
- <span id="page-10-11"></span>Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.
- <span id="page-10-14"></span>Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). *arXiv preprint arXiv:1606.08415*, 2016.
- <span id="page-10-3"></span>Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR)*, volume 1, pp. 3, 2022. URL <https://openreview.net/forum?id=Ziq3BhMu3w>. Available at OpenReview.
- <span id="page-10-10"></span>Shashank Mohan Jain. Introduction to transformers for nlp. *With the Hugging Face Library and Models to Solve Problems*, 2022.
- <span id="page-10-13"></span>Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut ´ Lavril, Thomas Wang, Timothee Lacroix, and William El Sayed. Mistral 7b. ´ *arXiv preprint arXiv:2310.06825*, 2023. URL <https://arxiv.org/abs/2310.06825>.

- <span id="page-11-7"></span>Rakshit Kalra. Memory management for modern llms: Fitting elephants into shoeboxes. *Medium*, 2023. [https://medium.com/@kalra.rakshit/](https://medium.com/@kalra.rakshit/memory-management-for-modern-llms-fitting-elephants-into-shoeboxes-d48f4e85bc9e) [memory-management-for-modern-llms-fitting-elephants-into-shoeboxes-d48f4e85bc9e](https://medium.com/@kalra.rakshit/memory-management-for-modern-llms-fitting-elephants-into-shoeboxes-d48f4e85bc9e).
- <span id="page-11-2"></span>Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. *arXiv preprint arXiv:2001.04451*, 2020.
- <span id="page-11-6"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. *arXiv preprint arXiv:2309.06180*, 2023a.
- <span id="page-11-14"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. vllm: A high-throughput and memory-efficient inference and serving library for large language models, 2023b. URL <https://docs.vllm.ai/en/latest/>.
- <span id="page-11-8"></span>Pierre Lienhart. Llm inference series: 4. kv caching, a deeper look, 2024. URL [https://medium.com/@plienhar/](https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8) [llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8](https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8). Accessed: Mar. 20, 2025.
- <span id="page-11-5"></span>Cheng Luo, Zefan Cai, Hanshi Sun, Jinqi Xiao, Bo Yuan, Wen Xiao, Junjie Hu, Jiawei Zhao, Beidi Chen, and Anima Anandkumar. Headinfer: Memory-efficient llm inference by head-wise offloading. *arXiv preprint arXiv:2502.12574*, 2025.
- <span id="page-11-4"></span>Cheng luo et al. Mini-sequence transformer: Optimizing intermediate memory for long sequences training. *Conference on Neural Information Processing Systems (NeurIPS)*, 37:1–12, 2024.
- <span id="page-11-13"></span>Meta AI. Llama 3.2: Revolutionizing edge ai and vision with open models, 2024. URL <https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/>.
- <span id="page-11-9"></span>NVIDIA. Streamlining ai inference performance and deployment with nvidia tensorrt-llm chunked prefill, 2024. URL [https://developer.nvidia.com/blog/](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/) [streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/). Accessed: Mar. 15, 2025.
- <span id="page-11-12"></span>Markus N. Rabe and Charles Staats. Self-attention does not need *o*(*n* 2 ) memory. *arXiv:2112.05682*, 2021. URL <https://arxiv.org/abs/2112.05682>.
- <span id="page-11-0"></span>Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*, pp. 1–16. IEEE, November 2020.
- <span id="page-11-11"></span>Prajit Ramachandran, Barret Zoph, and Quoc V Le. Searching for activation functions. *arXiv preprint arXiv:1710.05941*, 2017.
- <span id="page-11-1"></span>Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pp. 3505–3506. ACM, August 2020. doi: 10.1145/3394486.3406703.
- <span id="page-11-10"></span>Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Leonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram ´ e, Johan Ferret, ´ et al. Gemma 2: Improving open language models at a practical size. *arXiv preprint arXiv:2408.00118*, 2024.
- <span id="page-11-15"></span>Sgl-Project. sglang: A high-performance inference framework for large language models, 2025. URL <https://github.com/sglang-project/sglang>. Accessed: 2025-03-23.
- <span id="page-11-3"></span>Noam Shazeer. Fast transformer decoding: One write-head is all you need. *arXiv preprint arXiv:1911.02150*, 2019.

<span id="page-12-10"></span>Noam Shazeer. Glu variants improve transformer. *arXiv preprint arXiv:2002.05202*, 2020.

<span id="page-12-2"></span>Mostofa Shoeybi, Mostofa Ali Patwary, Rajbhandari Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. *arXiv preprint*, arXiv:1909.08053, 2019.

<span id="page-12-3"></span>Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*, 2024.

<span id="page-12-1"></span>Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Hongyu Fei, and Donald Metzler. Long range arena: A benchmark for efficient transformers. *arXiv preprint*, arXiv:2011.04006, 2020. URL [https://arxiv.org/abs/2011.](https://arxiv.org/abs/2011.04006) [04006](https://arxiv.org/abs/2011.04006).

<span id="page-12-8"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.

<span id="page-12-0"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 2017.

<span id="page-12-6"></span>Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Selfattention with linear complexity. *arXiv preprint arXiv:2006.04768*, 2020.

<span id="page-12-4"></span>Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824–24837, 2022.

<span id="page-12-5"></span>Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. *Advances in neural information processing systems*, 36:11809–11822, 2023.

<span id="page-12-7"></span>Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. *arXiv preprint arXiv:1904.00962*, 2019.

