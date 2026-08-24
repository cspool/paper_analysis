# **B** Additional Experiment

To comprehensively evaluate the memory consumption of StarTrail and Ring Attention, we compared the maximum supported sequence lengths of StarTrail with those reported in the Ring Attention paper [24]. As shown in Table 2, although StarTrail requires slightly more memory, it still supports sequence lengths commonly used in training tasks.

#### **B.1** Discussion

#### **B.2** Larger Batch Sizes and Models

We utilized small batches and model sizes in our experiments because these choices do not affect the underlying improvements in communication efficiency and computation-to-communication ratio that StarTrail provides. For larger batch sizes, both communication and computation scale proportionally, leaving the overlapping ability unchanged. Similarly, while larger models involve more layers or

larger hidden sizes, the key attention computations and corresponding ratios remain unaffected. Hence, our conclusions naturally extend to scenarios with larger batches and models.

### B.2.1 FlashAttention3 and Hopper GPUs

In addition to the original FlashAttention [\[8\]](#page-10-11) used in our experiments, FlashAttention3 [\[35\]](#page-12-8) has been introduced, specifically designed for Hopper and newer Nvidia GPUs. For FP16 precision, which is utilized in this paper, FlashAttention3 achieves a 1.5-2.0x speedup on Hopper GPUs. As discussed in Section [3,](#page-3-0) reducing attention computation overhead results in more P2P communication not being overlapped, further emphasizing the need to reduce communication volume. With the increasing adoption of Hopper GPUs, the significance of the StarTrail system will also grow.

### B.2.2 StarTrail and Other Parallelisms

Model Parallelism As is well known, tensor parallelism shards activations by attention heads during attention computation, making it easily combinable with StarTrail with minimal effort. However, when combined with tensor parallelism, the need for attention heads can limit the scalability of head-based sequence parallel methods like DeepSpeed-Ulysses. Pipeline parallelism, on the other hand, divides the model across layers without altering the computation patterns within Transformer blocks, making StarTrail orthogonal to it.

Other Sequence Parallelism StarTrail is orthogonal with other attention-head-sharding-based sequence parallelism approaches, such as DeepSpeed-Ulysses [\[14\]](#page-10-9). While DeepSpeed-Ulysses distributes attention heads across different devices, StarTrail can independently partition activations along the sequence length dimension. In future work, we can explore combining StarTrail with DeepSpeed-Ulysses to expand the communication scheduling space, harnessing the scalability of StarTrail alongside the efficiency of DeepSpeed-Ulysses.

In summary, StarTrail can be seamlessly integrated with other parallel training techniques, enabling the creation of a hybrid distributed training system.

### B.3 Other Related Works

Attention Optimization. Traditional full attention mechanisms necessitate O(n 2 ) memory for storing the outputs of QK<sup>T</sup> , leading to significant computational and memory demands. To address these challenges within the GPU, several approaches have been devised to reduce both memory and computational requirements. Memory-efficient attention[\[32\]](#page-11-13) introduces a straightforward algorithm that requires only O(1) memory relative to the sequence length, with an extension for self-attention that needs only O(log n) memory. FlashAttention further minimizes I/O overhead and enhances overall efficiency. Additionally, optimization methods specifically tailored for inference, such as PagedAttention[\[19\]](#page-11-14), are also being developed to improve the efficiency of attention computations. In this work, we utilize FlashAttention within each iteration to reduce the computation overhead.

Long-Sequence Training Techniques. Sequence Parallelism[\[21\]](#page-11-15) was initially introduced to enhance the efficiency of parallel long-sequence training. Ring Attention[\[24\]](#page-11-5) improved communication efficiency through memory-efficient methods[\[32\]](#page-11-13), supporting near-infinite sequence lengths. DeepSpeed-Ulysses[\[14\]](#page-10-9) employs attention head splitting to achieve high efficiency, though it is constrained by the number of heads. Megatron Sequence Parallelism focuses on reducing memory costs during Tensor Parallelism, while DistFlashAttention[\[20\]](#page-11-6) features a load-balance scheme and a novel gradient checkpoint method. Our work builds on these innovations, introducing a system that supports large-scale training with an efficient communication scheme.

Techniques for Distributed Model Training. Distributed model training encompasses two primary areas: 1) Memory Management: Various techniques aim to conserve GPU memory during distributed training, such as mixed precision training[\[26\]](#page-11-16) and the ZeRO series[\[34\]](#page-12-6). In this work, we implement ZeRO-2 to manage optimizer states and gradients efficiently. 2) Hybrid Parallelism: Frameworks like Megatron[\[27\]](#page-11-17) and Colossal AI[\[4\]](#page-10-13) integrate multiple forms of parallelism. There are various existing Parallelism techniques like Pipeline Parallelism[\[13,](#page-10-6) [10,](#page-10-7) [23,](#page-11-3) [25\]](#page-11-4) and Tensor Parallelism[\[37\]](#page-12-1), which can be combined with StarTrail Parallelism to facilitate large-scale training. We are also considering the integration of additional frameworks such as [\[6\]](#page-10-14) to enhance overlapping capabilities in future implementations.