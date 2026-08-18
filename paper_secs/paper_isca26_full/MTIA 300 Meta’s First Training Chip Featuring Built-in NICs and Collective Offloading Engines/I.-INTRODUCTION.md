# I. INTRODUCTION

AI workloads continue to grow rapidly across the industry. At Meta, products like Facebook and Instagram rely on Deep Learning Recommendation Models (DLRMs) [26] to deliver personalized content, including ads, short videos, and friend posts. This growth has driven the in-house development of Meta's AI chips. We previously introduced MTIA 1 [10] and MTIA 2i [7] (a.k.a. MTIA 100 and 200), both optimized for DLRM inference, with MTIA 2i now deployed at a scale of hundreds of thousands of chips. This paper presents the next step in that evolution: Meta's first training chip, MTIA 300.

Meta supports two major AI workloads: GenAI and DLRMs. DLRM training differs from GenAI training: it requires fewer FLOPS but larger HBM capacity, higher network bandwidth, and more frequent communications. This combination often leads to low accelerator utilization. Unlike generalpurpose GPUs, MTIA 300 is optimized for DLRMs, featuring a design that intentionally diverges from GPU architectures to reflect this focus.

To motivate MTIA 300's design for DLRMs, we first summarize the workload characteristics. DLRMs often apply multi-layer perceptrons (MLPs) to dense features (e.g., user age) and embedding tables to sparse categorical features (e.g., post IDs), connected via a dense interaction layer. Dense components require high FLOPS (though far less than GenAI), while sparse components require irregularly accessing ultra-large embedding tables and are often memory-bound or instruction-bound. Because embedding tables often exceed a single accelerator's memory capacity (sometimes representing over 99% of model parameters), DLRMs employ hybrid parallelism in training: dense layers use data parallelism, and sparse layers shard embedding tables table-wise or row-wise [14]. This approach enables the training of large models without sacrificing compute efficiency in dense layers.

This mix of data and model parallelism for different model components introduces complex communication patterns. With data parallelism, each accelerator receives a local batch and performs an AllReduce to synchronize gradients. Model parallelism adds AllToAllv collectives to exchange features and redistribute results for forward and backward passes. Moreover, many DLRMs use the distributed Shampoo optimizer [12], [32] for dense components, which adds an AllGather during the optimization phase. Efficient execution of these collectives is necessary for achieving high performance.

As DLRMs stress the communication data path, MTIA 300 incorporates the following features to address this challenge:

- *Built-in NIC chiplets*: MTIA 300 adopts a chiplet architecture, embedding two NIC chiplets with a total of 12 highly optimized 800Gbps RDMA NICs. Built-in NICs avoid PCIe overhead between the accelerator and NICs, and the NICs can be used flexibly for scale-up or scale-out networks.
- *Collective offloading*: In GPUs, compute engines and the host CPU handle collective operations, which is often inefficient. MTIA 300, in contrast, utilizes dedicated message engines that deliver the same communication throughput for these operations as compute engines while using only one-third of the chip area.
- *Near memory compute*: Compute and message engines share the network-on-chip for memory and I/O access. To avoid congestion from high-bandwidth collectives, message engines are placed at the chip edges, next to HBM, cache, and I/O. The message engine's near-memory-compute logic block delivers high throughput for all reduction-based collectives, including Reduce, AllReduce, and ReduceScatter.

<sup>∗</sup> The full list of authors is in the appendix. Corresponding author: Chunqiang Tang, tang@meta.com.

In addition to outlining MTIA 300's unique hardware features and overall architecture, we describe the software stack, highlighting the collectives library, which provides a familiar interface while effectively leveraging MTIA's specialized message engines and built-in NICs for efficient communication.

Contributions. While many GPUs and AI ASICs have been reported [1], [3], [6], [11], [15], [17], [18], [22]–[24], [30], [33], our experience reveals unique requirements for DL-RMs that prior work does not address. To our knowledge, MTIA 300 is the first accelerator with built-in NIC chiplets and general-purpose collective offloading engines, avoiding the inefficiencies of using compute engines for collectives and enabling flexible networking for both scale-up and scale-out. In comparison, although the TPU's sparse core [16] can also offload remote access to embedding tables, it is specialized for a non-RDMA, non-switched torus network and lacks a general collective library interface, limiting its applicability to other industry accelerators, which are typically built around RDMA and similar collective library interfaces.

Although MTIA 300's distinguishing features—built-in NIC chiplets, collective offloading, and near-memory compute—were originally motivated by DLRM training, these design principles remain broadly applicable and have been adopted in subsequent MTIA generations optimized for GenAI models. While the development of MTIA 300 began several years ago to compete with H100 and H200 GPUs, its successor, MTIA 400 [34], was designed to rival GB300 GPUs. Furthermore, the upcoming MTIA 450 and 500 [34] target industry-leading GenAI inference performance against future GPUs. In this paper, we focus on MTIA 300, leaving the technical details of later generations for future publications.

The rest of the paper is organized as follows. Section II provides an overview of MTIA 300's chip architecture, focusing on its message engine for collective offloading. Section III details the rack and network architecture. Section IV describes the software stack, emphasizing the collective library that leverages MTIA 300's offloading capabilities. Section V evaluates MTIA 300's performance. Section VI discusses its challenges and limitations. Finally, Section VII reviews related work, and Section VIII concludes the paper.

#### II. MTIA 300 ARCHITECTURE

Figure 1 shows an overview of MTIA 300, which adopts a chiplet architecture. To highlight its distinguishing features, we first compare MTIA 300 with MTIA-2i and then with GPUs.

## A. Comparing MTIA 300 with MTIA-2i and GPUs

Table I compares the specifications of MTIA 300 and MTIA-2i [7]. With MTIA-2i designed for inference and MTIA 300 for training, MTIA 300 introduces several changes: ≈3x larger area, ≈10x higher Thermal Design Power (TDP), liquid cooling (versus air cooling), HBM3E (versus LPDDR), FP8 compute (versus INT8), >3x BF16 FLOPS, a 2.5D CoWoS package, a reticle-sized compute die, network chiplets supporting RoCE, and message engines to offload collective communication. Notably, the SIMD compute is increased

![](_page_1_Figure_8.jpeg)

Fig. 1: MTIA 300 with chiplets for compute, network, and HBM.

to >6x for FP32, resulting in a 16:1 GEMM:SIMD ratio compared to 32:1 in MTIA-2i; this increase is substantial given the rising demand and diversity of non-GEMM compute in training (e.g., table-batched embedding forward/backward, and optimizers).

Unlike general-purpose GPUs, MTIA 300 is optimized for DLRMs. Table II highlights the features that distinguish it from H100 GPUs. Notably, MTIA 300 deemphasizes peak FLOPS and emphasizes HBM bandwidth and networking, featuring embedded NIC chiplets and dedicated hardware support for collective communication offloading.

