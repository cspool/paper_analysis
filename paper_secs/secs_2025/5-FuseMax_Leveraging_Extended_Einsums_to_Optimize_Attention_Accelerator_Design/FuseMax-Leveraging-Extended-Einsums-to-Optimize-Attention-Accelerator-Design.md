# FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design

Nandeeka Nayak University of California, Berkeley Berkeley, CA, USA nandeeka@berkeley.edu

Xinrui Wu Tsinghua University Beijing, China xr-wu20@mails.tsinghua.edu.cn Toluwanimi O. Odemuyiwa University of California, Davis Davis, CA, USA todemuviwa@ucdavis.edu

Michael Pellauer NVIDIA Westford, MA, USA mpellaer@nvidia.com

Joel S. Emer Massachusetts Institute of Technology / NVIDIA Cambridge, MA, USA jsemer@mit.edu

Christopher W. Fletcher University of California, Berkeley Berkeley, CA, USA cwfletcher@berkeley.edu

Abstract-Attention for transformers is a critical workload that has recently received significant 'attention' as a target for custom acceleration. Yet, while prior work succeeds in reducing attention's memory-bandwidth requirements, it creates load imbalance between operators that comprise the attention computation (resulting in severe compute under-utilization) and requires on-chip memory that scales with sequence length (which is expected to grow over time).

This paper ameliorates these issues, enabling attention with nearly 100% compute utilization, no off-chip memory traffic bottlenecks, and on-chip buffer size requirements that are independent of sequence length. The main conceptual contribution is to use a recently proposed abstraction—the cascade of Einsums to describe, formalize, and taxonomize the space of attention algorithms that appear in the literature. In particular, we show how Einsum cascades can be used to infer non-trivial lower bounds on the number of passes a kernel must take through its input data, which has implications for either required on-chip buffer capacity or memory traffic. We show how this notion can be used to meaningfully divide the space of attention algorithms into several categories and use these categories to inform our design process.

Based on the above characterization, we propose FuseMax—a novel mapping and binding of attention onto a spatial array-style architecture. On attention, in an iso-area comparison, FuseMax achieves an average  $6.7 \times$  speedup over the prior state-of-the-art, FLAT, while using 79% of the energy. Similarly, on full end-toend transformer inference, FuseMax achieves an average  $5.3 \times$ speedup over FLAT using 83% of the energy.

Index Terms-Tensor algebra, Extended Einsums, Spatial architectures, Attention

### I. Introduction

Over the past few years, transformers [52] have emerged as the model architecture of choice for a wide range of machine learning applications, from natural language processing [13], [17], [48], [49] to computer vision [18], [33] to speech recognition [4], [26]. This rise has been accompanied by a corresponding wave of proposals for accelerating transformers in both software [12], [14], [15] and hardware [28], [62].

This work was partially funded by NSF grants CNS-1954521, CNS-1942888, CNS-2154183, CCF-8191902, and CCF-2217099; as well as by an Intel gift and a Microsoft Research PhD fellowship.

Fortunately, many of the layers (projections, fully connected layers, etc.) used by transformers look very similar to prior generations of machine learning models. Their resourceintensive tensor products can be described and evaluated with existing tensor algebra accelerator modeling tools [29], [35], [41], and many of the other layers (e.g., layer normalization) have negligible impact on performance and can be safely ignored.

However, the attention layer [52]—usually described as a matrix multiplication, a softmax, and then another matrix multiplication—does not fit into either of these boxes. For example, the softmax is both memory intensive (featuring low algorithmic reuse) and compute intensive (featuring exponentiation and division). Furthermore, attention's characteristics preclude many "free lunches" often used to improve efficiency in other DNN models. For example, because all tensors are a function of the model inputs, there is no opportunity to amortize memory access costs with an increased batch size. Additionally, since none of the operands can be computed before the inputs are given, compression/strength reduction techniques (e.g., quantization [22], [60], sparsity [34], [46], [53], etc.) must be applied dynamically, leading to more complicated algorithms and hardware designs.

To illustrate the difficulty in accelerating attention, consider the state-of-the-art accelerator for attention: FLAT [28]. FLAT uses fusion to reduce attention memory bandwidth bottlenecks on a spatial architecture (e.g., a TPU [27]). Specifically, FLAT maps attention's matrix multiplications to the 2D spatial array and softmax operations to a separate 1D array. While FLAT's design does make attention compute bound, it becomes compute bottlenecked in the 1D array (the softmax), causing severe under utilization of the 2D array. While one could add additional PEs to the 1D array, this results in corresponding area costs.

Making matters worse, FLAT requires that the entire vector over which the softmax is performed be buffered on chip. This vector is proportional to the sequence length, which is growing rapidly with time (e.g., Google reports 10 million length sequences in research, which would require 100s of MegaBytes to buffer [44]). When the vector/sequence length grows beyond allowable buffer capacity, FLAT is forced to spill, which contributes significantly to attention energy consumption and can even make attention memory-bandwidth bound.

This paper. We address the above challenges by proposing a novel spatial architecture – *FuseMax* – to accelerate attention, with particular emphasis on removing bottlenecks imposed by the softmax. Our architecture addresses all of the aforementioned issues associated with FLAT. Namely:

- FuseMax is compute bound, but provides almost 100% utilization of both the 2D and 1D arrays throughout the attention layer, without adding additional PEs to the 1D array.
- FuseMax's on-chip memory requirements are invariant to sequence length and require no extra spills to memory regardless of sequence length.

The paper's technical core is split into three parts.

First, Section III demonstrates a novel analysis on kernels that uses the recently proposed *cascade of Einsums* abstraction [35]. In a nutshell, an Einsum defines an iteration space over tensors and what computation is done on and between tensors at each point in the iteration space. A cascade of Einsums is a sequence of dependent Einsums that can be used to describe and specify a larger kernel.

While prior work [35], [39] provides a precise definition for Einsums, a major contribution in our work is to show how this definition can be leveraged to inform accelerator design. Specifically, we recognize that the cascade makes explicit *precisely* what dependencies there are between Einsums. We show how this can be used to make non-trivial deductions about a kernel's allowed fusion granularity and algorithmic minimum per-tensor live footprint. The relationship between the live footprint and the buffer capacity, in turn, has implications for the required data movement.

In more detail, this analysis provides insight into the number of *passes* an algorithm performs, i.e., the number of times a given element of an input must be revisited after visiting every other element of the input. Normally, one strives to choose a dataflow that exploits maximal reuse in a given element (or tile of elements) to avoid having frequently reload it. However, some algorithms preclude this strategy. In this work, we describe how to count the number of passes a cascade requires and present two methods for reducing the number of passes. In general, fewer passes is preferable; although, interestingly, we find that decreasing the number of passes can increase the required compute. Given that an Einsum cascade is mapping/scheduling agnostic, this analysis provides insight given any possible scheduling of the cascade onto hardware.

Next, Section IV applies the cascade of Einsums abstraction to describe and formalize the attention kernel. Using the notion of passes introduced in Section III, we taxonomize the space of numerically stable attention proposals that appear in the literature. For example, in a na¨ıve implementation of attention, one must traverse the entire softmax input to build the softmax denominator *and only after that* can one revisit and scale each input (softmax numerator) by the denominator. Because this analysis is performed on the cascade of Einsums, our lower bounds on passes hold for any choice of mapping, including applications of fusion. For example, despite using fusion, FLAT employs a 3-pass cascade and its reliance on large onchip buffering is a symptom of trying to avoid three passesworth of DRAM traffic. We, then, show how transforming the attention cascade reduces the number of passes required.

Additionally, we find that expressing attention as a cascade of Einsums reveals that optimizations that were previously conflated can actually be applied separately. We specifically call out one that is used by 1-pass algorithms to eliminate the need for a second pass after the final softmax denominator has been calculated. We recognize that this optimization has the added benefit of decreasing the required divisions, which is not only useful for but can be applied to 2- and 3-pass cascades as well.

Finally, Section V uses the insights from Section IV as a starting point to develop a novel mapping and binding for attention that can be lowered to a spatial architecture. We call our architecture FuseMax. FuseMax adopts the 1-pass attention cascade used in FlashAttention-2 [14]. However, despite using the cascade from FlashAttention-2, binding this cascade to a spatial architecture is non-trivial. In particular, FlashAttention-2 binds the cascade onto a GPU, an architecture that features homogeneous PEs, each with relatively large per-PE storage, and expensive inter-PE communication. Spatial architectures feature opposite characteristics: heterogeneous PEs, each with smaller per-PE storage, and cheap (but restricted) inter-PE communication. Specifically, the networks that connect the PEs within the 2D array allow efficient communication primarily between neighbors. We overcome these differences and demonstrate a novel mapping and binding for the 1-pass cascade that achieves high utilization for entire transformer layers. Our architecture requires only minimal changes to a standard spatial architecture and is performance/energy robust to long sequence lengths (e.g., 1M tokens and beyond).

To summarize, we make the following contributions:

- We show how cascades of Einsums can be used to inform accelerator design, both in terms of reasoning about compute requirements and per-tensor live footprints. We formalize lower bounds on the number of passes a cascade imposes given any possible mapping of the cascade onto hardware.
- We use cascades of Einsums, and the observation about pass lower bounds, to provide a taxonomy and precise specification of numerically stable attention algorithms in the literature. Orthogonally, we show how previously entangled attention optimizations can be applied across attention algorithms.
- We propose a novel mapping and binding for attention for a spatial architecture—which we call FuseMax—that achieves high utilization for both 2D and 1D array PEs,

and has memory traffic requirements that are independent of sequence length.

We evaluate FuseMax on BERT [17], TrXL [13], T5 [49], and XLM [13] and demonstrate a 6.7× speedup on attention with 79% of the energy and a 5.3× speedup on full end-to-end inference with 83% of the energy relative to FLAT.

