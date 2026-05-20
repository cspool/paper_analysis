2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO) 

**==> picture [43 x 42] intentionally omitted <==**

**==> picture [42 x 42] intentionally omitted <==**

**==> picture [42 x 43] intentionally omitted <==**

# FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design 

Nandeeka Nayak _University of California, Berkeley_ Berkeley, CA, USA nandeeka@berkeley.edu 

Xinrui Wu _Tsinghua University_ Beijing, China xr-wu20@mails.tsinghua.edu.cn 

Toluwanimi O. Odemuyiwa _University of California, Davis_ Davis, CA, USA todemuyiwa@ucdavis.edu 

Michael Pellauer _NVIDIA_ Westford, MA, USA mpellaer@nvidia.com 

Joel S. Emer _Massachusetts Institute of Technology / NVIDIA_ Cambridge, MA, USA jsemer@mit.edu 

Christopher W. Fletcher _University of California, Berkeley_ Berkeley, CA, USA cwfletcher@berkeley.edu 

_**Abstract**_ **—Attention for transformers is a critical workload that has recently received significant ‘attention’ as a target for custom acceleration. Yet, while prior work succeeds in reducing attention’s memory-bandwidth requirements, it creates load imbalance between operators that comprise the attention computation (resulting in severe compute under-utilization) and requires on-chip memory that scales with sequence length (which is expected to grow over time).** 

**This paper ameliorates these issues, enabling attention with nearly 100% compute utilization, no off-chip memory traffic bottlenecks, and on-chip buffer size requirements that are independent of sequence length. The main conceptual contribution is — to use a recently proposed abstraction—the** _**cascade of Einsums**_ **to describe, formalize, and taxonomize the space of attention algorithms that appear in the literature. In particular, we show how Einsum cascades can be used to infer non-trivial lower bounds on the number of** _**passes**_ **a kernel must take through its input data, which has implications for either required on-chip buffer capacity or memory traffic. We show how this notion can be used to meaningfully divide the space of attention algorithms into several categories and use these categories to inform our design process.** 

**Based on the above characterization, we propose FuseMax—a novel mapping and binding of attention onto a spatial array-style architecture. On attention, in an iso-area comparison, FuseMax achieves an average** 6 _._ 7 _×_ **speedup over the prior state-of-the-art, FLAT, while using** 79% **of the energy. Similarly, on full end-toend transformer inference, FuseMax achieves an average** 5 _._ 3 _×_ **speedup over FLAT using** 83% **of the energy.** 

_**Index Terms**_ **—Tensor algebra, Extended Einsums, Spatial architectures, Attention** 

## I. INTRODUCTION 

Over the past few years, transformers [52] have emerged as the model architecture of choice for a wide range of machine learning applications, from natural language processing [13], [17], [48], [49] to computer vision [18], [33] to speech recognition [4], [26]. This rise has been accompanied by a corresponding wave of proposals for accelerating transformers in both software [12], [14], [15] and hardware [28], [62]. 

This work was partially funded by NSF grants CNS-1954521, CNS1942888, CNS-2154183, CCF-8191902, and CCF-2217099; as well as by an Intel gift and a Microsoft Research PhD fellowship. 

Fortunately, many of the layers (projections, fully connected layers, etc.) used by transformers look very similar to prior generations of machine learning models. Their resourceintensive tensor products can be described and evaluated with existing tensor algebra accelerator modeling tools [29], [35], [41], and many of the other layers (e.g., layer normalization) have negligible impact on performance and can be safely ignored. 

However, the attention layer [52]—usually described as a matrix multiplication, a softmax, and then another matrix multiplication—does not fit into either of these boxes. For example, the softmax is both memory intensive (featuring low algorithmic reuse) _and_ compute intensive (featuring exponentiation and division). Furthermore, attention’s characteristics preclude many “free lunches” often used to improve efficiency in other DNN models. For example, because all tensors are a function of the model inputs, there is no opportunity to amortize memory access costs with an increased batch size. Additionally, since none of the operands can be computed before the inputs are given, compression/strength reduction techniques (e.g., quantization [22], [60], sparsity [34], [46], [53], etc.) must be applied dynamically, leading to more complicated algorithms and hardware designs. 

To illustrate the difficulty in accelerating attention, consider the state-of-the-art accelerator for attention: FLAT [28]. FLAT uses fusion to reduce attention memory bandwidth bottlenecks on a spatial architecture (e.g., a TPU [27]). Specifically, FLAT maps attention’s matrix multiplications to the 2D spatial array and softmax operations to a separate 1D array. While FLAT’s design does make attention compute bound, it becomes compute bottlenecked in the 1D array (the softmax), causing severe under utilization of the 2D array. While one could add additional PEs to the 1D array, this results in corresponding area costs. 

Making matters worse, FLAT requires that the entire vector over which the softmax is performed be buffered on chip. This vector is proportional to the sequence length, which is growing rapidly with time (e.g., Google reports 10 million 

979-8-3503-5057-9/24/$31.00 ©2024 IEEE 1458 DOI 10.1109/MICRO61859.2024.00107 

length sequences in research, which would require 100s of MegaBytes to buffer [44]). When the vector/sequence length grows beyond allowable buffer capacity, FLAT is forced to spill, which contributes significantly to attention energy consumption and can even make attention memory-bandwidth bound. 

**This paper.** We address the above challenges by proposing a novel spatial architecture – _FuseMax_ – to accelerate attention, with particular emphasis on removing bottlenecks imposed by the softmax. Our architecture addresses all of the aforementioned issues associated with FLAT. Namely: 

- FuseMax is compute bound, but provides almost 100% utilization of both the 2D and 1D arrays throughout the attention layer, without adding additional PEs to the 1D array. 

- FuseMax’s on-chip memory requirements are invariant to sequence length and require no extra spills to memory regardless of sequence length. 

The paper’s technical core is split into three parts. 

First, Section III demonstrates a novel analysis on kernels that uses the recently proposed _cascade of Einsums_ abstraction [35]. In a nutshell, an Einsum defines an iteration space over tensors and what computation is done on and between tensors at each point in the iteration space. A cascade of Einsums is a sequence of dependent Einsums that can be used to describe and specify a larger kernel. 

While prior work [35], [39] provides a precise definition for Einsums, a major contribution in our work is to show how this definition can be leveraged to inform accelerator design. Specifically, we recognize that the cascade makes explicit _precisely_ what dependencies there are between Einsums. We show how this can be used to make non-trivial deductions about a kernel’s allowed fusion granularity and algorithmic minimum per-tensor live footprint. The relationship between the live footprint and the buffer capacity, in turn, has implications for the required data movement. 

In more detail, this analysis provides insight into the number of _passes_ an algorithm performs, i.e., the number of times a given element of an input must be revisited after visiting every other element of the input. Normally, one strives to choose a dataflow that exploits maximal reuse in a given element (or tile of elements) to avoid having frequently reload it. However, some algorithms preclude this strategy. In this work, we describe how to count the number of passes a cascade requires and present two methods for reducing the number of passes. In general, fewer passes is preferable; although, interestingly, we find that decreasing the number of passes can increase the required compute. Given that an Einsum cascade is mapping/scheduling agnostic, this analysis provides insight given any possible scheduling of the cascade onto hardware. 

Next, Section IV applies the cascade of Einsums abstraction to describe and formalize the attention kernel. Using the notion of passes introduced in Section III, we taxonomize the space of numerically stable attention proposals that appear in the literature. For example, in a na¨ıve implementation of attention, one must traverse the entire softmax input to build the softmax 

denominator _and only after that_ can one revisit and scale each input (softmax numerator) by the denominator. Because this analysis is performed on the cascade of Einsums, our lower bounds on passes hold for any choice of mapping, including applications of fusion. For example, despite using fusion, FLAT employs a 3-pass cascade and its reliance on large onchip buffering is a symptom of trying to avoid three passesworth of DRAM traffic. We, then, show how transforming the attention cascade reduces the number of passes required. 

Additionally, we find that expressing attention as a cascade of Einsums reveals that optimizations that were previously conflated can actually be applied separately. We specifically call out one that is used by 1-pass algorithms to eliminate the need for a second pass after the final softmax denominator has been calculated. We recognize that this optimization has the added benefit of decreasing the required divisions, which is not only useful for but can be applied to 2- and 3-pass cascades as well. 

Finally, Section V uses the insights from Section IV as a starting point to develop a novel mapping and binding for attention that can be lowered to a spatial architecture. We call our architecture FuseMax. FuseMax adopts the 1-pass attention cascade used in FlashAttention-2 [14]. However, despite using the cascade from FlashAttention-2, binding this cascade to a spatial architecture is non-trivial. In particular, FlashAttention-2 binds the cascade onto a GPU, an architecture that features homogeneous PEs, each with relatively large per-PE storage, and expensive inter-PE communication. Spatial architectures feature opposite characteristics: heterogeneous PEs, each with smaller per-PE storage, and cheap (but restricted) inter-PE communication. Specifically, the networks that connect the PEs within the 2D array allow efficient communication primarily between neighbors. We overcome these differences and demonstrate a novel mapping and binding for the 1-pass cascade that achieves high utilization for entire transformer layers. Our architecture requires only minimal changes to a standard spatial architecture and is performance/energy robust to long sequence lengths (e.g., 1M tokens and beyond). 

To summarize, we make the following contributions: 

- We show how cascades of Einsums can be used to inform accelerator design, both in terms of reasoning about compute requirements and per-tensor live footprints. We formalize lower bounds on the number of passes a cascade imposes given any possible mapping of the cascade onto hardware. 

- We use cascades of Einsums, and the observation about pass lower bounds, to provide a taxonomy and precise specification of numerically stable attention algorithms in the literature. Orthogonally, we show how previously entangled attention optimizations can be applied across attention algorithms. 

- We propose a novel mapping and binding for attention for a spatial architecture—which we call FuseMax—that achieves high utilization for both 2D and 1D array PEs, 

1459 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

and has memory traffic requirements that are independent of sequence length. 

- We evaluate FuseMax on BERT [17], TrXL [13], T5 [49], and XLM [13] and demonstrate a 6 _._ 7 _×_ speedup on attention with 79% of the energy and a 5 _._ 3 _×_ speedup on full end-to-end inference with 83% of the energy relative to FLAT. 

## II. BACKGROUND 

In this section, we describe the concepts and terminology used in the remainder of the paper. 

## _A. Tensors_ 

This paper focuses on algebraic computations on tensors, where a tensor is a multidimensional array. A tensor’s _rank_ refers to a specific dimension of the tensor, while the tensor’s _shape_ is the set of valid coordinates for each of the tensor’s ranks. We use the notation _N_ -tensor to denote a tensor with _N_ ranks, where a 0-tensor is a scalar, a 1-tensor is a vector, a 2-tensor is a matrix, etc. 

We adopt the format-agnostic _fibertree_ abstraction of tensors, where a tensor is represented as a tree of fibers, as detailed in prior work [25], [35], [38], [43], [51], [55], [57], [58], using the specific version described in TeAAL [35, Section 2.1]. In this abstraction, a _fiber_ consists of the set of coordinates for a given rank with common coordinates for all higher-level ranks. Each coordinate is coupled with a _payload_ . The payload may contain a reference to a fiber in the next lower rank, or to a leaf data _value_ . 

## _B. Traditional Einsums_ 

An Einsum expression defines a computation on a set of tensor operands using an iteration space that specifies the set of points where the computations are performed [35], [39]. For example, we describe matrix-matrix multiplication (GEMM) with the following Einsum: 

**==> picture [171 x 11] intentionally omitted <==**

where _A_ and _B_ are input 2-tensors of shape _K × M_ and _K×N_ , respectively. _Z_ is an output 2-tensor with shape _M ×N_ . Throughout this paper, we use the same symbol for both the shape and _name_ of a rank (e.g., rank _K_ in _A_ has a shape of _K_ ). 

The _iteration space_ of this Einsum is [0 _, K_ ) _×_ [0 _, M_ ) _×_ [0 _, N_ ). An evaluation of this Einsum must: (1) walk every ( _k, m, n_ ) point in the iteration space; and, at each point (2) project into the _data space_ of all input tensors, (3) multiply the corresponding data values, and (4) place the result at the corresponding data point in _Z_ . If a value already exists at an ( _m, n_ ) point in _Z_ (due to computation at the same ( _m, n_ ) point for a different _k_ in the iteration space), reduce the two values together using addition. Note that the Einsum specifies _what_ to compute; it does not indicate the order in which one walks the iteration space. These aspects are left to the _mapping_ [9], [35], [41]. 

We also note that we can view the iteration space itself as a tensor. In the example above, this tensor has shape _K × M × N_ . Therefore, we define a special fibertree—called the iteration space fibertree or _is-fibertree_ —that is the fibertree representation of this iteration space tensor. 

## _C. Extended Einsums_ 

Traditional Einsums sufficiently express standard tensor algebra, including those supported in Basic Linear Algebra Subprograms (BLAS) [19], [30] and tensor network contractions [1]. However, they cannot handle more complex computations. The recently proposed Extended General Einsums notation (EDGE) [39], extends Einsums to handle graph algorithm computations. We find this abstraction useful for also expressing complex tensor algebra computations and use its notation throughout the paper. We now briefly summarize the portions of EDGE that we leverage. 

_1) User-Defined Computations:_ EDGE separates computations into three “actions”: map ([�] ), reduce ([�] ), and populate (=) [39]. Map specifies the pair-wise computation between the shared ranks of two tensors, reduce specifies the computation for the reduction step of an Einsum, and default populate (=) places a computed value from the right-hand side (RHS) of the Einsum to its location on the left-hand side (LHS). 

Each map and reduce action contains two operations: merge and compute. Compute defines the operation to apply between two data values, and can be _any_ user-defined function. Merge specifies which regions of the iteration space to touch; execution will not need to access the data space corresponding to culled points. Together, merge and compute precisely define the computations in an Einsum. Common merge operations include intersection ( _∩_ ), which touches points with non-zero values in _both_ operands; and union ( _∪_ ), which touches points where at least one of the operands is non-zero. 

The full EDGE specification for GEMM is then: 

**==> picture [211 x 23] intentionally omitted <==**

where[�] _k_[specifies][a][map][action][between] _[A]_[and] _[B]_[on][the] _k_ rank and the intersection merge operator ( _∩_ ) culls _k_ points where at least one operand is zero. The compute operator ( _×_ ) multiplies the data values of coordinates surviving intersection. The reduce action ([�] _k_[)][on][the] _[k]_[rank][gathers][all][non-empty] points in the _k_ rank and reduces them using addition (+). In this work, we use three user-defined computations: 

- 1) Maximum (max( _∪_ )) takes the maximum of two values. Suppose we have the following expression: _Zm_ = _Am · Bm_ ::[�] _m_[max(] _[∪]_[)][. The union merge operator (] _[∪]_[) filters] out any _m_ coordinates where both operands contain 0 (and places 0 in the output). The max compute operator then returns the maximum of the two operands. 

- 2) Divide ( _÷_ ( _←_ )) divides two data values. Given the following expression, _Zm_ = _Am · Bm_ ::[�] _m[÷]_[(] _[←]_[)][,] the merge operator ( _←_ ) only touches _m_ points where there is a non-zero value in the _B_ operand (see [39, 

1460 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

Appendix]), and the compute operator divides the data value in _A_ with the data value in _B_ . 

- 3) Subtraction and Exponentiation: To apply the exponential to an expression that subtracts two tensors, we use the following expression: _Zm_ = _Am · Bm_ ::[�] _m_[sub-then-exp][(][1][)][.][The][user-defined][operator] (sub-then-exp) performs _Am_ minus _Bm_ then applies the exponential to the result. The merge operator, 1 , is EDGE’s “pass-through” operator, which touches all _m_ points in the iteration space. 

In addition to map and reduce, EDGE enables the expression of user-defined _unary_ operations on tensors. For example, we can express the application of the non-linear, sigmoid function ( _σ_ ) on each element of a tensor _A_ as _Zm_ = _σ_ ( _Am_ ). 

_2) Shorthand Notation:_ Throughout this paper, we take advantage of EDGE’s shorthand notation [39] in the following ways: 

- We drop all reduce actions that consist of add and union in the compute and merge operator, respectively ([�] +( _∪_ )). Thus, _Zm_ = _Ak,m_ ::[�] _k_[+(] _[∪]_[)][becomes] _Zm_ = _Ak,m_ . 

- We express all map actions using infix notation; that is, _Ak,m · Bk,n_ ::[�] _k[×]_[(] _[∩]_[)][becomes] _[A][k,m][ ×][ B][k,n]_[.] 

- When max is part of a map action ( _Am · Bm_ :: � _m_[max(] _[∪]_[)][), we replace it with the following shorthand:] max( _Am, Bm_ ). 

- When _÷_ is part of a map action ( _Am · Bm_ ::[�] _m[÷]_[(] _[←]_[)][),] we replace it with the following: _Am/Bm_ . 

- When sub-then-exp is part of the map action ( _Am · Bm_ :: � _m_[sub-then-exp][(][1][)][),][we][replace][it][with][the][shorthand] _e[A][m][−][B][m]_ . 

- We can express rank variable expressions with only one valid coordinate (e.g., _Si_ : _i_ =2) using just the coordinate (in this case, _S_ 2). 

_3) Filtering Rank Expressions:_ EDGE also enables expressing Einsums that touch only a subset of the data space of their constituent tensors. For example, we may express the prefix sum of a tensor _Ak_ with the following Einsum: 

**==> picture [61 x 11] intentionally omitted <==**

For each coordinate _i_ , _Si_ +1 is built by reducing together the subset of _A_ whose coordinates are _≤ i_ . Note that this definition of prefix sum computes the entire sum for a given _i_ without iteratively reusing the previous sum. 

_4) Expressing Iterative Computations:_ EDGE expresses recursion and iteration through generative/iterative ranks. We use the term _standard_ ranks to differentiate non-iterative ranks from iterative ranks. We can express the iterative prefix sum as follows: 

**==> picture [160 x 25] intentionally omitted <==**

Here, _S_ is a tensor with the iterative rank, _I_ , ranging from 0 to _K_ (inclusive). Statement 4 indicates the stopping condition for the iterative expression (when _i_ is greater than or equal to _K_ ). 

_5) Cascades of Einsums:_ TeAAL [35] introduces the concept of _cascades_ of Einsums, which expresses directed acyclic graphs (DAGs) of Einsum expressions as a sequence of subEinsums. One can view the unrolled iterative expression in Einsum 3 as a cascade: 

**==> picture [78 x 54] intentionally omitted <==**

Finally, we use the EDGE _Initialization_ label to specify computations that initialize tensors, which occur once. We use the EDGE _Extended Einsum(s)_ label to specify the computation that occurs on each iteration of a cascade of Einsums [39]. For example, see (Einsum) Cascade 5. 

## _D. Mapping and Binding_ 

While the cascade of Einsums specifies what computation is required, the _mapping_ and _binding_ describe how it should occur [9], [35], [41], [51]. We use the concept of _logical tasks_ to define these terms. A logical task is a grouping of points in the iteration spaces of all Einsums. Tasks are defined such that each point in the iteration spaces is assigned to exactly one task. Logical tasks can be as small as a single point or as large as an entire iteration space. In the final schedule, each logical task must be assigned to exactly one compute unit that finishes the given task before moving onto the next task. 

The mapping, therefore, describes a _task graph_ , a directed, acyclic graph whose nodes are logical tasks and edges are dependencies between the tasks. Mapping specifications typically include aspects such as loop order, partitioning, and work scheduling (sequential vs. parallel operations) [35]. Thus, the dependencies in the task graph can be true dependencies (enforced by the cascade) or additional ordering constraints imposed by the mapping specification. 

The binding describes how the tasks are bound to the actual hardware, including which compute unit each task is associated with, when that task will be executed, and where the inputs and outputs are stored in the memory hierarchy. This binding must obey the dependencies present in the task graph and the physical limitations of the architecture but is otherwise unconstrained. 

## _E. Tensor Algebra Accelerators_ 

In recent years, the popularity of domain-specific tensor algebra accelerators has increased. A typical accelerator based on a spatial architecture consists of off-chip main memory, an on-chip shared global buffer, various scratchpads, and a 1D and/or 2D processing engine (PE) array where each PE contains compute units [9], [27], [28], [38], [62]. This design minimizes memory transfer latency while maximizing compute utilization [7]–[9], [11], [27]. Various tools enable the quick modeling and design space exploration of tensor algebra accelerators, including Timeloop [41] and Accelergy [56], GAMMA [61], and DOSA [23]. 

1461 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

III. PASSES PERFORMED BY A CASCADE OF EINSUMS 

Our first contribution is to demonstrate a novel analysis that can be applied to a cascade of Einsums. The key insight is that cascades of Einsums provide a precise description of the iteration space for each Einsum and the data space for each constituent tensor, enabling us to derive the algorithmic minimum live footprint for each tensor, with implications for the allowed fusion schedules and required buffer capacity/memory traffic. Because this analysis relies only on the cascade of Einsums, it holds for any choice of mapping and binding. 

## _A. Calculating the Number of Passes_ 

We will apply our analysis to attention in Section IV. To illustrate ideas, we first start with a simple pedagogical example, shown in Cascade 1. 

**==> picture [253 x 49] intentionally omitted <==**

Cascade 1: An example 2-pass cascade. 

Einsum 5 performs a dot product between _Ak_ and _Bk_ , and Einsum 6 multiplies the first Einsum’s result _Y_ by _Ak_ again to produce _Z_ . If we want to minimize data traffic of _Ak_ , we need to choose a dataflow for each Einsum that keeps _Ak_ stationary and fuses the two Einsums together. In other words, the dataflow must finish using the first element of _Ak_ before moving onto the next. However, such a dataflow does not exist for this cascade. Any implementation must visit _every_ element of _Ak_ to compute _Y_ before it can revisit _any_ element of _Ak_ to compute _Z_ . 

We define a _pass_ that a cascade performs over a particular fiber of a particular rank and tensor to be a traversal of every element of that fiber. Each time an element _must_ be revisited _after_ visiting every other element of that fiber, there is an additional pass. For example, Cascade 1 performs two passes over the _K_ rank of _Ak_ . 

Since an Einsum’s iteration space can also be represented as a fibertree (i.e., an _is-fibertree_ – see Section II-B), we extend our definition of an iteration space for a cascade of Einsums by considering its iteration space to be the sequence of the is-fibertrees for each Einsum. Now, in a scenario where fibers for a particular rank exist in multiple is-fibertrees; in each, they project to the same tensor; and there is a dependency such that all of the elements of the earlier is-fibertree’s fiber must be read before any element can be read again by the later is-fibertree (for all mappings of the cascade), we refer to that read-read sequence as creating an additional _pass_ . When there is a sequence of _N_ such read-read dependencies, we say the cascade is an ( _N_ + 1)-pass cascade. For our example, Cascade 1 requires two passes of the _K_ rank. 

## _B. Implications of the Number of Passes_ 

The number of passes a cascade performs is relevant because it restricts possible fusion schedules. Einsums within a 

pass can be fused at will, producing and consuming a tile of the intermediate at a time. Einsums in different passes cannot be fused. Revisiting Cascade 1, Einsums 5 and 6 cannot be fused on the _K_ rank. Any implementation must visit all elements of the _K_ fiber of _A_ to produce _Y_ before it can visit any of the elements of that fiber to produce _Z_ . 

This analysis also provides a non-trivial lower bound on the tensors’ live footprints. For example, the algorithmic minimum live footprint for tensor _A_ is a fiber of shape _K_ . In other words, an architecture must either have enough buffer space to hold an entire _K_ fiber of _A_ or spill and reload that fiber, incurring memory traffic proportional to the shape of _K_ . We note that this analysis is mapping independent. There is no dataflow for this cascade that enables a smaller live footprint. 

## _C. Reducing the Number of Passes via Reassociation_ 

Given the restrictions that multi-pass cascades place on the allowed dataflows and tensor live footprints, it can be beneficial to manipulate the cascade to reduce the number of passes required. Crucially, these manipulations are functionally equivalent and only change how _Z_ is computed. In this section, we will present two methods for doing so, though we leave a full analysis of the space of pass-reduction approaches to future work. 

_1) Deferring the Multiplication by Y :_ First, we recognize that, by the distributive property, Einsum 6 can be factored to perform the reduction of _Ak_ first, before multiplying the result by _Y_ . Doing so, we get the following cascade: 

**==> picture [253 x 64] intentionally omitted <==**

Cascade 2: A reassociation of Cascade 1 that defers the _Y ×_ to compute _Z_ with 1-pass of the _K_ rank. 

Now, because there is no read-after-write dependency between Einsums 7 and 8, both Einsums can be included in the same pass. In fact, because Einsum 8 reduces away the _K_ rank, Cascade 2 is a 1-pass cascade with respect to this rank. This reassociation actually provides a second benefit over Cascade 1: Einsum 9 now only requires one multiplication (as opposed to _K_ multiplications in Einsum 6). 

_2) Iteratively Constructing Y and Z:_ Alternatively, we can iteratively construct _Y_ and _Z_ as we perform the pass through _Ak_ . To do so, we will take a similar approach to the prefixsum (see Sections II-C3-II-C4) and build intermediate _Y_ s and _Zs_ . 

**==> picture [182 x 26] intentionally omitted <==**

Just like with the prefix sum, this version requires a lot of extra compute, but, because _Y_ = _RYK_ and therefore _Z_ = _RZK_ , the result is the same. 

1462 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [253 x 148] intentionally omitted <==**

Cascade 3: A reassociation of Cascade 1 that iteratively constructs _Y_ and _Z_ with 1-pass of the _K_ rank. 

We remove this extra work by making the _I_ ranks of _RYi_ +1 and _RZi_ +1 iterative. This is shown in Cascade 3. Iterative _RYi_ +1 (Einsum 12) looks very similar to the iterative prefixsum. However, computing _RZi_ +1 is a little more complicated. 

To derive the expression for _RZi_ +1, we start by introducing one more intermediate _Si_ , which is the prefix sum for _Ak_ : 

**==> picture [157 x 11] intentionally omitted <==**

Now, we can combine Einsums 17 and 18 to write _RZi_ in terms of this prefix-sum: 

**==> picture [162 x 9] intentionally omitted <==**

Dividing both sides by _RYi_ , we derive an alternate definition for _Si_ : 

**==> picture [42 x 23] intentionally omitted <==**

_Si_ +1 can also be written using this alternative definition: 

**==> picture [165 x 23] intentionally omitted <==**

We can combine Einsums 19 and 20 to compute _RZi_ +1 in terms of _RZi_ (i.e., iteratively): 

**==> picture [139 x 25] intentionally omitted <==**

Distributing _RYi_ +1 and performing some reassociation, we get Einsum 13. 

Cascade 3 is also a 1-pass cascade, performing one pass of the _K_ rank of _Ak_ (indexed with the variable _i_ ) and iteratively building _RYi_ +1 and _RZi_ +1. Unfortunately, unlike Cascade 2, Cascade 3 does require extra compute over the original Cascade 1. However, memory bandwidth-limited workloads can afford to trade off extra compute for reduced memory traffic, and Cascade 3 may still provide benefit. 

**==> picture [253 x 108] intentionally omitted <==**

**----- Start of picture text -----**<br>
Add & Norm<br>FFN<br>Add & Norm<br>SoftmaxMatMul DeprojectionAttention ...<br>MatMul Proj Proj Proj<br>Q K V<br>(a) Encoder architecture (b) Required compute<br>**----- End of picture text -----**<br>


Fig. 1: Overview of transformer encoder inference. 

IV. TAXONOMIZING ATTENTION AS EINSUM CASCADES 

Our second contribution is to apply the cascade of Einsums abstraction and the notion of passes to transformer models to describe, taxonomize, and highlight trade-offs in the space of attention implementations. This section first looks at the transformer model as a whole, identifying attention as an important kernel (Section IV-A). We then give an overview of attention and a “straightforward” (but inefficient) algorithm for softmax by writing them as cascades of Einsums (Sections IV-BIV-C). Finally, we show how optimizations to softmax can be described by modifying the cascades and provide a taxonomy of the space using the number of passes required by each cascade (Sections IV-D-IV-E). 

## _A. Transformers_ 

Transformer models generally follow the architecture defined in [52]. Our work, which addresses the impact of long sequence lengths during self-attention, focuses on the encoder architecture.[1] Figure 1a gives an overview. The transformer first projects the input (by multiplying it by weight tensors) to form a _query_ , _key_ , and _value_ . Self-attention is made up of three operations: a matrix multiplication of the query and key, a softmax on the result, and another matrix multiplication, which combines the softmax output with the value. The attention output is then deprojected (again, multiplying by a weight tensor), normalized, passed through a two-layer feed-forward neural network (FFN), and normalized once more. 

As the sequence length grows, the relative importance of the different operations changes. Figure 1b shows that at shorter sequence lengths, the _weight-times-activation_ “linear” layers are a larger fraction of the total required compute, while at long sequence lengths, the attention operation dominates. In all cases, the additional non-linearities (e.g., the normalization, the ReLU between the FFN layers, etc.) have negligible impact. In the next section, we focus on describing attention more precisely, and use our analysis to understand prior work on efficient implementations. 

> 1During the decoder phase, inference is severely bottlenecked on the memory traffic required to read the KV cache [24], and therefore the onchip accelerator design has less impact on performance. 

1463 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

_B. Redefining Attention’s “Matrix Multiplications”_ 

In the original transformer paper [52], the kernel was described with the following equation: 

**==> picture [229 x 25] intentionally omitted <==**

However, this equation says almost nothing about what the inputs _Q_ , _K_ , and _V_ look like or what iteration space needs to be traversed. We clarify these points by rewriting the above as a cascade of Einsums, with the exception of the softmax, whose cascade we will explore in Section IV-C. The first step is to give each of the ranks names: _M_ and _P_ are the sequence lengths for _Q_ and _K_ / _V_ , respectively, and _E_ and _F_ are the embeddings for _Q_ / _K_ and _V_ , respectively. 

**==> picture [191 x 23] intentionally omitted <==**

**==> picture [182 x 12] intentionally omitted <==**

**==> picture [185 x 12] intentionally omitted <==**

Here, Einsums 22[2][,][3] and 24 look like matrix multiplications. Taking Einsum 24 as an example, for each point in the iteration space _F ×M ×P_ , we perform a multiplication using elements from two 2-tensors ( _Am,p_ and _Vf,m_ ) to produce a 2-tensor output ( _AVf,p_ ), which requires reducing across the inputs’ shared rank _M_ . Einsums 22-24 can be modified to refer to the full batched, multi-head self attention [52] by adding the batch ( _B_ ) and head ( _H_ ) ranks to all tensors. This changes the characteristics of the kernel. Adding the _B_ and _H_ ranks means that Einsums 22 and 24 behave like many independent matrix multiplications instead of one monolithic matrix multiplication. The challenges with attention, described in Section I, still follow clearly from this modification. Because _all_ tensors contain a _B_ rank, the matrix multiplications are all unique to the specific batch’s inputs. Therefore, none of these tensors can be computed before the inputs are given, and there is no data sharing between the different elements in the batch. Hence, to simplify notation, we assume the presence of the _B_ and _H_ ranks but omit writing them throughout the rest of paper. 

## _C. Softmax as a Cascade of Einsums_ 

We now apply the same precise notation to the softmax. A softmax [5] over a 1-tensor is traditionally expressed with the following equation: 

**==> picture [157 x 26] intentionally omitted <==**

> 2Einsums do not require the transpose, since this information is implicit in the indices. 

> 3In Einsum 22, we also substitute _E_ for _dk_ following the notation defined in Section II-B, where the shape of a rank is also its rank name. 

In the context of attention, this operation becomes two dimensional and can be expressed using the following cascade with input _QKm,p_ : 

**==> picture [175 x 43] intentionally omitted <==**

For each point in the iteration space ( _m_ , _p_ ), we exponentiate _QKm,p_ to generate the softmax numerator ( _SNm,p_ in Einsum 26), reduce _SNm,p_ with addition to produce the softmax denominator ( _SDp_ in Einsum 27), and finally, divide the numerator and denominator to produce the final result ( _Am,p_ in Einsum 28). 

_1) Improving Numerical Stability:_ Because _e[QK][m,p]_ can easily become extremely large, the above formulation suffers from overflow. Therefore, practical implementations [2], [42] often prefer the numerically stable variant that replaces Einsum 26 with: 

**==> picture [191 x 37] intentionally omitted <==**

and drop the ~~_√_~~ 1 _E_[term][when][computing] _[QK][m,p]_[.][4][To][compute] the _global maximum_[5] _GMp_ , we reduce _QKm,p_ with the operator max (instead of +). Notice that subtracting _GMp_ from _QKm,p_ in the exponent is equivalent to dividing by _e[GM][p]_ , and because the _e[GMp]_ 1[term appears in both the numerator (] _[SN][m,p]_ via Einsum 30) and denominator ( _SDp_ via Einsum 27), the result ( _Am,p_ ) stays the same. This construction improves numerical stability by bounding the values of the softmax numerator _SNm,p_ to the range (0 _,_ 1]. 

## _D. Optimizing Softmax Compute_ 

We now describe an optimization to attention that reduces compute requirements, specifically division. This optimization was used in FlashAttention-2 [14]. We point out that it can be applied more broadly, i.e., to any cascade we discuss in Section IV-E. Einsum 28 requires _M ×P_ divisions. While this is the best we can do for an independent softmax, we note that attention does not use the softmax in isolation [52]. Instead, it subsequently multiplies the result, _Am,p_ , and another tensor, _Vf,m_ , per Einsum 24, reproduced here: 

**==> picture [92 x 11] intentionally omitted <==**

To optimize the full attention cascade, we can refactor Einsums 28 and 24 by, instead, first combining _SNm,p_ and _Vf,m_ (Einsum 31) and reducing across the _M_ rank and then performing the division (Einsum 32), as follows: 

**==> picture [180 x 26] intentionally omitted <==**

4The 1[was][introduced][to][bound][the][magnitude][of] _[SN][m,p]_[[][52][].] ~~_√_~~ _E_[term] Because the numerically stable softmax variant already accomplishes this, the scaling is often omitted [12], [14], [15]. 

> 5“Global” here refers to over the entire _M_ 

1464 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

|**3-pass**|**2-pass**|**1-pass**|
|---|---|---|
|PyTorch [42]<br>TensorFlow [2]<br>FLAT [28]<br>E.T. [6]|TileFlow [62]<br>Choi et al. [12]|FlashAttention [15]<br>FlashAttention-2 [14]<br>Rabe and Staats [47]|



TABLE I: Classifying prior attention algorithms. 

This reassociation does _F × P_ divisions instead of _M × P_ divisions. Since _M_ is the sequence length and _F_ is an embedding dimension (i.e., _M ≫ F_ ), this reassociation _reduces_ the required divisions (by a factor of _[M] F_[).] 

## _E. Optimizing Softmax Live Footprint and Memory Traffic_ 

We now apply the analysis described in Section III to analyze attention’s live footprint and memory traffic. We consider the _exact attention_ literature, omitting works that either do not model/evaluate the softmax or include approximation strategies that improve performance at the cost of reduced accuracy (increased perplexity). We discuss the latter in Section VII. 

We find that existing approaches to attention can be classified as either 3-pass, 2-pass, or 1-pass cascades, where an _N_ -pass cascade performs _N_ passes of a given _M_ fiber. See Table I. Next, we describe the key ideas of each. 

_1) 3-Pass Attention Cascades:_ The 3-pass cascade is the straightforward, numerically stable cascade that we already discussed in Section IV-C1, namely Einsums 29-30 followed by Einsums 27-28, reproduced in Cascade 4 for clarity. 

|_QKm,p_ <br>_GMp_|= <br> =|_Qe,p × Ke,m_<br>_QKm,p_ ::<br>�|max(_∪_)|/* Pass 1 */|(33)<br>(34)|
|---|---|---|---|---|---|
|||_m_||||
|_SNm,p_|=|_eQKm,p−GMp_||/* Pass 2 */|(35)|
|_SDp_ <br>_Am,p_ <br>_AVf,p_|= <br> = <br> =|_SNm,p_<br>_SNm,p/SDp_<br>_Am,p × Vf,m_||/* Pass 3 */|(36)<br>(37)<br>(38)|



Cascade 4: The 3-pass attention cascade. 

In Pass 1, we compute _QKm,p_ and _GMp_ ; in Pass 2, we compute _SNm,p_ and _SDp_ ; and in Pass 3, we compute _Am,p_ and _AVf,p_ . Notice that we must finish an entire _M_ fiber of Einsum 34 (reading an entire _M_ fiber of _QKm,p_ ) before _GMp_ is ready to start Einsum 35 (where we must read the same _M_ fiber of _QKm,p_ again). Similarly, we must finish an entire _M_ fiber of Einsum 36 (reading an entire _M_ fiber of _SNm,p_ ) before _SDp_ is ready to start Einsum 37 (where we must read the same _M_ fiber of _SNm,p_ again). To summarize, as a consequence of the _dependencies_ between Einsums, this cascade must perform three passes over each _M_ fiber. This holds for any choice of mapping (including ones that perform fusion). 

_2) 2-Pass Attention Cascades:_ We now briefly summarize the 2-pass cascade, deferring details due to space. Rather than computing the global max and then starting the softmax (as 

in the 3-pass cascade), the 2-pass cascade first partitions the input, computes a per-partition _local max_ and applies it to form a variant of _SNm,p_ whose elements are likewise partitioned and adjusted by the local max. Analogously, each partition gets a local denominator (also adjusted by the same local max). While this is occurring, it builds the global max from the local max values. Next, in a second pass, it uses the global max to correct the per-partition numerators and denominators and compute the softmax output. 

_3) 1-Pass Attention Cascades:_ While prior work proposes multiple different 1-pass cascades [14], [15], [47], the main ideas are the same in each. Rather than using the per-partition local max to compute the local numerator and denominator, instead keep a _running max_ that represents the max value seen so far. Each time a new running max is computed, also adjust previous results (e.g., numerator-times- _V_ , denominator, etc.) with this max. 

This transformation can be described more precisely using the reassociations presented in Section III-C. First, we modify Cascade 4 to multiply the softmax numerator-times- _V_ and then compute the division (as described in Section IV-D). This reassociation combines the second and third passes of Cascade 4 (see Section III-C1). To ensure numerical stability, we cannot use the same strategy to combine the first and second passes. So we instead use the iterative approach (see Section III-C2). 

We are now ready to describe FlashAttention-2’s 1-pass cascade (shown as Cascade 5). We later use it to build FuseMax. Note the evidently increased compute relative to the 3-pass cascade. We will carefully design the binding in Section V to hide these overheads on a spatial architecture. 

We will start by expressing the partitioning of both of the inputs _Ke,m_ and _Vf,m_ into M1 chunks of M0 elements each (Einsums 39-40). After computing _BQKm_ 1 _,m_ 0 _,p_ , this allows us to perform operations like maximum on individual _M_ 0 fibers, rather than on the whole tensor (Einsum 45). The problem is, of course, that the local maximum is not necessarily the same for all _M_ 0 fibers and so will not just cancel nicely like the global maximum. 

We resolve this by instead using the running maximum ( _RMm_ 1 _,p_ )—the global maximum of all inputs seen so far— instead of the local maximum. We recognize that _M_ 1 can also serve as an iterative rank, and iteratively build up _RMm_ 1 _,p_ . After initializing _RMm_ 1: _m_ 1=0 _,p_ to _−∞_ (Einsum 41), we compute a new running maximum _RMm_ 1+1 _,p_ using the running maximum computed in the previous iteration _RMm_ 1 _,p_ and the new local maximum _LMm_ 1 _,p_ (Einsum 46). 

We can now use the running maximum to compute a local numerator _SLNm_ 1 _,m_ 0 _,p_ (Einsum 47), a local denominator _SLDm_ 1 _,p_ (Einsum 48), and even the softmax numeratortimes- _V SLNVf,m_ 1 _,p_ (Einsum 49) using the partitioned _BVf,m_ 1 _,m_ 0 (Einsum 40). 

Now consider the softmax denominator. Eventually, we would like to reduce _SLDm_ 1 _,p_ into a 1-tensor, but because its values may have been computed with different maximums, we cannot simply use addition. Instead, by introducing a 

1465 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

Initialization: 

**==> picture [186 x 71] intentionally omitted <==**

Extended Einsums: 

**==> picture [219 x 201] intentionally omitted <==**

Cascade 5: A 1-pass attention cascade. Note that _M_ 1 is used as a standard rank (e.g., to access _BQKm_ 1 _,m_ 0 _,p_ ) and as an iterative rank (e.g., to access _RMm_ 1 _,p_ ). The stopping condition for all iterative ranks is _m_ 1 _≥ M_ 1 (Statement 56). 

new running denominator _RDm_ 1 _,p_ with iterative rank _M_ 1, we can correct the old denominator _RDm_ 1 _,p_ to the new running maximum _RMm_ 1+1 _,p_ and then perform the addition. We start by initializing the running denominator at the start of the computation to 0 (Einsum 42). Then, at each point _m_ 1, the correction factor _PRMm_ 1 _,p_ allows us to correct the previous running denominator _RDm_ 1 _,p_ with the new maximum (Einsum 51). In other words, _RDm_ 1 _,p_ is downscaled by _e[RM][m]_[1] _[,p]_ . _SPDm_ 1 _,p_ “switches” the downscaling factor on _RDm_ 1 _,p_ to _e[RM][m]_[1+1] _[,p]_ by multiplying _RDm_ 1 _,p_ by _e[RMm] e[RMm]_[1+1][1] _[,p][,p]_ ( _PRMm_ 1 _,p_ ). Once _SLDm_ 1 _,p_ and _SPDm_ 1 _,p_ have the same maximum, they can be combined to produce the new running denominator _RDm_ 1+1 _,p_ (Einsum 52). We can do the same to compute the running numerator-times- _V_ (Einsums 43, 53-54). 

Finally, _AVf,p_ can be computed by dividing the final numerator-times- _V_ by the final denominator. By construction, at this point, _RNVf,M_ 1 _,p_ and _RDM_ 1 _,p_ are both downscaled by the same maximum _RMM_ 1 _,p_ (conveniently, also the global maximum) and can be correctly combined. 

## V. MAPPING AND BINDING ATTENTION 

Based on the framework from Section IV, we now describe FuseMax, an efficient mapping and binding of an attention algorithm (specifically the 1-pass cascade in Cascade 5) to a spatial array-style architecture. To enable maximum flexibility while binding, FuseMax’s mapping places each iteration space point in its own logical task. 

The goal when binding a cascade onto hardware is to fully utilize all available compute units. In our evaluation of prior work (Figure 6 and Section VI-B), we observe that at short sequence lengths, the 2D PE array is under-utilized because it must wait for the 1D PE array to compute the softmax. At longer sequence lengths, both arrays are under-utilized since the workload becomes memory-bandwidth limited. 

FuseMax’s binding addresses these issues to achieve full utilization on both the 1D and 2D PE arrays. First, we decrease the compute performed by the 1D array by (1) applying the division reduction optimization (Section IV-D) and (2) sharing the other operations (sum/max/exp) between the 1D and 2D arrays. Similarly, we ensure that the workload is never memory-bandwidth limited by deeply fusing all Einsums in the cascade to restrict the live footprint to only what can be buffered on-chip. No matter the sequence length, our dataflow is never forced to spill any of its intermediates off-chip. 

**Architecture.** FuseMax is a spatial array architecture based on the TPUv2/TPUv3 [37, Figure 1(e)]. The off-chip DRAM and a large global buffer both feed data to connected 2D and 1D arrays (see Figure 2). We set parameters to match the cloud configuration in prior work [28]. 

**==> picture [183 x 141] intentionally omitted <==**

**----- Start of picture text -----**<br>
���<br>����<br>������ �������<br>�����������<br>�������<br>�����������<br>�������<br>������������<br>��������������<br>Fig. 2: Spatial array architecture assumed for FuseMax.<br>���<br>**----- End of picture text -----**<br>


Figure 3 shows the evolution of the 2D PE array architecture, from a fixed-dataflow multiply-accumulate TPU PE (Figure 3a) to a flexible-dataflow multiply-accumulate PE (Figure 3b) to a FuseMax PE (Figure 3c). Note, although both the 1D and 2D PE arrays in FuseMax perform exponentiation, we implement exponentiation with 6 sequential multiplyaccumulate operations [36], [53] and therefore do not require a dedicated exponentiation unit. 

**Mapping.** Prior attention accelerators [28], [62] explore fusing many of attention’s loop nests together. However, because these accelerators all use multi-pass cascades, the algorithmic minimum live footprint of some tensors (e.g., 

1466 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [253 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
�������������<br>������������<br>�������� �������� ���������<br>(a) TPU [27] PE (b) FLAT [28] PE (c) FuseMax PE<br>Fig. 3: 2D PE architecture evolution.<br>**----- End of picture text -----**<br>


_QKm,p_ ) is _O_ ( _M_ ), meaning that for long sequence lengths, intermediates cannot be buffered on chip. 

FuseMax leverages fusion in conjunction with the 1-pass cascade to eliminate the memory traffic of these tensors, regardless of the sequence length. Specifically, we partition on both _M_ and _P_ (forming _M_ 1 _, M_ 0 and _P_ 2 _, P_ 1 _, P_ 0), and maximally fuse all levels in the attention loopnest as shown in Mapping 1. That is, all Einsums in Cascade 5 are fused except for the last (which is fused to the rest only on _P_ 2). 

**==> picture [253 x 176] intentionally omitted <==**

Mapping 1: The FuseMax mapping as a loopnest. We partition on both _M_ and _P_ and map the innermost ranks _M_ 0 and _P_ 0 to the spatial array PEs. ComputeRNVTile performs Einsums 44-54 from Cascade 5. ComputeAVTile performs Einsum 55. Note that each Einsum represents a loopnest: by writing all Einsums in ComputeRNVTile under a single loopnest, we mean that we are maximally fusing those loopnests. Outer loops over _B_ and _H_ (if performing batched multihead attention) are not shown. 

While prior work implementing attention in hardware [28], [62] does utilize the 2D spatial array for the tensor products, it fails to do so for the softmax, choosing instead to use the 1D array. Because there are far fewer total PEs in the 1D array than the 2D array, the softmax becomes a bottleneck. FuseMax improves utilization of the 2D spatial array by using it for both the tensor products and the exponentiation operator in the softmax. FuseMax parallelizes across the _M_ 0 and _P_ 0 ranks throughout the attention kernel (see Mapping 1). We set _M_ 0 _× P_ 0 = # 2D Array PEs. The large spatial reductions required when parallelizing across the _M_ 0 rank are easily handled by the low-cost inter-PE communication network. 

**Binding.** The dependencies between different Einsums in our cascade necessitate a binding that implements fine-grain 

pipeline parallelism to achieve high utilization of both the 1D and 2D spatial arrays. Figure 4 shows the waterfall diagram for FuseMax in the steady state. Time is broken into epochs. Each epoch performs the same set of tile-granular operations at specific tile-relative coordinates (given by _a, b, c, d_ in the figure). Across all epochs, the kernel evaluates all tiles and each Einsum in Cascade 5 is mapped to either the 2D or 1D array for all epochs (as shown in the figure). 

A major design consideration when binding the attention kernel is how to overcome the latency of fills and drains to/from the spatial array. Consider a tile of _QKm,p_ of shape _M_ 0 _× P_ 0. Per Einsum 22, the iteration space to evaluate this tile is _E × M_ 0 _× P_ 0 which becomes _E_ cycles on the spatial array. For the networks we evaluate, _E_ = 64 or 128. Assume _E_ = 64. Using an output stationary dataflow, while each PE performs 64 MACCs, it takes _∼_ 256 cycles to both fill and drain the spatial array. Without careful interleaving, this combination of parameters causes low utilization because, for example, the running max _RMm_ 1+1 _,p_ 1 _,_ : cannot be computed until a tile of _QKm_ 1 _,_ : _,p_ 1 _,_ : is completed and spatially reduced (drained) to form the local max _LMm_ 1 _,p_ 1 _,_ : (Einsums 45-46). 

Our binding address the above issues with two levels of interleaving. First, we interleave the construction of dependent tiles across epochs. This is reminiscent of software pipelining. For example, in Figure 4 the _d_ -th tile of _BQK_ and _LM_ are completed in Epoch _i_ (as they correspond to a fill followed by a drain and can be easily pipelined). The _RM_ (which has to wait for the drain) for tile _d_ takes place _in a later epoch_ . Instead, Epoch _i_ computes an earlier tile’s running maximum _RM_ [ _c_ ]. 

Second, we interleave the construction of certain tiles within an epoch at a fine (e.g., cycle-by-cycle) granularity. See the notation ‘ _A|B_ ’ in Figure 4. This is to ensure high utilization of both the 2D and 1D PE arrays at all times. To make this more clear, Figure 5 shows the start up and steady-state interleaving of _SLNV_ and _BQK_ in the 2D array and _SPNV_ and _RNV_ in the 1D array. In each cycle, a given PE in the 2D array computes a value for either _BQK_ or _SLNV_ and this alternates cycle by cycle. Each neighbor-neighbor link in the array is active in every cycle—carrying data for one of the two operation types. By interleaving _SLNV_ with _BQK_ , the 1D PEs can concurrently compute _SPNV_ and _RNV_ . 

Putting everything together, as Section VI-B will show, the above enables high utilization of all 2D and 1D array PEs. 

**FuseMax on GPUs.** FuseMax’s mapping and binding cannot be directly applied to GPUs. FuseMax’s architecture features heterogeneous PEs, each with smaller per-PE storage, and cheap (but restricted) inter-PE communication. Specifically, the networks that connect the PEs within the 2D array allow efficient, fixed-latency communication primarily between neighbors, including between the bottom of the 2D array and the 1D array. However, the GPU architecture features opposite characteristics: homogeneous PEs, each with relatively large per-PE storage, and expensive, loosely coupled inter-PE communication. While concurrent work [50] has explored using the GPU’s Tensor Cores to compute _BQK_ and 

1467 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [412 x 126] intentionally omitted <==**

**----- Start of picture text -----**<br>
�������<br>�������� ����� ������� ����� ����� ����� �<br>�������� ����� � ���� ���� ����� ����� ��������<br>����<br>��������� ���������<br>**----- End of picture text -----**<br>


Fig. 4: FuseMax pipelining at a glance. Each tensor name (e.g., _SLNV_ ) corresponds to the Einsum used to compute that tensor (see Cascade 5). _a_ , _b_ , _c_ and _d_ denote tile-relative coordinates where _a < b < c < d_ . If Epoch _i_ produces tiles with coordinates _a, b, c, d_ , Epoch _i_ + 1 produces tiles with identifiers _a_ + 1 _, b_ + 1 _, c_ + 1 _, d_ + 1. And so on. ‘ _A|B_ ’ denotes ‘computing tile _A_ is interleaved with computing tile _B_ .’ ‘ _A → B_ ’ denotes ‘computing tile _A_ is done before computing tile _B_ .’ Computing _AVf,p_ is not shown. The green and blue time periods making up an epoch take almost the same number of cycles. 

**==> picture [412 x 286] intentionally omitted <==**

**----- Start of picture text -----**<br>
� � �<br>�<br>�<br>�������� �������� ��������<br>�������� �������� ��������<br>�<br>����� ����� �����<br>����� �����<br>������<br>� � � �� ��<br>� � � � ���� ���<br>� ����<br>� � � �<br>� � � �<br>������<br>�������� �������� �<br>�������� �������� ���<br>� � �<br>� � � �<br>� � �<br>�<br>�<br>�<br>**----- End of picture text -----**<br>


Fig. 5: Intial pipeline fill ( _t_ = 0 to _t_ = 2) and steady-state ( _t_ = 3 and _t_ = 4) for the intra-epoch interleaving of _SLNV |BQK_ and _SPNV |RNV_ to maximize 2D and 1D PE utilization, respectively, on a toy 2x2 array. Each color indicates a tensor and each number indicates a point in that tensor (e.g., the point _BV_ 0 moves from the top left PE at _t_ = 1 to the top right PE at _t_ = 2). To reason about signal timing, we use input (but not output) latches for data in each PE, so moving data appears on output wires. Some stationary tensors (e.g., _BQK_ ) and Einsums (e.g., _SLD_ ) are omitted for clarity. 

_SLNV_ and using software pipelining to hide the latency of the other compute, the GPU’s loosely coupled threads require frequent synchronization to maintain correctness. FuseMax takes advantage of the tight coupling between the 2D and 1D arrays to statically schedule compute between the arrays, enabling high utilization across the board without sychronization. 

## VI. EVALUATION 

In this section, we demonstrate how FuseMax’s cascade, architecture, and binding work together to achieve improvements in both performance and energy relative to the state of the art, for both attention and end-to-end transformer inference. 

## _A. Experimental Set-Up_ 

First, we present the experimental setup details common to all following subsections. 

**Workloads.** We evaluate all accelerators and configurations using the same transformer models used by FLAT [28]: BERTBase [17] (BERT), TrXL-wt103 [13] (TrXL), T5-small [49] (T5), and XLM [13]. We omit FlauBERT [31] because it uses the same hyperparameters as TrXL. We also note that though T5 is an encoder-decoder model, we only evaluate the encoder in this work. Following FLAT, we use a batch size _B_ = 64 for all evaluations. 

1468 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [78 x 7] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) 1D PE array utilization<br>**----- End of picture text -----**<br>


**==> picture [78 x 7] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b) 2D PE array utilization<br>**----- End of picture text -----**<br>


Fig. 6: Utilization of the different PE arrays on the unfused baseline, FLAT, and three configurations building up FuseMax. 

**==> picture [516 x 85] intentionally omitted <==**

Fig. 7: 2D array utilization by Einsum across different configurations—FLAT (FL), +Cascade (+C), +Architecture (+A), and +Binding (+B)—and sequence lengths on BERT. 

**==> picture [516 x 78] intentionally omitted <==**

Fig. 8: Speedup of attention for FLAT and three configurations building up FuseMax over an unfused baseline. 

**==> picture [516 x 77] intentionally omitted <==**

Fig. 9: Energy consumption of attention for FLAT and three configurations building up FuseMax over an unfused baseline. 

**==> picture [516 x 77] intentionally omitted <==**

Fig. 10: Speedup of transformer inference on FLAT and three configurations building up FuseMax over an unfused baseline. 

1469 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [516 x 77] intentionally omitted <==**

Fig. 11: Energy consumption of transformer inference on FLAT and three configurations building up FuseMax over an unfused baseline. 

**Modeling with Timeloop and Accelergy.** We perform our evaluation using two tools for tensor algebra accelerator modeling and design space exploration: Timeloop [41] and Accelergy [56]. We use these tools to build models of the accelerator architectures at a 45nm technology node and evaluate each Einsum individually. Results from individual Einsums are combined using heuristics presented in prior work for evaluating full cascades [35]. Together, these tools allow us to evaluate execution time, energy, and area for all our designs. We perform floating-point division using the design in Xia et al. [59], scaled down to a 45nm technology node [56]. 

**Unfused Baseline.** We build the unfused baseline by combining the costs of three phases: _QK_ (Einsum 22), the 3-pass softmax (Cascade 4), and _AV_ (Einsum 24). Because this baseline is unfused, each phase can be scheduled independently, but proceed sequentially and require outputs to be written to memory between phases. We use Timeloop to search for efficient mappings to perform _QK_ and _AV_ . Additionally, we model the softmax for the unfused baseline by allowing the accelerator to load the _M_ fibers of the input on-chip one-byone (spilling if there is not enough space) before performing the compute. We model the memory traffic, compute, and energy required to perform all Einsums required for attention. 

**FLAT Baseline.** Our main baseline is the state-of-theart attention accelerator FLAT [28]. Though we started with the FLAT authors’ original code, we found and corrected a number of bugs. Through private correspondence with the FLAT authors, we verified the bugs were indeed bugs. We also discovered a couple of larger conceptual errors, which the authors told us to avoid by restricting FLAT to only search through configurations without these issues. 

Beyond correcting the FLAT codebase, we created and validated a Timeloop model that reproduces the FLAT authors’ (corrected) code to within _<_ 1% error. However, the FLAT codebase does not model the cost to perform the softmax. Specifically, their model ignores the cost of the data transfers required for the softmax (between any levels of the memory hierarchy) and uses 2[30] 1D PEs for compute. When comparing FuseMax and FLAT in this work, we augment our Timeloop model to model softmax correctly per the 3-pass cascade implicitly assumed by FLAT using only 256 1D PEs. 

**FuseMax Configurations.** To demonstrate the sources of the improvements achieved by FuseMax, we present three configurations, one associated with each of the major changes we propose: +Cascade uses the 1-pass cascade on the FLAT architecture, +Architecture adds the FuseMax architecture but 

implements a binding that fully produces and consumes one _M_ 0 _× P_ 0 tile of _BQK_ before starting the next, and +Binding adds FuseMax’s pipelined/interleaved binding. 

**Hardware parameters.** Figure 2 shows the selected hardware parameters. We chose the PE array dimension to match FLAT’s cloud accelerator and then set the global buffer capacity so that the overall chip area was as close to FLAT’s as possible. Also following FLAT, we use a 940 MHz frequency. We use Accelergy to model the area of both designs and find that FuseMax is 6.4% smaller. 

## _B. Evaluating Attention_ 

We now evaluate FuseMax to demonstrate the benefits it provides on the attention kernel by comparing it to the two baselines. 

**Utilization.** Figure 6a shows the utilization of the 1D PE array when performing attention. FLAT’s utilization drops for sequence lengths _≥_ 256K—it becomes memory bandwidth limited because it must spill the _QK_ and _A_ tensors to memory. By using a 1-pass cascade (+Cascade), FuseMax’s utilization becomes independent of sequence length. We also note that without the FuseMax binding (+Architecture), the 1D array is forced to stall and utilization drops. Adding in this binding (+Binding) enables FuseMax to fully utilize the 1D array again. 

Similarly, Figure 6b shows the utilization of the 2D array. Because of the large amount of compute required for the softmax, most configurations achieve poor utilization of this array. In fact, because the 1-pass cascade increases the compute required, +Cascade’s 2D array utilization is lower than FLAT’s at short sequence lengths. On the other hand, FuseMax (+Binding) achieves high utilization across the board and, at long sequence lengths, reaches almost 100% utilization. Both baselines achieve slightly higher utilization on XLM, which can be attributed to the higher intensity caused by a larger embedding dimension ( _E_ / _F_ ). 

Figure 7 explores this phenomenon in more detail, breaking down the utilization by Einsum. FuseMax effectively hides both the costs of the memory traffic and softmax compute, allowing it to achieve high 2D array utilization while spending most of the cycles on the tensor products. 

**Speedup.** Figure 8 shows that FuseMax achieves an average speedup of 10 _×_ over the unfused baseline and 6 _._ 7 _×_ over FLAT. We note FuseMax achieves lower speedup on XLM only because the baselines are able to achieve higher utilization of the 2D array on this transformer (Figure 6b). 

1470 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [248 x 189] intentionally omitted <==**

Fig. 12: Pareto-optimal curves for FuseMax at sequence length 256K. 

**Energy.** Figure 9 shows that FuseMax uses 77% the energy of the unfused baseline and 79% the energy of FLAT.[6] The energy use of the unfused baseline and FLAT are dominated by the DRAM access energy, the global buffer access energy, and the _QK_ and _AV_ (Einsums 22 and 24) compute energy. FuseMax achieves its energy savings by significantly reducing the DRAM and global buffer access energies. In fact, _≥_ 95% of the energy used by FuseMax across all models and sequence lengths goes to the compute performed by the MACC functional units in the 2D array. 

## _C. Evaluating Transformer Inference_ 

To evaluate the benefits of FuseMax on end-to-end transformer inference, we include the other required linear layers (Section IV-A). We use Timeloop to search for optimal mappings for these linear layers and use the same mappings for all three accelerator configurations. The attention modeling remains the same as Section VI-B. 

**Speedup.** Figure 10 shows the performance improvement achieved by FuseMax. Across the sequence lengths tested, FuseMax achieves an average speedup of 7 _._ 6 _×_ over the unfused baseline and 5 _._ 3 _×_ over FLAT. As discussed in Section IV-A, as sequence length grows, attention becomes a larger fraction of the total required compute. Therefore, at 1M tokens, FuseMax achieves an average 10 _×_ speedup over the unfused baseline and 7 _._ 5 _×_ speedup over FLAT. 

**Energy.** Figure 11 shows the energy reduction achieved by FuseMax. Here, we see similar results: as attention becomes a larger fraction of the kernel, the energy reduction increases. FuseMax uses 82% of the unfused baseline and 83% of FLAT’s energy during end-to-end inference. 

## _D. Pareto-Optimality of FuseMax_ 

We further observe that by varying the size of the PE array (between 16 _×_ 16 and 512 _×_ 512) and setting the global and per- 

> 6FLAT reports larger energy savings over the unfused baseline because it only reports energy associated with DRAM traffic during the tensor products. 

PE buffers to accommodate the resulting pipelined/interleaved binding, we generate a family of designs for efficient transformer inference. 

## VII. RELATED WORK 

Spatial architectures have been applied successfully to a variety of domains in academia [9], [10], [40], [45] and industry [3], [27]. Beyond FLAT [28] (discussed in the main body of the paper), TileFlow [62] is a framework for modeling and searching for efficient fused dataflows (including for attention) on spatial architectures. Though TileFlow does explore a broader space of dataflows than FLAT, even implementing the 2-pass softmax cascade (Section IV-E2), its dataflows remain softmax-compute limited. Recent work has explored the scheduling/compilation of a multi-Einsum kernels [21], [54], [62]. However, these works explore a limited set of transformations, making FuseMax’s inter-Einsum interleaving not discoverable. 

Quantization and sparsity have also been successfully applied to reduce the transformer inference compute and live footprint. We view these schemes as complementary to our work. GPTQ [20], AWQ [32], and LLM.int8() [16] quantize model weights to 4 or 8 bits without significant accuracy degradation. Outlier-aware quantization schemes like GOBO [60] and OliVe [22] quantize both weights and activations to a low-bit precision on specific hardware designs. SpAtten [53] prunes entire tokens and heads, while Sanger [34] and DOTA [46] use quantized or low-rank projected _Q_ and _K_ tensors to estimate which values of _QK_ and _A_ can be safely pruned. All of these algorithms are expressible as cascades of Einsums, and therefore, may be combined with FuseMax to improve performance and energy efficiency, though we leave their specification and implementation to future work. 

## VIII. CONCLUSION 

This paper advanced the state of the art in spatial accelerator design for transformer inference. To do so, we expressed attention and its variants as cascades of Einsums. We used these cascades to reason about attention’s characteristics, independent of its mapping/scheduling. Using these principles, we proposed FuseMax—an accelerator that uses deep fusion and fine-grain pipelining to map attention onto a spatial architecture. FuseMax achieves _∼_ 100% utilization of both PE arrays, demonstrating 6 _._ 7 _×_ speedup over the prior stateof-the-art (FLAT) using 79% of the energy on attention and 5 _._ 3 _×_ speedup over FLAT using 83% of the energy on end-toend inference. 

Our work shows that cascades of Einsums provide a powerful abstraction for representing and analyzing domainspecific kernels. Future work may explore their application to other attention variants (e.g., those exploiting quantization and sparsity) or even other domains (e.g., fully homomorphic encryption, scientific computing, relational algebra, etc.). Doing so enables mapping-agnostic analysis and may elucidate previously undiscovered cascades and schedules for these algorithms. 

1471 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

## ACKNOWLEDGMENT 

We thank the anonymous MICRO and MLArchSys reviewers for their feedback on submitted versions of the work; Abhimanyu Bambhaniya, Sheng-Chun Kao and Tushar Krishna for discussions on FLAT; and finally Tanner Andrulis and Angshuman Parashar for help with Timeloop. We would also like to thank Nafea Bshara, Ron Diamant, Serina Tan, Stephen Neuendorffer, Yakun Sophia Shao, Hongbin Zheng, and others at Amazon, AMD, and NVIDIA for helpful discussions about the work as it matured. 

## APPENDIX 

## _F. Expected Results_ 

Graphs will be displayed within the Jupyter notebook and/or found in workspace/outputs/generated/<timestamp or default>/figs/. They can be compared with Figures 6-12 in the paper or the corresponding figures in workspace/outputs/pregenerated/figs/. 

## _G. Methodology_ 

Submission, reviewing and badging methodology: 

- https://www.acm.org/publications/policies/ artifact-review-and-badging-current 

- https://cTuning.org/ae 

## _A. Abstract_ 

## REFERENCES 

In this artifact, we provide Timeloop and Accelergy models of the accelerator FuseMax, an accelerator for encoder-style transformer inference. For ease-of-use, we provide a Docker container and a set of Jupyter notebooks through which to run the experiments. This artifact can be evaluated on an x86-84 machine with 5 GB of disk space. 

## _B. Artifact check-list (meta-information)_ 

- **Algorithm:** Timeloop/Accelergy model of the FuseMax accelerator and the baselines it was evaluated against 

- **Program:** Python, Timeloop, Accelergy 

- **Run-time environment:** Docker, Jupyter 

- **Hardware:** x86-64 machine 

- **Output:** Plots generated from scripts 

- **Experiments:** Modeling of the five different accelerator design points via Timeloop and Accelergy models 

- **How much disk space required (approximately)?:** 5GB 

- **How much time is needed to prepare workflow (approximately)?:** 20 minutes 

- **How much time is needed to complete experiments (approximately)?:** 9 hours 

- **Publicly available?:** Yes 

- **Archived (provide DOI)?:** Provided after evaluation 

## _C. Description - How to access_ 

The artifact is hosted on Github at https://github.com/ FPSG-UIUC/micro24-fusemax-artifact. Following the instructions in this repository will allow you to install the relevant dependences, run the experiments, and display the graphs. System requirements can be found at https://github.com/FPSG-UIUC/micro24-fusemax-artifact/ blob/main/README.md#system-requirements. 

## _D. Installation_ 

Installation instructions can be found at https: //github.com/FPSG-UIUC/micro24-fusemax-artifact/blob/ main/README.md#installation. 

## _E. Evaluation_ 

Evaluation instructions can be found at https: //github.com/FPSG-UIUC/micro24-fusemax-artifact/blob/ main/README.md#run-experiments. 

- [1] “Tensor network contractions,” ser. Lecture Notes in Physics, vol. 964. Springer Cham, 2020. 

- [2] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard _et al._ , “Tensorflow: a system for largescale machine learning,” in _OSDI’16_ . 

- [3] AWS. (2024) Trainium architecture. [Online]. Available: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/ arch/neuron-hardware/trainium.html 

- [4] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: a framework for self-supervised learning of speech representations,” in _NIPS ’20_ . 

- [5] J. S. Bridle, “Probabilistic interpretation of feedforward classification network outputs, with relationships to statistical pattern recognition,” in _NATO Neurocomputing_ , 1989. 

- [6] S. Chen, S. Huang, S. Pandey, B. Li, G. R. Gao, L. Zheng, C. Ding, and H. Liu, “E.t.: Re-thinking self-attention for transformer models on gpus,” in _SC’21_ . 

- [7] T. Chen, Z. Du, N. Sun, J. Wang, C. Wu, Y. Chen, and O. Temam, “Diannao: A small-footprint high-throughput accelerator for ubiquitous machine-learning,” _ACM Sigplan Notices_ , 2014. 

- [8] Y. Chen, Y. Xie, L. Song, F. Chen, and T. Tang, “A survey of accelerator architectures for deep neural networks,” _Engineering_ , 2020. 

- [9] Y.-H. Chen, J. Emer, and V. Sze, “Eyeriss: A spatial architecture for energy-efficient dataflow for convolutional neural networks,” in _ISCA’16_ . 

- [10] ——, “Eyeriss v2: A flexible and high-performance accelerator for emerging deep neural networks,” in _ArXiv_ , 2018. 

- [11] Y. Chen, T. Luo, S. Liu, S. Zhang, L. He, J. Wang, L. Li, T. Chen, Z. Xu, N. Sun, and O. Temam, “Dadiannao: A machine-learning supercomputer,” in _MICRO’14_ . 

- [12] J. Choi, H. Li, B. Kim, S. Hwang, and J. H. Ahn, “Accelerating transformer networks through recomposing softmax layers,” in _IISWC’22_ . 

- [13] A. Conneau and G. Lample, “Cross-lingual language model pretraining,” in _NIPS’19_ . 

- [14] T. Dao, “Flashattention-2: Faster attention with better parallelism and work partitioning,” in _ArXiv_ , 2023. 

- [15] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. R´e, “Flashattention: Fast and memory-efficient exact attention with io-awareness,” in _ArXiv_ , 2022. 

- [16] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “Llm.int8(): 8-bit matrix multiplication for transformers at scale,” in _ArXiv_ , 2022. 

- [17] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in _NAACL’19_ . 

- [18] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in _ICLR’21_ . 

- [19] I. S. Duff, M. A. Heroux, and R. Pozo, “An overview of the sparse basic linear algebra subprograms: The new standard from the BLAS technical forum,” in _TOMS’02_ . 

- [20] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “GPTQ: Accurate post-training compression for generative pretrained transformers,” in _ArXiv_ , 2022. 

- [21] M. Gilbert, Y. N. Wu, A. Parashar, V. Sze, and J. S. Emer, “Looptree: Enabling exploration of fused-layer dataflow accelerators,” in _ISPASS’23_ . 

1472 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

- [22] C. Guo, J. Tang, W. Hu, J. Leng, C. Zhang, F. Yang, Y. Liu, M. Guo, and Y. Zhu, “Olive: Accelerating large language models via hardwarefriendly outlier-victim pair quantization,” in _ISCA’23_ . 

- [23] C. Hong, Q. Huang, G. Dinh, M. Subedar, and Y. S. Shao, “DOSA: Differentiable model-based one-loop search for DNN accelerators,” in _MICRO’23_ . 

- [24] C. Hooper, S. Kim, H. Mohammadzadeh, M. W. Mahoney, Y. S. Shao, K. Keutzer, and A. Gholami, “Kvquant: Towards 10 million context length llm inference with kv cache quantization,” in _ArXiv_ , 2024. 

- [25] O. Hsu, M. Strange, R. Sharma, J. Won, K. Olukotun, J. S. Emer, M. A. Horowitz, and F. Kjølstad, “The sparse abstract machine,” in _ASPLOS’23_ . 

- [26] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” _TASLP’21_ . 

- [27] N. P. Jouppi, C. Young, N. Patil, D. Patterson, G. Agrawal, R. Bajwa, S. Bates, S. Bhatia, N. Boden, A. Borchers, R. Boyle, P.-l. Cantin, C. Chao, C. Clark, J. Coriell, M. Daley, M. Dau, J. Dean, B. Gelb, T. V. Ghaemmaghami, R. Gottipati, W. Gulland, R. Hagmann, C. R. Ho, D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz, A. Jaffey, A. Jaworski, A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar, S. Lacy, J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin, G. MacKean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan, R. Narayanaswami, R. Ni, K. Nix, T. Norrie, M. Omernick, N. Penukonda, A. Phelps, J. Ross, M. Ross, A. Salek, E. Samadiani, C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Steinberg, A. Swing, M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan, R. Walter, W. Wang, E. Wilcox, and D. H. Yoon, “In-datacenter performance analysis of a tensor processing unit,” in _ISCA’17_ . 

- [28] S.-C. Kao, S. Subramanian, G. Agrawal, A. Yazdanbakhsh, and T. Krishna, “Flat: An optimized dataflow for mitigating attention bottlenecks,” in _ASPLOS’23_ . 

- [29] H. Kwon, P. Chatarasi, M. Pellauer, A. Parashar, V. Sarkar, and T. Krishna, “Understanding reuse, performance, and hardware cost of DNN dataflow: A data-centric approach,” in _MICRO’19_ . 

- [30] C. L. Lawson, R. J. Hanson, D. R. Kincaid, and F. T. Krogh, “Basic linear algebra subprograms for fortran usage,” in _TOMS’79_ . 

- [31] H. Le, L. Vial, J. Frej, V. Segonne, M. Coavoux, B. Lecouteux, A. Allauzen, B. Crabb´e, L. Besacier, and D. Schwab, “Flaubert: Unsupervised language model pre-training for french,” in _ArXiv_ , 2019. 

- [32] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “Awq: Activation-aware weight quantization for llm compression and acceleration,” in _MLSys’24_ . 

- [33] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in _ICCV’21_ . 

- [34] L. Lu, Y. Jin, H. Bi, Z. Luo, P. Li, T. Wang, and Y. Liang, “Sanger: A co-design framework for enabling sparse attention using reconfigurable architecture,” in _MICRO’21_ . 

- [35] N. Nayak, T. Odemuyiwa, S. Ugare, C. W. Fletcher, M. Pellauer, and J. Emer, “Teaal: A declarative framework for modeling sparse tensor accelerators,” in _MICRO’23_ . 

- [36] P. Nilsson, A. U. R. Shaik, R. Gangarajaiah, and E. Hertz, “Hardware implementation of the exponential function using taylor series,” in _NORCHIP’14_ . 

- [37] T. Norrie, N. Patil, D. H. Yoon, G. Kurian, S. Li, J. Laudon, C. Young, N. Jouppi, and D. Patterson, “The design process for google’s training chips: Tpuv2 and tpuv3,” _IEEE Micro_ , 2021. 

- [38] T. O. Odemuyiwa, H. Asghari-Moghaddam, M. Pellauer, K. Hegde, P.A. Tsai, N. Crago, A. Jaleel, J. D. Owens, E. Solomonik, J. Emer, and C. Fletcher, “Accelerating sparse data orchestration via dynamic reflexive tiling,” in _ASPLOS’23_ . 

      - L. Fang, J. Bai, and S. Chintala, “Pytorch: an imperative style, highperformance deep learning library,” in _NIPS’19_ . 

   - [43] M. Pellauer, J. Clemons, V. Balaji, N. C. Crago, A. Jaleel, D. Lee, M. O’Connor, A. Parashar, S. Treichler, P. Tsai, S. W. Keckler, and J. S. Emer, “Symphony: Orchestrating sparse and dense tensors with hierarchical heterogeneous processing,” in _TOCS’23_ . 

   - [44] S. Pichai and D. Hassabis. (2024) Our next-generation model: Gemini 1.5. [Online]. Available: https://blog.google/technology/ai/ google-gemini-next-generation-model-february-2024/#context-window 

   - [45] R. Prabhakar, Y. Zhang, D. Koeplinger, M. Feldman, T. Zhao, S. Hadjis, A. Pedram, C. Kozyrakis, and K. Olukotun, “Plasticine: A reconfigurable architecture for parallel paterns,” in _SIGARCH Computer Architecture News_ , 2017. 

   - [46] Z. Qu, L. Liu, F. Tu, Z. Chen, Y. Ding, and Y. Xie, “Dota: detect and omit weak attentions for scalable transformer acceleration,” in _ASPLOS ’22_ . 

   - [47] M. N. Rabe and C. Staats, “Self-attention does not need _o_ ( _n_[2] ) memory,” in _ArXiv_ , 2022. 

   - [48] A. Radford and K. Narasimhan, “Improving language understanding by generative pre-training,” 2018. [Online]. Available: https://api. semanticscholar.org/CorpusID:49313245 

   - [49] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” in _JMLR’20_ . 

   - [50] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao, “Flashattention-3: Fast and accurate attention with asynchrony and lowprecision,” in _ArXiv_ , 2024. 

   - [51] V. Sze, Y. Chen, T. Yang, and J. S. Emer, _Efficient Processing of Deep Neural Networks_ , ser. Synthesis Lectures on Computer Architecture. Springer, 2020. 

   - [52] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in _NIPS’17_ . 

   - [53] H. Wang, Z. Zhang, and S. Han, “Spatten: Efficient sparse attention architecture with cascade token and head pruning,” in _HPCA’21_ . 

   - [54] J. Wang, L. Guo, and J. Cong, “Autosa: A polyhedral compiler for highperformance systolic arrays on fpga,” in _FPGA’21_ . 

   - [55] J. Won, C. Hong, C. Mendis, J. Emer, and S. Amarasinghe, “Unified convolution framework: A compiler-based approach to support sparse convolutions,” in _MLSys’23_ . 

   - [56] Y. N. Wu, J. S. Emer, and V. Sze, “Accelergy: An architecture-level energy estimation methodology for accelerator designs,” in _ICCAD’19_ . 

   - [57] Y. N. Wu, P. Tsai, S. Muralidharan, A. Parashar, V. Sze, and J. S. Emer, “HighLight: Efficient and flexible DNN acceleration with hierarchical structured sparsity,” in _MICRO’23_ . 

   - [58] Y. N. Wu, P.-A. Tsai, A. Parashar, V. Sze, and J. S. Emer, “Sparseloop: An analytical approach to sparse tensor accelerator modeling,” in _MICRO’22_ . 

   - [59] J. Xia, W. Fu, M. Liu, and M. Wang, “Low-latency bit-accurate architecture for configurable precision floating-point division,” in _Applied Sciences_ , 2021. 

   - [60] A. H. Zadeh, I. Edo, O. M. Awad, and A. Moshovos, “Gobo: Quantizing attention-based nlp models for low latency and energy efficient inference,” in _MICRO’20_ . 

   - [61] G. Zhang, N. Attaluri, J. S. Emer, and D. Sanchez, “Gamma: Leveraging gustavson’s algorithm to accelerate sparse matrix multiplication,” in _ASPLOS’21_ . 

   - [62] S. Zheng, S. Chen, S. Gao, L. Jia, G. Sun, R. Wang, and Y. Liang, “Tileflow: A framework for modeling fusion dataflow via tree-based analysis,” in _MICRO’23_ . 

- [39] T. O. Odemuyiwa, J. S. Emer, and J. D. Owens, “The EDGE language: Extended general einsums for graph algorithms,” in _ArXiv_ , 2024. 

- [40] A. Parashar, M. Pellauer, M. Adler, B. Ahsan, N. Crago, D. Lustig, V. Pavlov, A. Zhai, M. Gambhir, A. Jaleel, R. Allmon, R. Rayess, S. Maresh, and J. Emer, “Efficient spatial processing element control via triggered instructions,” _IEEE Micro_ , 2014. 

- [41] A. Parashar, P. Raina, Y. S. Shao, Y.-H. Chen, V. A. Ying, A. Mukkara, R. Venkatesan, B. Khailany, S. W. Keckler, and J. Emer, “Timeloop: A systematic approach to dnn accelerator evaluation,” in _ISPASS’19_ . 

- [42] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. K¨opf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, 

1473 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:16:29 UTC from IEEE Xplore.  Restrictions apply. 

