# <span id="page-4-1"></span>4 DFVG Algorithm

#### 4.1 Overview

In this section, we present the core algorithmic design of **DFVG**. The overall idea is to realize an efficient and energyfriendly speculative decoding process under the collaboration of heterogeneous hardware, through optimizations in draft generation, branch control, and verification scheduling. Fig. 5, provides an overview of **DFVG**'s speculative decoding. The FPGA draft model generates multiple candidate branches under hardware budget constraints, while the GPU verification model accepts valid tokens and rolls back invalid paths. This pipelined collaboration balances efficiency and correctness within limited resources. After Fig. 5, Alg. 1 summarizes the **DFVG** speculative decoding process, where the FPGA generates branches under resource constraints and the GPU verifies candidates in parallel. This pseudocode provides a step-by-step view of the draft-verify pipeline for later analysis.

#### 4.2 Adaptive Dynamic Allocation for Parallel Tree

**Motivation**: Traditional SpecInfer [15] employs static predefined configurations to construct token trees, which cannot dynamically adjust branching strategies according to the uncertainty of model outputs. To address this limitation, we propose a budget-constrained integer programming approach (ADAPT) that maximizes the performance of speculative decoding under limited computational resources.

**Problem Formulation**: Given a computational budget B and probability distributions of the small model at various positions, our objective is to determine the optimal token tree structure. Let  $x_{i,j,l} \in \{0,1\}$  be a binary decision variable indicating whether to select the l-th token from the vocabulary for branching at the j-th node in the i-th layer. Let  $p_{i,j,l}$  be the probability that selecting the l-th token at the j-th

<span id="page-5-1"></span>![](_page_5_Figure_2.jpeg)

**Figure 6.** Comparison of parallel decoding methods. (a) Speculated token tree. (b) Sequence-based decoding causes redundant computation. (c) Tree-based decoding leads to sparse and irregular masks. (d) Our TreeSort-Verify reorders tokens for block-parallel execution with higher efficiency.

node in the *i*-th layer passes verification by the large model, which we simply take as the confidence of the draft model.

**Optimization Objective**: Our goal is to maximize the expected number of successfully verified tokens:

$$\max \sum_{i=1}^{D} \sum_{j=1}^{N_i} \sum_{l=1}^{V} p_{i,j,l} \cdot x_{i,j,l}$$
 (6)

where D is the maximum speculative depth,  $N_i$  is the number of nodes in the i-th layer, and V is the vocabulary size.

#### **Constraints:**

① The computational budget constraint ensures that the total number of branches does not exceed available resources:

$$\sum_{i=1}^{D} \sum_{j=1}^{N_i} \sum_{l=1}^{V} x_{i,j,l} \le B \tag{7}$$

② Structural constraints limit the number of branches per layer due to hardware parallelism constraints:

$$\sum_{j=1}^{N_i} \sum_{l=1}^{V} x_{i,j,l} \le k_{\max}, \quad \forall i \in \{1, 2, \dots, D\}$$
 (8)

③ The pipeline depth constraint ensures full exploitation of heterogeneous hardware pipeline characteristics. The lower bound of depth D is determined by the computational latency ratio between the draft model (FPGA) and verification model (GPU). Let  $T_{\rm draft}$  be the single-layer draft inference latency on FPGA and  $T_{\rm verify}$  be the verification inference latency on GPU. To achieve computational overlap and maximize resource utilization, the depth lower bound should satisfy:

$$D \ge D_{\min} = \left\lceil \frac{T_{\text{verify}}}{T_{\text{draft}}} \right\rceil \tag{9}$$

