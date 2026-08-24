# <span id="page-2-1"></span>C. Coarse-Grained Compression: Relevant Function Selection

The coarse-grained compression aims to select high-level code chunks that are most relevant to the task instruction. This process consists of three steps:

Function-Level Chunking. We first split the source code into chunks along function or class boundaries. Functions naturally encapsulate coherent logic and exhibit strong modularity [31]. Chunking at this level ensures that retained code segments are both syntactically valid and semantically self-contained, which is essential for preserving program integrity.

**Instruction-aware Relevance Ranking.** To measure the relevance of each chunk to the task instruction, we employ an instruction-aware ranking mechanism based on approximated mutual information (1). Chunks are scored and ranked in descending order, allowing us to prioritize those most informative for the given task.

**Budget-Constrained Function Selection.** Finally, we greedily select the top-ranked chunks under a coarse-grained token budget  $B_{\rm coarse}$ , which is the division of the final token budget B by the configurable fine-grained compression ratio  $R_{\rm fine}$ . This greedy selection balances efficiency and coverage: a larger budget allows more functions to pass into the fine-grained stage, potentially improving downstream quality but at

Algorithm 1: Pseudo code of Adaptive Fine-Grained Budget Allocation

```
Input: Large functions {f1, ..., fN } with min-max
       normalized AMI scores {AMI1, ..., AMIN } and
       token counts {T1, ..., TN }; total token budget
       for large functions Blarge; baseline retention
       ratio Rbase; importance parameter β
Output: Function-wise adjusted retention rates
         {R1, ..., RN }
R ← ∅ // Initialize retention rate map
for fi ∈ {f1, ..., fN } do
   Rbiased,i ← Rbase · (1 + β · (2 × AMIi − 1)); //
     Compute biased rate
   Clamp Rbiased,i to [0, 1];
for fi ∈ {f1, ..., fN } do
   Ri ← Rbiased,i · P
                         Blarge
                       i Rbiased,j ·Tj
                                 ; // Adjust rate
return R1, ..., RN ;
```

higher computational cost, while a smaller budget accelerates processing at the risk of discarding useful code. Chunks not selected are replaced with placeholders (e.g., comment markers or ellipses), which preserve the global structure while reducing overall context length.

#### <span id="page-3-0"></span>*D. Fine-Grained Compression: Intra-Function Pruning*

After selecting relevant function-level chunks in the first stage, we apply finer-grained compression to further reduce context length while preserving critical content. This process involves three steps:

Block-Level Chunking. The main challenge in intra-function compression is pruning code without breaking internal logic. To address this, each function is segmented into smaller, semantically coherent blocks. A naive idea is to split code by whitespace lines, but such line-based heuristics often misalign with semantic boundaries. Inspired by techniques in natural language processing [\[32\]](#page-11-31), we employ a perplexity-based method to identify semantic block boundaries within code. While perplexity-based grouping has shown effectiveness in natural language segmentation, it remains under-explored in code. Consecutive lines in code often form strong semantic associations, making perplexity a useful signal. Within a semantically coherent region, perplexity tends to decrease as context accumulates [\[32\]](#page-11-31). We treat each line of code as the smallest atomic unit and group consecutive lines based on their perplexity scores, calculated as in [\(3\)](#page-2-3). When a line's perplexity exhibits a sharp local increase, exceeding that of its neighbors by at least α times of the standard deviation over all lines, we mark it as a block boundary. Such high-perplexity lines typically mark the beginning of a new block, reflecting underlying semantic or structural changes. This perplexityguided aggregation allows blocks to capture meaningful code segments while preserving the code structure.

Adaptive Budget Allocation. Functions selected in the coarse-grained stage vary in importance. Hence, applying a uniform compression ratio across all of them is suboptimal. To address this, we introduce an adaptive budget allocation mechanism that distributes the fine-grained token budget proportionally to function importance. Functions with higher AMI scores receive more token budgets, preserving greater detail, while very small functions Fsmall (shorter than five lines) are kept in full. Algorithm [1](#page-3-1) summarizes the procedure.

We first define the baseline retention ratio for large functions:

<span id="page-3-2"></span>
$$R_{\text{base}} = \frac{B - \sum_{j \in \mathcal{F}_{\text{small}}} T_j}{\sum_{k \in \mathcal{F}_{\text{large}}} T_k},\tag{4}$$

where B is the final token budget, Fsmall and Flarge represent the sets of small and large functions respectively, and T<sup>j</sup> denotes the number of tokens in function j.

For functions f1, . . . , f<sup>N</sup> selected in the coarse-grained stage, we perform min-max normalization to all AMI scores to AMInorm,i.

For each large function f<sup>i</sup> , and its normalized AMI score AMInorm,i ∈ [0, 1], a biased retention ratio is then computed as

$$R_{\text{biased},i} = R_{\text{base}} \cdot (1 + \beta \cdot (2 \times \text{AMI}_{\text{norm},i} - 1)), \quad (5)$$

where Rbase is the baseline retention ratio for large functions (Equation [4\)](#page-3-2). The importance parameter β adjusts sensitivity to importance. When the importance parameter is set to 0, there is no bias, meaning all functions are treated equally. A more positive β increases the emphasis on important functions, allocating more tokens to them. All retention rates are clamped to [0, 1] and globally rescaled so that the total number of retained tokens matches the target token budget for large functions Blarge:

$$R_i = R_{\text{biased},i} \cdot \frac{B_{\text{large}}}{\sum_j R_{\text{biased},j} \cdot T_j},\tag{6}$$

where T<sup>j</sup> represents the number of tokens in the j-th function. This adjustment preserves the relative importance between functions while ensuring the global constraint is satisfied.

Dynamic Block Selection. For each function, LongCodeZip identifies a subset of blocks to retain, aiming to maximize the total relevance within the constraints of the allocated token budget. This strategy ensures that the compressed context achieves the highest possible information density. We formulate this selection as a classic 0/1 knapsack problem: each block is treated as an item, where the value corresponds to its normalized AMI score and the weight corresponds to its token length. The detailed procedure is outlined in Algorithm [2.](#page-4-0) We employ a dynamic programming approach to compute the optimal subset of blocks that satisfies the budget constraint while maximizing the cumulative value.

#### IV. EXPERIMENTAL SETUP

