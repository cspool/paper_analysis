# <span id="page-14-0"></span>C Baselines

In this section, we briefly introduce the design principles of each baseline.

FLASHATTN. [Dao](#page-9-6) [\(2024\)](#page-9-6) introduces an attention computation method with hardware awareness that leverages the architecture of modern GPUs. FLASHATTN applies tiling techniques to the calculation of attention scores and fully utilizes the high-bandwidth memory to accelerate computation. FLASHATTN is an accurate attention method that preserves the computation results, primarily focusing on accelerating attention on a single GPU.

MINFERENCE. As introduced in [Jiang et al.](#page-10-8) [\(2024\)](#page-10-8), three approximate attention patterns can be utilized to approximate FULLATTN computation. MINFERENCE first searches for the head configuration by assigning an approximate pattern to each head. During inference, only a limited number of attention score entries are calculated, accelerating the prefill by reducing the computation in attention. MINFERENCE focuses on applying approximate attention without sequence parallelism.

RINGATTN. Introduced by [Li et al.](#page-10-6) [\(2023\)](#page-10-6), RINGATTN distributes the context across multiple GPUs (hosts). Online softmax is applied to the attention calculation, where each host computes the partial attention score passed by the previous host, using local context for H −1 rounds (H is the number of hosts). The core idea is to overlap the communication of passing partial attention scores with the attention calculation. RINGATTN is primarily used for training LLMs with extremely long contexts but is not optimized for inference.

ULYSSES. Similar to RINGATTN, ULYSSES [\(Ja](#page-9-4)[cobs et al.,](#page-9-4) [2023\)](#page-9-4) was also introduced for longcontext training. ULYSSES applies sequence parallelism, with each host holding a partial context. During attention calculation, three AlltoAll communications on Q, K, and V are performed to distribute the full context of specific heads to the corresponding host. After the calculation, another AlltoAll communication is conducted on the attention score to revert the distribution from splitting heads to splitting the sequence.

STARATTN. Introduced by [Acharya et al.](#page-9-5) [\(2024\)](#page-9-5), STARATTN is a pioneer in combining approximate attention with sequence parallelism. STARATTN introduces *anchor blocks*, which contain the initial tokens of the input sequence and are the same size as the blocks on each host. Each host calculates only the partial attention between the anchor block and the current context block, with no communication required. Although STARATTN reduces communication and synchronization, its large anchor blocks introduce overhead in the FFN,

limiting the speedup gains.

Here, we present the computation per forward call for each method in Table 9. Notably, since FLASHATTN, RINGATTN, and ULYSSES are FULLATTN methods, they share the same computation formula. We do not include MINFERENCE, as its computation depends on the search results of the head configurations. We provide a visualization of Table 9 in Figure 4(c).

#### <span id="page-15-0"></span>D Task Abbreviations

<span id="page-15-3"></span>Here, we provide a mapping between the benchmark task abbreviations and their full names used in the main text in Table 10 and 11.

| Abbr.     | Full Name        |
|-----------|------------------|
| R.PassKey | Retrieve.PassKey |
| R.Number  | Retrieve.Number  |
| R.KV      | Retrieve.KV      |
| E.Sum     | En.Sum           |
| E.QA      | En.QA            |
| E.Dia     | En.Dia           |
| Z.QA      | Zh.QA            |
| C.Debug   | Code.Debug       |
| M.Find    | Math.Find        |
|           |                  |

<span id="page-15-4"></span>Table 10: Task name mapping of  $\infty$ Bench.

| Abbr.      | Full Name                 |
|------------|---------------------------|
| SG1        | Single NIAH 1             |
| SG2        | Single NIAH 2             |
| SG3        | Single NIAH 3             |
| MK1        | Multi-keys NIAH 1         |
| MK2        | Multi-keys NIAH 2         |
| MK3        | Multi-keys NIAH 3         |
| MV         | Multi-values NIAH         |
| MQ         | Multi-queries NIAH        |
| VT         | Variable Tracking         |
| CWE        | Common Words Extraction   |
| <b>FWE</b> | Frequent Words Extraction |
| QA1        | Question Answering 1      |
| QA2        | Question Answering 2      |

Table 11: Task name mapping of RULER.

