# <span id="page-14-0"></span>A BACKGROUND & RELATED WORK: KV CACHE COMPRESSION

The attention mechanism relies on three key components: the Query (Q), Key (K), and Value (V) vectors. For each token, LLM computes a d-dimensional Q vector and compares it against all stored N × d K vectors, where N is the length of the sequence processed. The result of this comparison is used to weigh the corresponding V vectors, producing the final output. Mathematically, the attention operation is defined as:

$$Attention(Q, K, V) = Softmax\left(\frac{QK^{\top}}{\sqrt{d}}\right)V$$
 (6)

LLM inference is generally divided into two phases: a prefill phase for processing input tokens and a decoding phase for generating new tokens. In decoding, each token generation reloads the entire KV Cache from previous tokens, causing time and memory inefficiencies.

KV cache compression methods fall into two categories: 'training-free' methods (using eviction and quantization without model retraining) and 'training-required' methods (designing more efficient attention structures). Our approach focuses on enhancing training-free methods for broader applicability. Eviction selectively discards less important tokens, while quantization lowers the precision of key and value states to save memory. However, both methods risk significant information loss at high compression rates—especially 2-bit quantization, which can greatly reduce accuracy.

#### A.1 KV CACHE EVICTION

Eviction methods aim to reduce KV cache memory usage in Large Language Models (LLMs) by discarding less important tokens. The early work H2O [\(Zhang et al., 2024\)](#page-11-0) selects "heavy hitter" tokens based on cumulative attention scores, though this risks evicting tokens that may become important later. Keyformer [\(Adnan et al., 2024\)](#page-9-5) improves on H2O by combining "Key Attention" with a "window attention" mechanism, retaining both historically significant and recent tokens for better accuracy. MiniCache [\(Liu et al., 2024b\)](#page-10-10) reduces memory by reusing Key and Value states across layers. This method assumes that some key and value representations are redundant across model layers and can be shared. InfLLM [\(Xiao et al., 2024\)](#page-10-11) addresses very long contexts by dividing them into blocks and retaining 'representative tokens' for block eviction decisions.

## A.2 KV CACHE QUANTIZATION

Quantization reduces storage and boosts computational speed by using fewer bits to represent values. Earlier works, like AWQ [\(Lin et al., 2023\)](#page-10-12) and Qserve [\(Lin et al., 2024\)](#page-10-13), applied 4-bit quantization to the KV cache with minimal accuracy loss. Recent methods aim to compress the KV cache further while preserving accuracy. QAQ [\(Dong et al., 2024\)](#page-9-2) dynamically adjusts the precision of the in-GPU quantized cache by offloading all original-precision KV data to CPU memory. GEAR [\(Kang](#page-10-14) [et al., 2024\)](#page-10-14) improves accuracy by storing the quantization error of the KV cache as a sparse matrix with low-rank decomposition. KiVi [\(Liu et al., 2024c\)](#page-10-6) introduces a 2-bit quantization by retaining a recent window of full-precision tokens, balancing memory efficiency and accuracy.

#### A.3 TRAINING-REQUIRED APPROACHES

An early memory-reducing attention design is Multi-Query Attention (MQA, [\(Shazeer, 2019\)](#page-10-15)), where all query heads share a single pair of key and value heads. While this reduces memory, it significantly impacts accuracy. Grouped-Query Attention (GQA, [\(Ainslie et al., 2023\)](#page-9-12)) addresses this by grouping query heads, with each group sharing the same key and value heads, preserving the generalization ability of multi-head attention while reducing KV cache size. Deepseek V2 [\(Liu](#page-10-16) [et al., 2024a\)](#page-10-16) introduces Multi-Head Latent Attention (MLA), which compresses key and value states using LoRA-based projections. To prevent disruption of position embeddings from LoRA compression, specific channels are reserved for position information only, excluding them from LoRA compression.

## B OVERVIEW OF TEST DATASETS

<span id="page-15-0"></span>Table B5: Overview of all test datasets. 'Avg len' (average length) is computed using the number of words for the English (code) datasets and the number of characters for the Chinese datasets. 'Accuracy (CLS)' refers to classification accuracy, while 'Accuracy (EM)' refers to exact match accuracy

| Task Group             | Dataset             | Avg len | Metric         | Language       | #data |
|------------------------|---------------------|---------|----------------|----------------|-------|
| Math                   | GSM8K               | 240     | Accuracy (EM)  | English        | 1319  |
|                        | NarrativeQA         | 18,409  | F1             | English        | 200   |
| Simple Decomment OA    | Qasper              | 3,619   | F1             | English        | 200   |
| Single-Document QA     | MultiFieldQA-en     | 4,559   | F1             | English        | 150   |
|                        | MultiFieldQA-zh     | 6,701   | F1             | Chinese        | 200   |
|                        | HotpotQA            | 9,151   | F1             | English        | 200   |
| Multi-Document OA      | 2WikiMultihopQA     | 4,887   | F1             | English        | 200   |
| Muiti-Document QA      | MuSiQue             | 11,214  | F1             | English        | 200   |
|                        | DuReader            | 15,768  | Rouge-L        | Chinese        | 200   |
|                        | GovReport           | 8,734   | Rouge-L        | English        | 200   |
| Summarization          | QMSum               | 10,614  | Rouge-L        | English        | 200   |
| Summarization          | MultiNews           | 2,113   | Rouge-L        | English        | 200   |
|                        | VCSUM               | 15,380  | Rouge-L        | Chinese        | 200   |
|                        | TREC                | 5,177   | Accuracy (CLS) | English        | 200   |
| Few-shot Learning      | TriviaQA            | 8,209   | F1             | English        | 200   |
| rew-shot Learning      | SAMSum              | 6,258   | Rouge-L        | English        | 200   |
|                        | LSHT                | 22,337  | Accuracy (CLS) | Chinese        | 200   |
|                        | PassageCount        | 11,141  | Accuracy (EM)  | English        | 200   |
| Synthetic Task         | PassageRetrieval-en | 9,289   | Accuracy (EM)  | English        | 200   |
|                        | PassageRetrieval-zh | 6,745   | Accuracy (EM)  | Chinese        | 200   |
| <b>Code Completion</b> | LCC                 | 1,235   | Edit Sim       | Python/C#/Java | 500   |
| Code Completion        | RepoBench-P         | 4,206   | Edit Sim       | Python/Java    | 500   |

