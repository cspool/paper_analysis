# <span id="page-10-0"></span>A. Detailed Experimental Settings

Benchmark Details. We evaluate V2Drop on various multi-modal understanding benchmarks detailed as follows:

- GQA [\[15\]](#page-8-15) comprises scene graphs, questions, and images, designed to test visual scene understanding and multi-aspect image reasoning capabilities.
- MMBench [\[29\]](#page-9-21) evaluates models through a three-level hierarchical structure with 20 specific ability dimensions, enabling comprehensive assessment of perception and reasoning capabilities.
- MME [\[11\]](#page-8-16) comprises 14 subtasks evaluating perceptual and cognitive abilities through manually constructed instruction-answer pairs, mitigating data leakage issues.
- POPE [\[20\]](#page-8-17) evaluates object hallucination through binary questions about object presence, using accuracy, recall, precision, and F1 metrics across three sampling strategies.
- ScienceQA [\[30\]](#page-9-22) spans natural, language, and social sciences with hierarchical categorization, evaluating multimodal understanding and multi-step reasoning capabilities.
- TextVQA [\[34\]](#page-9-23) evaluates models' ability to read and reason about text within images through visual questionanswering tasks requiring integrated textual understanding.
- AI2D [\[16\]](#page-8-18)comprises 5,000 scientific diagrams with accompanying questions that test visual-spatial reasoning capabilities across educational content.
- MMStar [\[4\]](#page-8-19)provides 12,000 high-resolution images designed to evaluate spatial, temporal, and commonsense reasoning across multimodal understanding tasks.
- MVBench [\[18\]](#page-8-20) defines 20 video understanding tasks that require deep comprehension of temporal dimensions, beyond single-frame analysis.
- VideoMME [\[12\]](#page-8-21) comprises 900 videos and 2,700 multiple-choice questions across six domains, with durations from 11 seconds to 1 hour, categorized into short, medium, and long subsets.

Baseline Models. The baseline LVLMs, as follows:

- LLaVA-1.5 [\[23\]](#page-8-1) enhances multimodal understanding by scaling visual instruction tuning with academic-taskoriented datasets and improved training recipes. It incorporates a two-stage training approach that first aligns vision and language representations, then fine-tunes on diverse instruction-following data, achieving strong performance on visual reasoning, OCR, and multimodal dialogue tasks across various benchmarks.
- Qwen2-VL [\[36\]](#page-9-0) enhances multimodal perception by investigating scaling laws for vision-language models. By scaling model size (2B, 8B, and 72B parameters) and training data, it achieves competitive performance across diverse tasks. It supports any resolution input, enabling superior performance on document parsing, OCR, visual reasoning, and video understanding while maintaining strong text-image alignment.
- LLaVA-OneVision [\[17\]](#page-8-0) unifies single-image, multiimage, and video tasks in a single model. It represents videos as long visual token sequences in the same "interleaved" format used for images, enabling smooth task transfer from images to videos and facilitating strong zero-shot video understanding capabilities.

Comparison Methods. We provide detailed introductions and comparisons of existing token compression methods mentioned in the main text, as follows:

- ToMe [\[3\]](#page-8-22) merges similar tokens in visual transformer layers through lightweight matching techniques, achieving acceleration without requiring additional training.
- LLaVA-PruMerge [\[31\]](#page-9-14) combines pruning and merging strategies by dynamically removing less important tokens using CLS-patch attention and clustering retained tokens based on key similarity.
- FastV [\[5\]](#page-8-6) focuses on early-stage token pruning by leveraging attention maps, effectively reducing computational overhead in the initial layers.
- DART [\[39\]](#page-9-4) introduces a duplication-aware token pruning approach that selects tokens based on their redundancy relative to pivot tokens rather than importance scores.
- HiRED [\[1\]](#page-8-5) allocates token budgets across image partitions based on CLS token attention, followed by the selection of the most informative tokens within each partition, ensuring spatially aware token reduction.
- PDrop [\[42\]](#page-9-6) adopts a progressive token-dropping strategy across model stages, forming a pyramid-like token structure that balances efficiency and performance.

<span id="page-11-3"></span>

| Benchmark        | Vanilla | Pruned Layers Selection |           |           |           |           |           |       | Other Methods |       |  |
|------------------|---------|-------------------------|-----------|-----------|-----------|-----------|-----------|-------|---------------|-------|--|
| Denemial K vaima | Vanna   | (4,14,30)               | (3,14,29) | (3,15,27) | (3,16,24) | (3,17,22) | (2,16,21) | FastV | SparseVLM     | PDrop |  |
| GQA              | 61.9    | 57.8                    | 58.6      | 58.5      | 58.8      | 58.5      | 57.9      | 52.7  | 57.1          | 57.6  |  |
| SQA              | 69.5    | 68.9                    | 69.1      | 69.3      | 69.1      | 69.3      | 69.5      | 67.3  | 68.8          | 68.7  |  |
| POPE             | 85.9    | 86.3                    | 85.0      | 85.1      | 85.0      | 85.1      | 83.9      | 64.8  | 82.3          | 83.6  |  |
| MME              | 1862    | 1753                    | 1826      | 1847      | 1813      | 1826      | 1759      | 1612  | 1766          | 1721  |  |
| MMB              | 64.6    | 63.2                    | 63.4      | 63.7      | 63.5      | 63.7      | 62.8      | 61.2  | 63.2          | 62.5  |  |
| TextVQA          | 58.2    | 53.1                    | 54.0      | 54.8      | 55.2      | 55.6      | 54.8      | 52.5  | 56.1          | 56.1  |  |
| Avg. (%)         | 100.0%  | 96.0%                   | 97.0%     | 97.5%     | 97.3%     | 97.6%     | 96.2%     | 88.2% | 96.0%         | 95.8% |  |

Table 6. **Supplementary results on pruned layers selection.** Performance with 192 retained tokens on LLaVA-1.5-7B across datasets. The notation (a, b, c) represents a three-stage pruning strategy with token reduction applied at the a-th, b-th, and c-th layers, respectively.

<span id="page-11-1"></span>

| Methods           | Throughput (item/s) |                     |                     |                     |  |  |  |  |
|-------------------|---------------------|---------------------|---------------------|---------------------|--|--|--|--|
| Wicthods          | MME                 | GQA                 | MMBench             | SQA                 |  |  |  |  |
| LLaVA-1.5-7B      | 8.02                | 7.5                 | 7.13                | 6.9                 |  |  |  |  |
| FastV             | 9.46(1.18×)         | 8.68(1.16×)         | 8.65(1.21×)         | 8.14(1.18×)         |  |  |  |  |
| Cosine Similarity | $9.95(1.24 \times)$ | $9.13(1.21\times)$  | $8.90(1.25 \times)$ | $8.14(1.18 \times)$ |  |  |  |  |
| L1 Norm           | 10.16 (1.27×)       | $9.23(1.23\times)$  | $9.01(1.26 \times)$ | $8.49(1.23 \times)$ |  |  |  |  |
| L2 Norm           | 10.11(1.26×)        | $9.18(1.22 \times)$ | $9.01(1.26 \times)$ | $8.42(1.22 \times)$ |  |  |  |  |

Table 7. Supplementary results on variation metric selection. Throughput with 128 retained tokens on LLaVA-1.5-7B across datasets. The notation (N $\times$ ) represents an N-fold throughput improvement compared to the baseline model LLaVA-1.5-7B.

<span id="page-11-4"></span>

| Methods              | Performance |      |         |      |         |  |  |
|----------------------|-------------|------|---------|------|---------|--|--|
| Methods              | MME         | POPE | MMBench | GQA  | TextVQA |  |  |
| LLaVA-1.5-7B         | 1862        | 85.9 | 64.6    | 61.9 | 58.2    |  |  |
| One-time dropping    | 1717        | 77.1 | 60.6    | 57.1 | 55.2    |  |  |
| Progressive dropping | 1826        | 85.1 | 63.7    | 58.5 | 55.9    |  |  |

Table 8. **Supplementary results on effects of progressive token dropping.** Performance with 192 retained tokens on LLaVA-1.5-7B across datasets.

- SparseVLM [46] ranks token importance using crossmodal attention and introduces adaptive sparsity ratios, complemented by a novel token recycling mechanism. Based on the abstract, here's a one-sentence summary without numbers:
- DyCoke [35] is a two-stage VideoLLM method that
  prunes similar tokens temporally and compresses lessattended visual tokens in KV cache using LLM attention
  weights. Its reliance on frame-set division and similaritybased compression limits aggressive token compression,
  and while compatible with Flash Attention [10], it requires explicit attention weights making it incompatible
  with efficient attention operators.

**Implementation Details.** Our experiments are conducted on NVIDIA A100-PCIe-80GB GPUs. The implementation was carried out in Python 3.10, utilizing PyTorch 2.1.2 and CUDA 12.1. All baseline settings follow the original paper.

<span id="page-11-2"></span>

| Methods           | Performance |      |         |      |  |  |  |
|-------------------|-------------|------|---------|------|--|--|--|
| wicthods          | MME         | GQA  | MMBench | SQA  |  |  |  |
| LLaVA-1.5-7B      | 1862        | 61.9 | 64.6    | 69.5 |  |  |  |
| FastV             | 1490        | 49.6 | 56.1    | 67.3 |  |  |  |
| Cosine Similarity | 1718        | 55.2 | 61.8    | 69.2 |  |  |  |
| L1 Norm           | 1698        | 56.1 | 60.9    | 68.7 |  |  |  |
| L2 Norm           | 1712        | 56.3 | 61.8    | 68.8 |  |  |  |

Table 9. **Supplementary results on variation metric selection.** Performance with 128 retained tokens on LLaVA-1.5-7B across datasets.

Experimental parameter details for  $V^2$ Drop. On LLaVA-1.5-7B, we conduct three-stage pruning at layers 3, 17, and 22. When retaining 192 tokens, we prune 50%, 70%, and 100% of Vision tokens at layers 3, 17, and 22. When retaining 128 tokens, we prune 72%, 75%, and 100% of Vision tokens at layers 3, 17, and 22, respectively. When retaining 64 tokens, we prune 95%, 95%, and 100% of Vision tokens at layers 3, 17, and 22, respectively.

## <span id="page-11-0"></span>**B.** Additional Experimental Results

