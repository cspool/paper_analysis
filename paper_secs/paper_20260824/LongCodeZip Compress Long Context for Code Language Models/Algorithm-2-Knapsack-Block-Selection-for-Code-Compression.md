# Algorithm 2: Knapsack Block Selection for Code Compression

```
Input: Blocks {b1, ..., bN } with min-max normalized
       AMI scores {AMI1, ..., AMIN } and token
       counts {T1, ..., TN }; token budget of this
       function Bi
                   ; user-defined preserved set P
Output: Selected blocks Bselected ⊆ {b1, ..., bN }
Bremain ← max(0, Bi −
                        P
                           j∈P Tj );
// Compute remaining budget
if Bremain = 0 then
   return P;
K ← ∅;
for i = 1 to N do
   if i /∈ P then
       add (i, Ti
                 , AMIi) to K;
Bselected ← 0/1 Knapsack DP(K, Bremain) ∪ P;
return Bselected;
```

RQ2: How does different parts of LongCodeZip contribute to the performance?

RQ3: Does LongCodeZip exhibit cross-model generalization capabilities?

RQ4: What is the efficiency benefit of LongCodeZip in downstream tasks?

#### *B. Datasets*

We evaluate our method on long code context benchmarks across three common tasks: code completion, code summarization, and code question answering. These tasks reflect practical developer needs and assess whether compressed code retains sufficient information for downstream performance. For each task, we construct prompts following the benchmark papers [\[1\]](#page-11-0), [\[21\]](#page-11-20), [\[13\]](#page-11-12). Dataset statistics are shown in Table [I.](#page-5-0)

The Long Code Completion dataset [\[1\]](#page-11-0) targets the code completion task under long-context of relevant functions. To highlight long-context difficulties, we filtered the test set to 500 Python examples with input contexts longer than 5,000 tokens. The Long Module Summarization dataset [\[21\]](#page-11-20) contains 216 examples from 43 Python repositories. To focus on the challenging long-context scenario, we also further filtered the original dataset to 139 examples that have more than 2,000 context tokens. RepoQA [\[13\]](#page-11-12) is a multilingual benchmark that contains 600 long code question answering tests across 60 repositories and 6 programming languages. It requires the model to locate and return a function within the long context using a natural language instruction, similar to a retrieval task.

#### *C. Baselines and Models*

We evaluate LongCodeZip against a variety of competitive baselines:

- 1) No Compression: The full code context is used without any compression, representing the upper bound of performance.
- 2) No Context: The model is evaluated with only the task instruction, without any context, representing the lower bound.

- 3) Random Baselines: Random Token randomly removes individual tokens, while Random Line randomly removes the whole lines of code.
- 4) Retrieval-based Methods: RAG (Sliding Window) uses fixed-size overlapping chunks, whereas RAG (Function Chunking) splits code at function boundaries. Both methods use the state-of-the-art code embedding model UniXCoderbase [\[30\]](#page-11-29), [\[29\]](#page-11-28).
- 5) Code Compression Methods: We compare against the compression components from DietCode [\[26\]](#page-11-25) and Slim-Code [\[27\]](#page-11-26). DietCode was originally implemented for Python and Java, while SlimCode supports only Java. To enable direct comparison on other benchmarks, we reproduce the SlimCode for Python with tree-sitter[2](#page-4-1) .
- 6) Text Compression Methods: We also include several state-of-the-art prompt compression methods for natural languages, including LLMLingua [\[23\]](#page-11-22), LongLLMLingua [\[33\]](#page-11-32), and LLMLingua-2 [\[34\]](#page-11-33).

All methods are evaluated on a diverse set of code LLMs, covering both earlier models like Deepseek-Coder-6.7B [\[10\]](#page-11-9) and latest models like Qwen2.5-Coder-7B [\[11\]](#page-11-10) and Seed-Coder-8B [\[12\]](#page-11-11). Specifically, we use the instruct version of these models from Huggingface[3](#page-4-2) . To further demonstrate the generalizability of LongCodeZip, we also extend our evaluation to state-of-the-art closed-source models, including GPT-4o[4](#page-4-3) and Claude-3.7-Sonnet[5](#page-4-4) .

#### *D. Evaluation Metrics*

The evaluation of LongCodeZip encompasses two primary dimensions: compression efficiency and downstream generation performance with compressed context. We report compression *Ratio* on all tasks:

$$Ratio = \frac{|C_{\text{original}}|}{|C_{\text{compressed}}|} \tag{7}$$

where |Ccompressed| and |Coriginal| denote the number of tokens in the compressed and original contexts respectively.

For the code completion task, We follow LongCoder [\[1\]](#page-11-0) to evaluate the performance of the models in terms of Exact Match (EM) and Edit Similarity (ES).

For the code summarization task, we follow [\[21\]](#page-11-20) to use a third party model GPT-4O-MINI [6](#page-4-5) to evaluate which summary better explains the code between the ground truth and the generated one. This LLM-as-Judge evaluation strategy is widely adopted in both NLP and software engineering domains [\[35\]](#page-11-34), [\[36\]](#page-11-35), as it has been demonstrated to align well with human preferences and provides more nuanced evaluation compared to traditional metrics [\[37\]](#page-11-36), [\[38\]](#page-11-37). The model chooses the better summary after reviewing both options alongside the code. To

<span id="page-4-1"></span><sup>2</sup><https://tree-sitter.github.io/tree-sitter/>

<span id="page-4-2"></span><sup>3</sup><https://huggingface.co/models>

<span id="page-4-3"></span><sup>4</sup><https://openai.com/index/hello-gpt-4o/>

<span id="page-4-4"></span><sup>5</sup><https://www.anthropic.com/news/claude-3-7-sonnet>

<span id="page-4-5"></span><sup>6</sup><https://platform.openai.com/docs/models/gpt-4o-mini>

TABLE I: Datasets used for evaluating long-context code compression.

<span id="page-5-0"></span>

| Dataset                   | # Examples | Avg. Context Len. | Avg. Ground Truth Len. | Languages                       |
|---------------------------|------------|-------------------|------------------------|---------------------------------|
| Long Code Completion      | 500        | 9,328.2           | 12.4                   | Python                          |
| Long Module Summarization | 139        | 10,809.6          | 1,758.1                | Python                          |
| Repo QA                   | 600        | 11,524.6          | 156.0                  | Python, Java, JS, Rust, Go, C++ |

avoid bias, we prompt it twice with the order reversed. We then compute *CompScore* as:

$$CompScore = \frac{1}{2} [\mathcal{P}(s_o \succ \hat{s}) + (1 - \mathcal{P}(\hat{s} \succ s_o))] \qquad (8)$$

where  $\mathcal{P}(s_o \succ \hat{s})$  is the probability that the referee model prefers the generated summary  $s_o$  over the reference  $\hat{s}$ , and  $\mathcal{P}(\hat{s} \succ s_o)$  is the probability for the reverse order.  $\mathcal{S}_{\text{comp}}$  ranges from 0 to 100, with 50 indicating equal preference.

For the code QA task, we follow [13] to evaluate the retrieval accuracy of needle functions, reporting the percentage of models that retrieve a correct match above a BLEU similarity threshold of 0.8 between the generated function  $f_o$  and the target function  $\hat{f}$ : BLEU $(\hat{f}, f_o) > 0.8$ .

#### E. Implementation Details

We tailored hyperparameters for each distinct task, consistently mirroring the generation model with our compression model. For **code completion**, which demands focused context, we set the token budget B to 2k, the fine-grained ratio ( $R_{\rm fine}$ ) to 0.8, and the importance adjustment parameter ( $\beta$ ) to 0.5. Conversely, **code summarization** necessitates understanding broader context within large modules; consequently, we increased B to 5k, reduced  $R_{\rm fine}$  to 0.3, and maintained  $\beta$  at 0.5. For the **RepoQA task**, where the objective is to precisely replicate entire functions, we set B to 2k and  $R_{\rm fine}$  to 1.0 to ensure the structural integrity of functions. These values for B,  $\beta$ , and  $R_{\rm fine}$  were determined through experiments on a small held-out set that did not overlap with the test data. All experiments were conducted on a system equipped with an Intel Xeon Gold 6254 CPU and an NVIDIA A100-80G GPU.

#### V. RESULTS

#### A. RQ1: Effectiveness on Code Compression

Tables II, III, and IV present the evaluation results of our approach on three downstream tasks, respectively. The best scores among compression methods are bolded. Across all three tasks and multiple backbone models, LongCodeZip consistently outperforms compression baselines by substantial and statistically significant margins (p < 0.001 via Wilcoxon signed-rank test on 10 repeated experiments), even when operating at comparable or stricter compression ratios.

Specifically, on the Long Code Completion task, RAG-based methods achieve higher ES and EM scores than other baseline methods, but still fall short of our approach. For instance, with Qwen2.5-Coder-7B, RAG (Function Chunking) achieves an ES score of 52.79 and an EM score of 26.00 at a 3.1× compression ratio. In contrast, our approach achieves 57.55 ES and 32.40 EM at a stricter 4.3× compression ratio, representing a 28% shorter compressed context than the RAG

TABLE II: Results on Long Code Completion

<span id="page-5-1"></span>

| Model               | Method                  | ES    | EM    | Ratio |
|---------------------|-------------------------|-------|-------|-------|
|                     | No Compression          | 57.14 | 34.40 | 1.0x  |
|                     | No Context              | 41.29 | 13.20 | -     |
|                     | Random Token            | 44.86 | 13.40 | 4.4x  |
| Deepseev Coden 6 70 | Random Line             | 50.54 | 21.20 | 4.5x  |
|                     | RAG (Sliding Window)    | 58.48 | 31.60 | 4.2x  |
| Deepseek-Coder-6.7B | RAG (Function Chunking) | 57.93 | 30.80 | 5.7x  |
|                     | LLMLingua               | 43.61 | 14.00 | 5.6x  |
|                     | LLMLingua-2             | 46.23 | 15.00 | 4.4x  |
|                     | LongLLMLingua           | 54.09 | 26.40 | 4.8x  |
|                     | DietCode                | 51.57 | 20.20 | 3.4x  |
|                     | SlimCode                | 48.84 | 19.80 | 4.5x  |
|                     | LongCodeZip             | 60.58 | 35.40 | 5.3x  |
|                     | No Compression          | 56.36 | 31.80 | 1.0x  |
|                     | No Context              | 38.14 | 9.60  | -     |
| Qwen2.5-Coder-7B    | Random Token            | 39.10 | 8.40  | 4.4x  |
|                     | Random Line             | 39.73 | 12.40 | 4.5x  |
|                     | RAG (Sliding Window)    | 50.81 | 24.60 | 2.8x  |
|                     | RAG (Function Chunking) | 52.79 | 26.00 | 3.1x  |
|                     | LLMLingua               | 21.56 | 5.40  | 3.4x  |
|                     | LLMLingua-2             | 41.29 | 12.20 | 4.4x  |
|                     | LongLLMLingua           | 23.88 | 9.00  | 3.2x  |
|                     | DietCode                | 43.91 | 13.20 | 3.4x  |
|                     | SlimCode                | 40.85 | 12.20 | 4.5x  |
|                     | LongCodeZip             | 57.55 | 32.40 | 4.3x  |
|                     | No Compression          | 64.04 | 40.20 | 1.0x  |
|                     | No Context              | 41.88 | 13.60 | -     |
|                     | Random Token            | 45.35 | 13.40 | 4.4x  |
|                     | Random Line             | 50.10 | 21.20 | 4.5x  |
|                     | RAG (Sliding Window)    | 58.51 | 32.40 | 2.8x  |
| SEED-CODER-8B       | RAG (Function Chunking) | 60.52 | 35.00 | 3.7x  |
|                     | LLMLingua               | 44.36 | 14.40 | 4.5x  |
|                     | LLMLingua-2             | 46.69 | 15.40 | 4.4x  |
|                     | LongLLMLingua           | 54.84 | 26.40 | 4.2x  |
|                     | DietCode                | 51.43 | 18.80 | 3.4x  |
|                     | SlimCode                | 50.45 | 19.80 | 4.5x  |
|                     | LongCodeZip             | 63.11 | 37.40 | 5.6x  |

method. This demonstrates that our method not only preserves more critical information for code completion but also does so with significantly greater compression efficiency.

In contrast to the code completion results, RAG-based methods do not show clear advantages over other baselines on the Long Module Summarization task. However, our approach remains the most competitive, achieving a *CompScore* of 28.01 with Deepseek-Coder-6.7B at a 2.5× compression ratio—surpassing other compression baselines by a considerable margin. This highlights the effectiveness of our method in preserving relevant semantic content for summarization, even with shorter input contexts.

On the RepoQA task, LLMLingua and LLMLingua-2 exhibit poor performance because token-level compression corrupts code syntax and structure, while LongLLMLingua improves this dramatically by performing coarse-grained document-level to fine-grained token-level compression, using instruction-aware contrastive perplexity to preserve code

<span id="page-6-0"></span>TABLE III: Results on Long Module Summarization

| Model               | Method                  | CompScore | Ratio |
|---------------------|-------------------------|-----------|-------|
|                     | No Compression          | 19.09     | 1.0x  |
|                     | No Context              | 2.49      | -     |
|                     | Random Token            | 11.88     | 1.8x  |
|                     | Random Line             | 17.62     | 1.8x  |
|                     | RAG (Sliding Window)    | 22.95     | 2.1x  |
| DEEPSEEK-CODER-6.7B | RAG (Function Chunking) | 18.47     | 2.1x  |
|                     | LLMLingua               | 17.65     | 2.1x  |
|                     | LongLLMLingua           | 21.62     | 1.7x  |
|                     | LLMLingua-2             | 18.48     | 2.1x  |
|                     | DietCode                | 17.35     | 2.1x  |
|                     | SlimCode                | 20.24     | 2.2x  |
|                     | LongCodeZip             | 28.01     | 2.5x  |
|                     | No Compression          | 56.00     | 1.0x  |
|                     | No Context              | 6.13      | -     |
| Qwen2.5-Coder-7B    | Random Token            | 34.09     | 1.8x  |
|                     | Random Line             | 46.19     | 1.8x  |
|                     | RAG (Sliding Window)    | 53.50     | 1.7x  |
|                     | RAG (Function Chunking) | 40.84     | 2.1x  |
|                     | LLMLingua               | 39.81     | 1.7x  |
|                     | LongLLMLingua           | 46.72     | 1.5x  |
|                     | LLMLingua-2             | 52.99     | 2.1x  |
|                     | DietCode                | 35.67     | 2.1x  |
|                     | SlimCode                | 44.13     | 2.2x  |
|                     | LongCodeZip             | 56.47     | 1.7x  |
|                     | No Compression          | 44.95     | 1.0x  |
|                     | No Context              | 17.42     | -     |
|                     | Random Token            | 34.16     | 1.8x  |
|                     | Random Line             | 41.27     | 1.8x  |
|                     | RAG (Sliding Window)    | 42.54     | 3.0x  |
| SEED-CODER-8B       | RAG (Function Chunking) | 43.19     | 2.1x  |
|                     | LLMLingua               | 32.00     | 3.1x  |
|                     | LongLLMLingua           | 49.73     | 2.4x  |
|                     | LLMLingua-2             | 53.88     | 3.2x  |
|                     | DietCode                | 44.74     | 2.1x  |
|                     | SlimCode                | 46.01     | 2.2x  |
|                     | LongCodeZip             | 55.07     | 3.5x  |

segments highly relevant to the instruction. Nonetheless, our approach consistently achieves the best performance across all models. Notably, on Deepseek-Coder-6.7B, our approach surpasses LongLLMLingua by 16% in overall score while compressing the context to half the length. This underscores the superior effectiveness of our method in both information retention and aggressive compression for long code understanding.

Notably, LongCodeZip demonstrates strong generalizability across state-of-the-art closed-source models. As comprehensively shown in Table V, on GPT-4o, LongCodeZip achieves an ES score of 64.72 (vs. 65.13 no-compression baseline) on Long Code Completion at a 4.3x compression ratio, closely matching the performance of the uncompressed input while significantly reducing context length. For the RepoQA task, LongCodeZip even surpasses the no-compression baseline, achieving 88.9 average score on GPT-40, demonstrating that removing irrelevant context can improve performance on complex reasoning tasks. On the more powerful Claude-3.7-Sonnet, LongCodeZip achieves 66.27 ES (vs. 66.24 baseline) with the same compression efficiency. For the RepoQA task, LongCodeZip also surpasses the no-compression baseline on Claude-3.7-Sonnet, achieving 90.7 average score, further demonstrating the effectiveness of our approach.

TABLE IV: Results on RepoQA

<span id="page-6-1"></span>

| Method                    | Py           | C++    | Java         | TS     | Rust | Go   | Avg. | Ratio        |
|---------------------------|--------------|--------|--------------|--------|------|------|------|--------------|
|                           | DEEP         | SEEK-  | Codei        | к-6.7в |      |      |      |              |
| No Compression            | 21.0         | 30.0   | 44.0         | 49.0   | 27.0 | 59.0 | 38.3 | 1.0x         |
| No Context                | 0.0          | 0.0    | 0.0          | 0.0    | 0.0  | 0.0  | 0.0  | -            |
| Random Token              | 0.0          | 1.0    | 2.0          | 1.0    | 0.0  | 6.0  | 1.7  | 3.6x         |
| Random Line               | 3.0          | 12.0   | 9.0          | 7.0    | 5.0  | 8.0  | 7.3  | 3.5x         |
| RAG (Sliding Window)      | 49.0         | 55.0   | 53.0         | 67.0   | 47.0 | 62.0 | 55.5 | 3.5x         |
| RAG (Function Chunking    |              | 40.0   | 30.0         | 36.0   | 49.0 | 57.0 | 42.3 | 4.0x         |
| LLMLingua                 | 0.0          | 2.0    | 6.0          | 1.0    | 2.0  | 4.0  | 2.5  | 3.6x         |
| LLMLingua-2               | 1.0          | 1.0    | 4.0          | 0.0    | 0.0  | 3.0  | 1.5  | 4.6x         |
| LongLLMLingua             | 52.0         | 54.0   | 65.0         | 62.0   | 56.0 | 67.0 | 59.3 |              |
| DietCode<br>SlimCode      | 13.0<br>15.0 | -      | 28.0<br>35.0 | -      | -    | -    | 20.5 | 3.7x<br>4.3x |
| LongCodeZip               | 76.0         | 69.0   | 80.0         | 75.0   | 73.0 | 79.0 | 75.3 | 5.3x         |
| Longcouczip               |              |        |              |        | 75.0 | 77.0 | 70.0 | 3.3A         |
|                           | _            | EN2.5- |              |        |      |      |      |              |
| No Compression            | 84.0         | 77.0   | 89.0         | 93.0   | 83.0 | 90.0 | 86.0 | 1.0x         |
| No Context                | 0.0          | 0.0    | 0.0          | 0.0    | 0.0  | 0.0  | 0.0  | -            |
| Random Token              | 1.0          | 3.0    | 4.0          | 2.0    | 4.0  | 7.0  | 3.5  | 3.6x         |
| Random Line               | 6.0          | 11.0   | 22.0         | 10.0   | 9.0  | 13.0 | 11.8 | 3.5x         |
| RAG (Sliding Window)      | 64.0         | 65.0   | 68.0         | 72.0   | 57.0 | 79.0 | 67.5 | 3.7x         |
| RAG (Function Chunking    |              | 47.0   | 59.0         | 39.0   | 58.0 | 69.0 | 54.3 |              |
| LLMLingua                 | 5.0          | 7.0    | 9.0          | 11.0   | 4.0  | 16.0 | 8.7  | 4.1x         |
| LLMLingua-2               | 1.0          | 2.0    | 8.0          | 1.0    | 1.0  | 4.0  | 2.8  | 4.6x         |
| LongLLMLingua<br>DietCode | 70.0<br>17.0 | 63.0   | 71.0<br>35.0 | 68.0   | 78.0 | 78.0 |      | 4.3x<br>3.7x |
| SlimCode                  | 20.0         | -      | 48.0         | -      | -    | -    |      | 3.7x<br>4.3x |
|                           |              |        |              |        |      |      |      |              |
| LongCodeZip               | 92.0         | 78.0   | 87.0         | 85.0   | 86.0 | 95.0 | 87.2 | 4.5x         |
|                           | S            | EED-C  | ODER-        | 8B     |      |      |      |              |
| No Compression            | 73.0         | 52.0   | 70.0         | 81.0   | 57.0 | 81.0 | 69.0 | 1.0x         |
| No Context                | 0.0          | 0.0    | 0.0          | 0.0    | 0.0  | 0.0  | 0.0  | -            |
| Random Token              | 2.0          | 3.0    | 4.0          | 1.0    | 1.0  | 10.0 | 3.5  | 3.6x         |
| Random Line               | 5.0          | 6.0    | 17.0         | 6.0    | 4.0  | 18.0 | 9.3  | 3.5x         |
| RAG (Sliding Window)      | 58.0         | 51.0   | 66.0         | 64.0   | 57.0 | 74.0 | 61.7 | 3.9x         |
| RAG (Function Chunking    | 49.0         | 40.0   | 50.0         | 30.0   | 47.0 | 64.0 | 46.7 | 4.5x         |
| LLMLingua                 | 4.0          | 3.0    | 9.0          | 8.0    | 5.0  | 10.0 | 6.5  | 4.3x         |
| LLMLingua-2               | 1.0          | 2.0    | 4.0          | 1.0    | 1.0  | 6.0  | 2.5  | 4.6x         |
| LongLLMLingua             | 71.0         | 60.0   | 74.0         | 65.0   | 74.0 | 83.0 |      | 5.1x         |
| DietCode                  | 16.0         | -      | 32.0         | -      | -    | -    | 24.0 | 3.7x         |
| SlimCode                  | 25.0         | -      | 50.0         | -      | -    | -    | 37.5 | 4.3x         |
| LongCodeZip               | 83.0         | 70.0   | 92.0         | 74.0   | 78.0 | 87.0 | 80.7 | 5.3x         |

We also conduct comprehensive comparisons with recent advanced approaches in code completion, including A<sup>3</sup>-CodGen [39], cAST [40], RepoGenix [41], and RLCoder [42] across all evaluated models. As shown in Table VI, Long-CodeZip consistently outperforms these advanced RAG methods across the most competitive open-source and closed-source models, SeedCoder and Claude-3.7-Sonnet. Our method can more efficiently retain essential information, achieving higher information density under the same token budget. This demonstrates the broad applicability and consistent effectiveness of our approach across diverse model architectures and capabilities. Notably, these RAG-based retrieval methods are complementary to our compression approach and could potentially be combined with our framework to further enhance performance by first retrieving relevant content and then applying our compression techniques.

Overall, our method achieves effectiveness on par with or better than the No Compression setting, and consistently outperforms all compression baselines across tasks and backbone models even under more aggressive compression.

TABLE V: Results with Closed-source Models

<span id="page-7-0"></span>

| Long Code Completion    |                          |       |       |            |                            | Long  | Long Module Summarization |                   |           | RepoQA |         |       |         |       |
|-------------------------|--------------------------|-------|-------|------------|----------------------------|-------|---------------------------|-------------------|-----------|--------|---------|-------|---------|-------|
| Method                  | CLAUDE-3.7-SONNET GPT-40 |       |       | CLAUDE-3.7 | CLAUDE-3.7-SONNET   GPT-40 |       |                           | CLAUDE-3.7-SONNET |           | GPT-   | GPT-40  |       |         |       |
|                         | ES                       | EM    | Ratio | ES         | EM                         | Ratio | CompScore                 | Ratio             | CompScore | Ratio  | Avg Acc | Ratio | Avg Acc | Ratio |
| No Compression          | 66.24                    | 41.20 | 1.0x  | 65.13      | 40.80                      | 1.0x  | 60.72                     | 1.0x              | 58.42     | 1.0x   | 89.7    | 1.0x  | 87.8    | 1.0x  |
| No Context              | 43.97                    | 14.20 | -     | 42.92      | 14.00                      | -     | 6.58                      | -                 | 6.41      | -      | 0.0     | -     | 0.0     | -     |
| Random Token            | 47.61                    | 14.00 | 4.4x  | 46.51      | 13.80                      | 4.4x  | 37.45                     | 1.8x              | 35.83     | 1.8x   | 3.8     | 3.6x  | 3.8     | 3.6x  |
| Random Line             | 52.61                    | 22.20 | 4.5x  | 51.42      | 21.80                      | 4.5x  | 50.12                     | 1.8x              | 48.24     | 1.8x   | 12.2    | 3.5x  | 12.1    | 3.5x  |
| RAG (Sliding Window)    | 61.44                    | 34.00 | 2.8x  | 60.03      | 33.20                      | 2.8x  | 58.03                     | 1.7x              | 55.85     | 1.7x   | 73.8    | 3.7x  | 73.0    | 3.7x  |
| RAG (Function Chunking) | 63.55                    | 36.80 | 3.1x  | 62.01      | 36.00                      | 3.1x  | 44.56                     | 2.1x              | 42.76     | 2.1x   | 55.0    | 4.3x  | 52.5    | 4.3x  |
| LLMLingua               | 46.58                    | 15.20 | 3.4x  | 45.53      | 14.80                      | 3.4x  | 43.21                     | 1.7x              | 41.57     | 1.7x   | 2.8     | 4.1x  | 2.7     | 4.1x  |
| LLMLingua-2             | 49.02                    | 16.20 | 4.4x  | 47.90      | 15.80                      | 4.4x  | 57.85                     | 2.1x              | 55.48     | 2.1x   | 3.0     | 4.6x  | 2.8     | 4.6x  |
| LongLLMLingua           | 57.58                    | 27.80 | 3.2x  | 56.24      | 27.20                      | 3.2x  | 50.86                     | 1.5x              | 48.89     | 1.5x   | 74.5    | 4.8x  | 73.2    | 4.8x  |
| DietCode                | 54.00                    | 19.80 | 3.4x  | 52.76      | 19.40                      | 3.4x  | 38.82                     | 2.1x              | 37.21     | 2.1x   | 26.7    | 3.7x  | 25.5    | 3.7x  |
| SlimCode                | 53.03                    | 20.80 | 4.5x  | 51.78      | 20.40                      | 4.5x  | 48.11                     | 2.2x              | 46.13     | 2.2x   | 38.3    | 4.3x  | 37.0    | 4.3x  |
| LongCodeZip             | 66.27                    | 40.20 | 4.3x  | 64.72      | 38.80                      | 4.3x  | 61.47                     | 1.7x              | 59.04     | 1.7x   | 88.9    | 5.1x  | 88.9    | 5.1x  |

<span id="page-7-1"></span>TABLE VI: Comparison with Advanced RAG Methods on Long Code Completion

| Model             | Method                 | ES    | EM    | Ratio |
|-------------------|------------------------|-------|-------|-------|
|                   | No Compression         | 64.04 | 40.20 | 1.0x  |
| SEED-CODER-8B     | A <sup>3</sup> -CodGen | 58.70 | 33.10 | 3.8x  |
|                   | cAST                   | 57.35 | 30.90 | 4.1x  |
|                   | RepoGenix              | 60.28 | 34.70 | 3.5x  |
|                   | RLCoder                | 58.14 | 32.30 | 4.0x  |
|                   | LongCodeZip            | 63.11 | 37.40 | 5.6x  |
|                   | No Compression         | 66.24 | 41.20 | 1.0x  |
|                   | A <sup>3</sup> -CodGen | 60.15 | 35.80 | 3.8x  |
| CLAUDE-3.7-SONNET | cAST                   | 58.92 | 33.60 | 4.1x  |
|                   | RepoGenix              | 62.48 | 37.40 | 3.5x  |
|                   | RLCoder                | 62.76 | 37.90 | 4.0x  |
|                   | LongCodeZip            | 66.27 | 40.20 | 4.3x  |

TABLE VII: Ablation Study Results

| Configuration                                                                                                                 | ES                                                               | EM                                                               | Ratio                        |
|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|------------------------------|
| LongCodeZip                                                                                                                   | 57.55                                                            | 32.40                                                            | 4.3x                         |
| Coarse-grained Ablations:<br>w/ Similarity-based Ranking<br>w/ Random Ranking                                                 | 49.66 (-7.89)<br>39.76 (-17.79)                                  | 25.20 (-7.20)<br>11.50 (-20.90)                                  | 4.3x<br>4.4x                 |
| Fine-grained Ablations: w/o Fine-grained Compression w/o Adaptive Budget Allocation w/ Line Chunking w/ Random Line Selection | 56.10 (-1.45)<br>55.21 (-2.34)<br>55.98 (-1.57)<br>55.07 (-2.48) | 31.20 (-1.20)<br>29.40 (-3.00)<br>31.20 (-1.20)<br>29.00 (-3.40) | 4.2x<br>4.3x<br>4.3x<br>4.3x |

