# <span id="page-7-0"></span>5 Results

In this section, we validate the following quality and efficiency claims for JRT-RNN:

- 1. In-context learning (ICL) quality JRT-RNN provides 99% of Transformer quality at 360M params./30Bn tokens, averaged across the recall-intensive ICL benchmarks. This represents 46.7% improvement over Based and 78.8% over Mamba. JRT-RNN provides 96% of Transformer quality at 1.3Bn params./ 50Bn tokens, representing 16.2% improvement over Based and 34.5% over Mamba on average.
- 2. Overall language modeling Beyond outperforming in recall, we show that JRT-RNN matches the baselines in general natural language understanding (SuperGLUE). We give a detailed analysis of the pretrained LMs, comparing perplexity on slices of the Pile test set to show the strengths and limitations.
- 3. Generation We show that JRT-RNN can provide 19.2× higher prefill throughput than FlashAttention-2 at 32k sequence length, batch size 16 on an NVidia H100 GPU.

Models. We compare JRT-RNN to two state-of-the-art recurrent autoregressive models, Based [\[7\]](#page-12-6) and Mamba [\[1\]](#page-12-0). We also compare to the Transformer++ (Llama architecture [\[32\]](#page-14-0)), which adds rotary encodings [\[54\]](#page-15-6) and gated linear units.

For JRT-RNN, we start from the Based linear recurrent architecture, since it has been shown in prior work to outperform prior sub-quadratic architectures (e.g., Mamba, GLA) at recall. An extended explanation of Based is in Appendix [D.](#page-28-0) We reiterate that the approaches in JRT-Prompt and JRT-RNN can be combined with any linear recurrent model.

Benchmarks. We evaluate on a range of ICL benchmarks. We use SuperGLUE to test general language understanding [\[55\]](#page-15-7). We next evaluate on a suite of recall-intensive tasks including: SWDE and FDA information extraction tasks [\[7,](#page-12-6) [29,](#page-13-11) [56,](#page-15-8) [57\]](#page-15-9), where the model needs to extract values for a specified attribute from in-context passages, and SQUADv2 [\[58\]](#page-15-10), Natural Questions [\[59\]](#page-15-11), TriviaQA [\[60\]](#page-15-12), and Drop [\[61\]](#page-15-13). In these tasks, the model needs to ground its answers in in-context documents. We release code and models to reproduce our results and provide details on the benchmarks and evaluations in Appendix [B.](#page-21-1)

<span id="page-8-0"></span>

|              |                                | FI    | OΑ    | $\mathbf{sw}$ | $\mathbf{DE}$ | N     | Q     | $\mathbf{SQUAD}$ | Trivia | $\mathbf{Drop}$ | Avg.        |
|--------------|--------------------------------|-------|-------|---------------|---------------|-------|-------|------------------|--------|-----------------|-------------|
| Architecture | Param/Tok                      | 512   | 1024  | 512           | 1024          | 512   | 1024  | Full             | Full   | Full            |             |
|              |                                | Acc ↑ | Acc ↑ | Acc ↑         | Acc ↑         | Acc ↑ | Acc ↑ | Acc ↑            | Acc ↑  | Acc ↑           | Acc ↑       |
| Transformer  | 360M/30B                       | 74.8  | 73.0  | 44.7          | 43.0          | 27.8  | 22.9  | 36.2             | 46.5   | 21.8            | 43.4        |
| Mamba        | 360M/30B                       | 41.1  | 24.3  | 22.2          | 13.6          | 16.4  | 12.5  | 25.5             | 43.0   | 17.3            | 24.0        |
| Based        | 360M/30B                       | 50.3  | 35.8  | 30.4          | 21.6          | 19.7  | 14.7  | 29.8             | 42.5   | 18.4            | 29.2        |
| JRT-RNN      | $360 \mathrm{M}/30 \mathrm{B}$ | 82.0  | 66.0  | 43.3          | 35.1          | 32.9  | 16.2  | 41.7             | 43.2   | 25.8            | 42.9        |
| Transformer  | 1.3B/10B                       | 75.3  | 71.5  | 41.6          | 41.0          | 29.6  | 25.8  | 38.7             | 48.8   | 22.6            | 43.9        |
| Mamba        | 1.3 B/10 B                     | 37.4  | 23.3  | 23.0          | 15.1          | 19.6  | 16.1  | 26.1             | 45.7   | 20.9            | 25.2        |
| Based        | 1.3B/10B                       | 66.3  | 49.0  | 32.3          | 26.3          | 19.7  | 15.7  | 30.7             | 44.2   | 19.1            | 33.7        |
| JRT-RNN      | 1.3 B/10 B                     | 78.5  | 60.6  | 38.5          | 32.7          | 26.5  | 16.7  | 51.6             | 44.8   | 28.4            | 42.0        |
| Transformer  | 1.3B/50B                       | 85.6  | 83.5  | 55.7          | 56.0          | 33.4  | 29.9  | 40.1             | 56.6   | 21.4            | 51.4        |
| Mamba        | 1.3 B/50 B                     | 55.4  | 40.1  | 44.0          | 33.7          | 27.6  | 23.2  | 32.2             | 54.5   | 20.7            | 36.8        |
| Based        | 1.3 B/50 B                     | 69.3  | 58.8  | 47.6          | 40.4          | 29.1  | 24.4  | 38.5             | 54.3   | 20.8            | 42.6        |
| JRT-RNN      | 1.3 B/50 B                     | 86.7  | 67.7  | 49.4          | 45.7          | 38.3  | 25.4  | 50.4             | 53.0   | 29.3            | <u>49.5</u> |

<span id="page-8-1"></span>Table 2: **Evaluation of JRT-RNN models.** We compare JRT-RNN to strong LMs proposed in prior work (Based, Mamba, and Transformer++) across parameter scales. In the table, we specify the length (number of tokens) of the documents provided in context (512, 1024, Full), where "Full" means the full document is included as prefill. Table 7 contains the average number of tokens per document in each benchmark.

| Arch.       | Param/Tokens | FDA<br>2k | $\begin{array}{c} \mathbf{SWDE} \\ 2\mathbf{k} \end{array}$ | <b>NQ</b><br>2k |
|-------------|--------------|-----------|-------------------------------------------------------------|-----------------|
| Transformer | 360M/10B     | 65.2      | 41.0                                                        | 23.0            |
| Mamba       | 360 M/10 B   | 12.4      | 13.4                                                        | 12.4            |
| Based       | 360 M/10 B   | 19.1      | 18.9                                                        | 13.9            |
| JRT-RNN     | 360 M/10 B   | 28.4      | 26.1                                                        | 15.4            |
| Transformer | 1.3B/50B     | 79.7      | 55.5                                                        | 30.2            |
| Mamba       | 1.3 B/50 B   | 21.0      | 29.9                                                        | 23.1            |
| Based       | 1.3 B / 50 B | 36.1      | 37.7                                                        | 23.4            |
| JRT-RNN     | 1.3 B / 50 B | 55.2      | 41.4                                                        | 26.2            |

Table 3: Evaluation at prefill lengths 2k, i.e. beyond the encoder region (length M=1024).

| Inference       | Param/Tokens               | <b>FDA</b> 512 | $\begin{array}{c} \textbf{SWDE} \\ 512 \end{array}$ | <b>NQ</b> 512 |
|-----------------|----------------------------|----------------|-----------------------------------------------------|---------------|
| Left-pad        | 360 M/30 B                 | 61.9           | 38.1                                                | 24.6          |
| Read-2 $\times$ | 360M/30B                   | 82.0           | 43.3                                                | 32.9          |
| Iterate         | 360 M/30 B                 | 76.3           | 40.7                                                | 29.2          |
| Left-pad        | 1.3 B/50 B                 | 75.8           | 49.3                                                | 30.9          |
| Read-2×         | 1.3 B/50 B                 | 86.7           | 49.4                                                | 38.3          |
| Iterate         | $1.3 {\rm B} / 50 {\rm B}$ | 80.2           | 43.3                                                | 34.2          |

Table 4: JRT-RNN with alternate inference strategies when l < M, for prefill and encoder lengths l and M.

#### 5.1 In-context learning quality

In Table 2, we find JRT-RNN outperforms the decoder-only baseline (Based) by 13.7 points at 360M parameters (30Bn tokens) and 6.9 points at 1.3B parameters (50Bn tokens) on average. JRT-RNN closes the gap to Transformer++ to within 0.5 points on average at 360M and 1.9 points on average at 1.3B parameters.

In Table 2, we left pad documents with length  $\langle M \rangle$ , where M=1024 is the encoder region's length during training (discussed in Section 4) – for the three results with length 512 documents we pad using JRT-Prompt and otherwise with the tokenizer's space token (discussed further below).

**Length extrapolation.** Though the encoder processes until length M = 1024 for our trained LMs, we excitingly find that the benefits of JRT extend to prefill lengths l s.t. l > M as well. In Section 5.1, we evaluate at the 360M and 1.3B parameter scales with documents of length 2000.

**Inference strategies.** In Table 3, we compare alternate inference strategies for JRT-RNN in the regime where the prefill length l is less than the encoder length M, l < M:

- **Decoding with padding**: We left-pad the prefill to length M to match the training distribution the model sees. Causal decoding starts at position M. This is the default for JRT-RNN.
- Read-twice pad: Instead of padding with a special token, we can "pad" by repeating the context (i.e., JRT-PROMPT). We use this at l = 512 for FDA, SWDE, and NQ in Table 2. Padding is a fixed cost for JRT-RNN, so it can be used creatively.
- Iterative encoding: We allow the model to non-causally view its previously generated tokens during decoding. We generate token  $y_l$  given the length l prefill, append it to the prefill, and then compute  $y_{l+1}$

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 3: **Perplexity slices.** We slice the Pile test set perplexities of the pretrained LMs into associative recall "AR" and non-recall "Other" slices. A token is an AR token if it corresponds to a bigram that is re-occurring in the context, since the LM can look to the prior occurrence to predict the next token (Def. in Section 5.2). **Top left** (**recall frequencies**) We plot y perplexity on AR bigram tokens that test the LMs' recall skills based on x the bigram frequency in training. **Top right (recall distances)** We plot y perplexity for AR tokens based on x the distances between the re-occurring bigrams in context. **Bottom (non-recall frequencies)** We plot y perplexity on non-recall tokens based on x the bigram frequency in training. Further details are in Appendix B.

again using the parallel view on the new input of length l+1. This protocol is expensive, but future work could consider *periodically* updating the non-causal encoder-state when decoding many tokens.

#### 5.2 Overall natural language understanding

While recall is important for in-context learning, it is important to validate that the models remain strong in their overall natural language understanding abilities.

Language modeling perplexity. A fundamental challenge is how to compare the inherent quality of models pre-trained with disparate objectives. In our setting, this is challenging since JRT-RNN additionally minimizes a masked language modeling objective beyond the standard causal next token prediction objective and sees 50% less data than the decoder-only models for the next token prediction task (when M=1024, N=2048). Overall JRT-RNN computes losses on 65% of the number of training data tokens seen by the decoder-only models (with 15% masked tokens in the encoder region).

Despite these differences, we consider a simple proxy of evaluating the perplexity of decoder-baselines in comparison to encoder-decoder JRT-RNN in the overlapping non-causal regions of both model types (i.e. the last 1024 tokens per input sequence of N=2048 for our trained models). Following prior work [23], we further *slice* the perplexity in two groups: (1) the associative recall "AR slice" includes tokens, referred to as "AR hits", that require the model to perform recall in order to predict the next token correctly and (2) the "Other slice" containing the remaining tokens (e.g., memorized knowledge). <sup>5</sup>

Slicing the model predictions on the Pile test set, we observe the following. Our measurement protocols are described in further detail in Appendix B.

<span id="page-9-1"></span><sup>&</sup>lt;sup>5</sup>As a heuristic rule, a token is an "AR hit" if it is completes a bigram that was previously seen in-context, and this bigram is infrequent during training (i.e., was not memorized by the model) [23]. For instance, in the sequence "In 1957, Dr. Seuss wrote ... In 1982, Dr. <u>Seuss</u>" the second <u>Seuss</u> would be included as an "AR hit" if "Dr. Seuss' is a rare bigram during training.

- 1. **Recall frequencies.** JRT-RNN excels in the "AR slice". For infrequently seen bigrams during training (unlikely to be memorized in the model parameters), JRT-RNN improves in perplexity relative to Based and Mamba, two strong causal recurrent baselines (Figure 3, top right).
- 2. **Recall distances.** In the "AR slice", the gap between JRT-RNN and the decoder-only baselines grows as the distances between repeated bigrams seen in-context grows. This provides further support beyond Section 5.1 that JRT-RNN can help with longer context recall tasks (Figure 3).
- 3. Non-recall frequencies. JRT-RNN is worse in perplexity than the decoder-only LMs for the non-recall "Other slice" for bigrams that are rarely seen during training. This slice tests the model's use of memorized knowledge (as opposed to knowledge provided in the context). This is expected as JRT-RNN computes losses 65% of the tokens of the decoder-only LMs. We expect this gap to decrease with scale and longer training durations (seen as the bigram frequencies increases) (Figure 3, top left). Future work could also consider decoupling sequence mixers from MLPs (knowledge stores) in training. How best to normalize training between encoder-decoder and decoder-only LMs is an open question.

Natural language understanding benchmarks. We use the downstream SuperGLUE benchmark, a canonical test of natural language understanding ability [55], to evaluate each architecture at the 360M and 1.3B parameter scales in Table 8. We validate that the different architectures perform similarly on average across these generic, short-context language tasks as observed in prior work [7, 62, 63].

## 5.3 Generation throughput

Generation can be decomposed into prompt "prefill processing" and decoding "next token prediction" steps. Since JRT-RNN does not modify the decoding step relative to standard decoder-only recurrent models, we focus our discussion on the prefill stage.

<span id="page-10-0"></span>Table 5: Latency (ms) of inference prefill for each implementation. Each point is the average of 20 iterations, run on an NVIDIA H100 GPU. In Table 5, we vary the sequence length at a fixed batch size of 16. In Table 5, we vary the batch size at a fixed sequence length of 16384.

| Implementation                                                                            | 2048 | 4096 | 8192  | 16384 | 32768 |
|-------------------------------------------------------------------------------------------|------|------|-------|-------|-------|
| Based PyTorch Fast Transformer CUDA Based Triton (FLA) Based Custom CUDA FlashAttention-2 | 17.1 | 74.5 | 284.6 | OOM   | OOM   |
|                                                                                           | 11.4 | 23.0 | 47.0  | 96.0  | OOM   |
|                                                                                           | 1.0  | 2.8  | 9.3   | 32.6  | 123.7 |
|                                                                                           | 0.3  | 0.6  | 1.2   | 2.3   | 4.5   |
|                                                                                           | 0.5  | 1.8  | 6.8   | 26.6  | 107.8 |
| JRT-RNN PyTorch                                                                           | 21.3 | 89.2 | OOM   | OOM   | OOM   |
| JRT-PROMPT Custom CUDA                                                                    | 0.6  | 1.2  | 2.3   | 4.5   | 9.0   |
| JRT-RNN Custom CUDA                                                                       | 0.4  | 0.8  | 1.5   | 2.8   | 5.6   |

| Implementation                       | 2                 | 4          | 8    | 16   | 32          | 64                  |
|--------------------------------------|-------------------|------------|------|------|-------------|---------------------|
| Based PyTorch                        | 140.9             | 281.5      | OOM  | OOM  | OOM         | OOM                 |
| Based Triton (FLA) Based Custom CUDA | $\frac{4.6}{1.2}$ | 8.7        | 16.7 | 32.4 | 64.2        | 127.8               |
| Flash Attention-2                    | 3.5               | 1.3<br>6.7 | 1.5  | 2.3  | 4.5<br>52.9 | $\frac{8.9}{108.2}$ |
| Fast Transformer CUDA                | 17.1              | 26.7       | 50.7 | 95.5 | OOM         | OOM                 |
| JRT-RNN PyTorch                      | 169.6             | 340.3      | OOM  | OOM  | OOM         | OOM                 |
| JRT-PROMPT Custom CUDA               | 2.3               | 2.5        | 2.9  | 4.5  | 9.0         | 17.8                |
| JRT-RNN Custom CUDA                  | 1.5               | 1.5        | 1.8  | 2.8  | 5.6         | 11.1                |

Using the Based CUDA kernel proposed in [7], JRT-PROMPT gives  $11.9 \times$  and  $13.7 \times$  higher throughput in processing the prompt prefill than the FlashAttention-2 and FLA Triton kernels respectively (prefill length 32768) (Table 5). JRT-PROMPT provides  $6.1 \times$  and  $7.2 \times$  higher throughput than the FlashAttention-2 and FLA kernels respectively as we increase the batch size to 64 (Table 5). For JRT-PROMPT, we double the prefill length compared to the baselines, using  $2 \times$  the time of the original Based prefill.

We next extend the Based kernel to support JRT-RNN and demonstrate that the implementation achieves  $19.2 \times$  and  $22.0 \times$  higher throughput than FA2 and FLA as we increase sequence length to 32768 (Table 5).

JRT-RNN provides 9.7× and 11.5× higher throughput respectively as we increase the batch size to 64 (Table 5). JRT-RNN takes 1.24× the time of the Based prefill, improving efficiency over JRT-PROMPT.

We benchmark the inference efficiency of JRT-PROMPT and JRT-RNN in Table 5 (additional details in Appendix D). As baselines, we consider popular and well-optimized softmax attention and linear attention implementation. For attention, we consider FlashAttention-2 [12]. For linear attention, we consider the linear attention CUDA kernel from Fast Transformers [53, 64] and a Triton parallel Based kernel from Flash Linear Attention (FLA) [65]. We also compare to PyTorch implementations of JRT-RNN and Based. All numbers are benchmarked on a NVidia H100 GPU.

## 6 Conclusion

Recurrent LLMs promise drastically more efficient inference relative to Transformers, however they are brittle during in-context learning. We identify the role of data order as a key reason, formalized via synthetics and theory. Our analysis suggest that putting data in the right order in context or non-causally processing the context can help efficient recurrent models better use their limited memory. We translate these insights to JRT-Prompt and JRT-RNN respectively. JRT-Prompt improves the quality of recurrent models by  $11.0 \pm 1.3$  points averaged across models and tasks, and our prototype architecture, JRT-RNN, provides a 13.7 point improvement at 360M parameters and 6.9 point improvement at 1.3B parameters. Both methods increase throughput relative to FlashAttention-2 using IO-aware CUDA implementations.

While much of the effort on sub-quadratic LMs seeks to directly mimic the experience of using quadratic Transformer LMs, our work emphasizes that we can exploit the asymmetries in efficiency to close the quality gaps: *multiple* linear passes over data is still asymptotically more efficient than quadratic attention. To facilitate reproducing this work, we release code and models at https://github.com/HazyResearch/prefix-linear-attention.

## Acknowledgments

We thank Michael Zhang, Michael Poli, Daniel Fu, Kawin Ethayarajh, John Thickstun, and Neel Guha for their helpful feedback and discussion during this work. We thank the Hazy Research lab and Together AI for supporting this work. We gratefully acknowledge the support of NIH under No. U54EB020405 (Mobilize), NSF under Nos. CCF2247015 (Hardware-Aware), CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and 1937301 (RTML); US DEVCOM ARL under Nos. W911NF-23-2-0184 (Long-context) and W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under Nos. N000142312633 (Deep Signal Processing), N000141712266 (Unifying Weak Supervision), N000142012480 (Non-Euclidean Geometry), and N000142012275 (NEPTUNE); Stanford HAI under No. 247183; NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba, TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce, Total, the HAI-GCP Cloud Credits for Research program, the Stanford Data Science Initiative (SDSI), and members of the Stanford DAWN project: Facebook, Google, and VMWare. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views, policies, or endorsements, either expressed or implied, of NIH, ONR, or the U.S. Government. AR's research is supported by NSF grant CCF #2247014.

