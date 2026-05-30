# 3 Experiments

We first describe our general experimental protocol for language modeling, used for the majority of our experiments. In Section [3.1,](#page-11-0) we evaluate on a high-quality English dataset, showing significantly stronger performance than baselines, as well as improved robustness and interpretability from avoiding tokenization. In Section [3.2,](#page-14-0) we extend our evaluation to diverse datasets including Chinese, code, and DNA, with even larger performance improvements, demonstrating H-Net's versatility as a general sequence model architecture. In Section [3.3,](#page-16-0) we provide comprehensive ablations that study individual architectural components and design choices.

Models. We compare against a standard tokenized Transformer following the Llama architecture (Touvron, Martin, et al. [2023;](#page-28-10) Grattafiori et al. [2024\)](#page-25-0).[11](#page-9-1) We additionally compare against several byte-level baselines:

- MambaByte (Wang et al. [2024\)](#page-28-4) is an isotropic model using pure Mamba-2 layers.
- LlamaByte is an isotropic model using pure Transformer layers.
- SpaceByte (Slagle [2024\)](#page-28-5) represents the canonical hierarchical architecture with external boundary supervision, which chunks on spaces and "space-like" bytes. [12](#page-9-2) On English, the space-like delimiter heuristic empirically has an average ratio of 6.0 bytes per chunk.
- SpaceByte++ is our modification of SpaceByte that includes our architectural modifications to the hierarchical structure (from Section [2.1\)](#page-3-3). In particular, it changes the outer encoder/decoder networks to use Mamba-2, and modifies the layer counts and widths slightly to match the H-Net models below.
- H-Net (space) further improves SpaceByte++ with our training improvements to the network (Section [2.3\)](#page-7-0), in particular, adding post-network norms, residual projections, and learning rate multipliers on the outer networks. H-Net (space) differs from our full H-Net only through the chunking function.

<span id="page-9-0"></span><sup>10</sup>Here, we use token in the autoregressive generation sense, referring to one time step, not in the literal BPE token sense.

<span id="page-9-2"></span><span id="page-9-1"></span><sup>11</sup>This was called the "Transformer++" in Gu and Dao [\(2024\)](#page-25-2); since by now it is firmly established, we remove the "++".

<sup>12</sup>BLT is another architecture with external supervision using entropy instead of delimiters, but is unfortunately too complex to set up and control as a baseline. We believe that the delimiter-based method is highly competitive. See Appendix [A.1.3.](#page-31-0)

<span id="page-10-0"></span>Table 1: Architectures for main language models, all data-/FLOP-matched.  $\mathcal{E}^0, \mathcal{D}^0$ ,  $\mathcal{E}^1, \mathcal{D}^1$ ,  $\mathcal{M}$ . T and M denote a Transformer and a Mamba-2 layer, respectively. For hierarchical byte-level models, the Tokenizer column lists the chunking mechanism. The numbers before DC indicate downsampling factor N in equation (10); for example, (3,3)-DC denotes  $N^0 = N^1 = 3$ . The BPIC (Bytes-Per-Innermost-Chunk) measure shows that each chunk dynamically determined by our 1-stage comprises similar number of bytes to the GPT-2 tokenizer, despite aiming for  $N^0 = 6$ . All Transformer layers in  $\mathcal{E}$  or  $\mathcal{D}$  networks, as well as LlamaByte, use Sliding Window Attention (SWA) with a window size of 1024.

| Model           | Input    | Tokenizer  | $L^0$ | BPIC $(L^S/L^0)$ | #Params | Architecture                | d_model (D)      |  |
|-----------------|----------|------------|-------|------------------|---------|-----------------------------|------------------|--|
| #FLOPs matc     | hed to G | PT-3 Large |       |                  |         |                             |                  |  |
| Transformer     | Token    | GPT2       | 1792  | 4.6              | 760M    | T24                         | 1536             |  |
| LlamaByte       |          | _          |       | 1.0              | 210M    | T16                         | 1024             |  |
| MambaByte       |          | _          |       | 1.0              | 190M    | M28                         | 1024             |  |
| SpaceByte       |          | Spacelike  |       | 6.0              | 570M    | T8 + T16 + T8               | 768 , 1536       |  |
| SpaceByte++     | Data     | Spacelike  | 8192  | 6.0              | 850M    | M4 + T28 + M4               | 1024, 1536       |  |
| H-Net (pool)    | Byte     | 6-Pool     | 8192  | 6.0              | 850M    | M4 + T28 + M4               | 1024, 1536       |  |
| H-Net (space)   |          | Spacelike  |       | 6.0              | 850M    | M4 + T28 + M4               | 1024, 1536       |  |
| H-Net (1-stage) |          | 6-DC       |       | 4.8              | 680M    | M4 + T22 + M4               | 1024, 1536       |  |
| H-Net (2-stage) |          | (3,3)-DC   |       | 7.0              | 870M    | M4 + T1M4 + T26 + M4T1 + M4 | 1024, 1024, 1536 |  |
| #FLOPs matc     | hed to G | PT-3 XL    |       |                  |         |                             |                  |  |
| Transformer     | Token    | GPT2       | 1792  | 4.6              | 1.3B    | T24                         | 2048             |  |
| SpaceByte++     |          | Spacelike  |       | 6.0              | 1.6B    | M4 + T31 + M4               | 1024, 2048       |  |
| H-Net (space)   | D (      | Spacelike  | 04.00 | 6.0              | 1.6B    | M4 + T31 + M4               | 1024, 2048       |  |
| H-Net (1-stage) | Byte     | 6-DC       | 8192  | 4.7              | 1.3B    | M4 + T24 + M4               | 1024, 2048       |  |
| H-Net (2-stage) |          | (3,3)-DC   |       | 6.9              | 1.6B    | M4 + T1M4 + T27 + M4T1 + M4 | 1024, 1536, 2048 |  |

- H-Net (pool) is a baseline ablating the effect of a simple chunking strategy that pools every *k* tokens, which is expected to be weaker than all of the data-dependent chunking strategies.
- H-Net (1-stage) is our full H-Net method with DC learned end-to-end (Section 2.2) with compression target  $N^0 = 6$ .
- H-Net (2-stage) is our full H-Net method, iterated to two nested stages using  $N^0 = 3$ ,  $N^1 = 3$ .

We provide results for two model scales, *Large (L)* and *XL*. Each scale is FLOP-matched to the corresponding GPT-3 (Brown et al. 2020) (*i.e.*, GPT-3 L and GPT-3 XL) variant of the tokenized Transformer (760M and 1.3B parameters respectively).

**Experimental Setup.** Following established practice (Xue et al. 2022; Wang et al. 2024; Slagle 2024), we measure performance using bits-per-byte (BPB) to ensure comparability across different input representations. For tokenized models, this amounts to simply rescaling the total negative log likelihood of a sequence (in tokens) by the total number of bytes. In addition, we systematically control the data and compute budget for all models (see Table 1), matching all models carefully in both **bytes-per-batch** and **FLOPs-per-byte**:<sup>13</sup>

- Data Budget: We train all models on the 100B token subset sampled from the FineWeb-Edu dataset (Penedo et al. 2024). All tokenizer-free models process 8192 utf-8 encoded bytes per sequence, while the Transformer uses 1792 tokens from the GPT2 tokenizer (roughly equivalent to 8192 bytes). We use batch size 256 for all models; the total batch size is just under 0.5M tokens per batch for the baseline BPE Transformer, roughly matching protocols from prior work (Gu and Dao 2024).
- Compute Budget: For calculating FLOPs, we follow standard methodology (Hoffmann et al. 2022) with an extension for Mamba-2 layers (see Appendix B). We use the BPE-tokenized Transformer's #FLOPs as a reference, and the number of layers of the other models is adjusted accordingly to match the reference #FLOPs.

Training employs AdamW (Loshchilov and Hutter 2017) optimizer with warmup-stable-decay (WSD) (Hu et al. 2024)

<span id="page-10-1"></span><sup>&</sup>lt;sup>13</sup>Another way to interpret this is that every model sees the exact same underlying data (regardless of tokenization) per minibatch, and every model aims to use the same number of FLOPs in every forward/backward pass.

scheduling with 10% linear warmup and 20% inverse-square-root decay (Ibrahim et al. 2024). Following Hägele et al. (2024) which recommends WSD schedulers with half the maximum learning rates as a cosine schedule, we adopt learning rates 2.5× higher than GPT-3 (Radford et al. 2019) standards; this corresponds to half of the maximum learning rate used in Gu and Dao (2024), yielding  $6.25 \times 10^{-4}$  for *Large*-scale models and  $5.0 \times 10^{-4}$  for *XL*-scale models. Architecture details include gated MLPs (Touvron, Martin, et al. 2023) in all Transformer layers and the main network's Mamba layers, while Mamba layers in  $\mathcal E$  and  $\mathcal D$  are without an MLP. For Transformer layers in  $\mathcal E$  and  $\mathcal D$ , we use Sliding Window Attention (SWA) (Beltagy, Peters, and Cohan 2020) with the window size of 1024. As discussed Section 2.1,  $\mathcal E$  and  $\mathcal D$  comprise mainly Mamba-2 layers.

### <span id="page-11-0"></span>3.1 Language Modeling

**Training Curves.** Figure 3 presents validation BPB metrics throughout training for both Large and XL model scales.

**Large Scale.** At the *Large* scale, we make note of the following comparisons.

- All isotropic models severely underperform hierarchical models. Among these, **MambaByte** is significantly better than **LlamaByte**, both the FLOP-matched sliding window attention (SWA) variant and even the global attention variant that is data-matched but uses 2× the FLOPs.
- H-Net (pool) is much worse than all other H-Net variants, validating that fixed-width chunking is not effective.
- **SpaceByte** is much worse than **SpaceByte++**, validating our strategy for network design as well as usage of Mamba in the outer networks (Section 2.1). **SpaceByte++** is in turn worse than **H-Net (space)**, validating our improved signal propagation techniques (Section 2.3).
- **H-Net (space)** is a very strong model reaching the performance of the **BPE Transformer**, validating the effect of data-dependent chunking strategies together with a well-designed hierarchical architecture.
- **H-Net (1-stage)** is stronger than **H-Net (space)**, validating that our dynamic chunking mechanism successfully learns how to segment data in a *context-dependent* way that improves over strong heuristics.
- H-Net (2-stage) is significantly better than H-Net (1-stage), validating that iterated dynamic chunking can potentially learn a nested hierarchy of useful features, and leverage compute and parameters even more effectively.

**XL Scale.** At the XL scale, we zoom in more closely and compare only the strongest set of methods: SpaceByte++, H-Net (space), H-Net (1-stage), and H-Net (2-stage).

The same trends hold as at the *Large* scale. Our SpaceByte++ baseline is strong, but slightly worse than the BPE Transformer baseline. On the other hand, all byte-level H-Net methods start off worse than the token-level Transformer, but scale better after enough data. H-Net (space), H-Net (1-stage), and H-Net (2-stage) cross over the tokenized Transformer after just 200B bytes, 100B bytes, and 30B bytes respectively. Beyond these points, H-Net's performance advantage widens progressively, demonstrating that the benefits of learnable dynamic chunking get strengthened with additional training data, as the model continuously refines its chunking strategy.

**Downstream Evaluations.** Table 2 presents zero-shot accuracy across diverse downstream benchmarks (Paperno et al. 2016; Zellers et al. 2019; Bisk et al. 2020; Clark, Cowhey, et al. 2018; Sakaguchi et al. 2021; Mihaylov et al. 2018) using 1m-evaluation-harness (Gao, Tow, et al. 2024) for models at Large and XL scales. SpaceByte++, H-Net (space), and H-Net (1-stage) all have similar performance to the BPE Transformer at *Large* scale, and slightly outperform it at the *XL* scale, consistent with their close training curves (and possibly reflecting some noise in the evaluations).

H-Net (2-stage) consistently achieves the highest performance across most tasks, outperforming 2.2% and 2.6% over the Transformer baseline at *Large* and *XL* scales respectively. Notably, the *Large* H-Net (2-stage) matches the average downstream performance of the *XL* BPE Transformer.

<span id="page-11-1"></span> $<sup>^{14}</sup>$ Just as in the original Mamba (Gu and Dao 2024) and Mamba-2 (Dao and Gu 2024) blocks, our Mamba layers have roughly  $6(D^s)^2$  parameters and Transformer layers have  $12(D^s)^2$  parameters in stage s.

<span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure 3: **Validation Bits-per-byte (BPB) throughout training** for different models at Large (760M, left) and XL (1.3B, right) scales with matched computational and data budgets for training. All models but Transformer take raw byte inputs (Transformer uses GPT-2 tokenizer). Vertical dotted lines indicate crossover points where H-Net begins to outperform Transformer with predefined BPE tokenization. From the curves we can clearly see the following: (1) all hierarchical models (*i.e.*, SpaceByte++, H-Net variants) outperform the isotropic models (*i.e.*, Transformer, MambaByte, LlamaByte); (2) dynamic chunking is more powerful than BPE tokenizers; and (3) DC is more effective than other chunking strategies. Furthermore, H-Net's 2-stage variant consistently outperforms 1-stage across both scales, demonstrating the effectiveness of deeper hierarchies. See Table 1 for architectural details.

<span id="page-12-1"></span>Table 2: **Zero-shot performance comparison across multiple benchmarks, all data-/FLOP-matched.** Evaluation results on seven downstream tasks at both Large (760M) and XL (1.3B) scales. GFLOPs/Byte is measured during evaluation on FineWeb-Edu validation set. See Table 1 for architectural details.

| Model              | Input   | GFLOPs/<br>Byte | F-Edu<br>bpb↓ | LMB.<br>acc↑ | Hella.<br>acc_n↑ | PIQA<br>acc↑ | ARC-E<br>ACC↑ | ARC-c<br>acc_n↑ | Wino.<br>acc↑ | Open.<br>acc_n↑ | Average<br>acc ↑ |
|--------------------|---------|-----------------|---------------|--------------|------------------|--------------|---------------|-----------------|---------------|-----------------|------------------|
| #FLOPs matched     | to GPT- | 3 Large         |               |              |                  |              |               |                 |               |                 |                  |
| Transformer        | Token   | 0.42            | 0.756         | 45.0         | 54.5             | 72.3         | 69.9          | 36.3            | 55.9          | 38.8            | 53.3             |
| LlamaByte          |         | 0.42            | 0.859         | 37.0         | 40.5             | 64.7         | 55.1          | 26.7            | 52.3          | 32.4            | 44.1             |
| LlamaByte (Global) |         | 0.95            | 0.845         | 36.4         | 41.5             | 65.7         | 57.2          | 27.1            | 49.8          | 32.2            | 44.3             |
| MambaByte          |         | 0.42            | 0.845         | 32.9         | 42.0             | 66.2         | 55.9          | 28.1            | 51.7          | 33.2            | 44.3             |
| SpaceByte          |         | 0.41            | 0.791         | 43.0         | 49.0             | 69.0         | 63.3          | 33.5            | 53.3          | 35.0            | 49.4             |
| SpaceByte++        | Byte    | 0.42            | 0.760         | 48.0         | 55.7             | 71.3         | 67.9          | 35.4            | 57.5          | 39.6            | <u>53.6</u>      |
| H-Net (pool)       |         | 0.42            | 0.780         | 43.2         | 54.7             | 69.7         | 67.9          | 34.7            | 54.8          | 36.4            | 51.6             |
| H-Net (space)      |         | 0.42            | 0.755         | 46.7         | <u>55.9</u>      | 72.4         | 68.8          | 34.6            | 57.6          | 38.0            | 53.4             |
| H-Net (1-stage)    |         | 0.43            | 0.755         | 46.2         | 55.5             | 71.0         | 68.1          | 35.6            | <u>58.6</u>   | 40.0            | <u>53.6</u>      |
| H-Net (2-stage)    |         | 0.43            | 0.743         | <u>46.9</u>  | 57.4             | 72.0         | 71.7          | 39.2            | 60.4          | 40.6            | 55.5             |
| #FLOPs matched     | to GPT- | 3 XL            |               |              |                  |              |               |                 |               |                 |                  |
| Transformer        | Token   | 0.69            | 0.730         | 48.1         | 58.0             | 73.1         | 72.2          | 37.5            | 58.6          | 40.8            | 55.5             |
| SpaceByte++        |         | 0.72            | 0.733         | 51.3         | 60.1             | 72.4         | 71.8          | 38.0            | 58.5          | 40.6            | 56.1             |
| H-Net (space)      | Byrto   | 0.70            | 0.726         | 50.3         | <u>61.5</u>      | 73.6         | 72.4          | 40.2            | 60.2          | 41.8            | <u>57.1</u>      |
| H-Net (1-stage)    | Byte    | 0.72            | 0.728         | 48.4         | 59.5             | 72.4         | 73.0          | 38.3            | 59.2          | 42.4            | 56.2             |
| H-Net (2-stage)    |         | 0.69            | 0.715         | <u>50.5</u>  | 62.2             | 73.7         | 74.2          | 42.2            | 60.5          | 44.0            | 58.2             |

<span id="page-13-0"></span>Table 3: Robustness evaluation on HellaSwag with textual perturbations, all data-/FLOP-matched. Zero-shot accuracy on five different perturbation types (AntSpeak, Drop, RandomCase, Repeat, UpperCase) for models trained exclusively on clean data without noise augmentation. Best and second best results in each column are denoted using bolded and underlined texts, respectively. The Robustness Score metric show that all byte-level models are more robust to adversarial text inputs than tokenizer-based Transformer. H-Net (2-stage) shows significantly enhanced robustness in textual perturbations, with the highest average accuracy across all noise types and highest robustness score. See Table 1 for architectural details, and Appendix D.1 for the definition of Robustness Score.

| Model                         | Input    |                                           |      | . Average↑  | Robustness                              |             |             |             |  |  |
|-------------------------------|----------|-------------------------------------------|------|-------------|-----------------------------------------|-------------|-------------|-------------|--|--|
|                               | 1111 0 1 | AntSpeak Drop RandomCase Repeat UpperCase |      | UpperCase   | 11, 21, 21, 21, 21, 21, 21, 21, 21, 21, | Score ↑     |             |             |  |  |
| #FLOPs matched to GPT-3 Large |          |                                           |      |             |                                         |             |             |             |  |  |
| Transformer                   | Token    | <u>31.1</u>                               | 29.9 | 27.1        | 27.8                                    | 38.9        | 30.9        | 20.2        |  |  |
| LlamaByte (W1024)             |          | 30.4                                      | 28.1 | 29.3        | 27.2                                    | 38.5        | 30.7        | 36.9        |  |  |
| LlamaByte (Global)            |          | <u>31.1</u>                               | 28.1 | 29.7        | 27.3                                    | 39.0        | 31.0        | 36.6        |  |  |
| MambaByte                     |          | 29.8                                      | 27.9 | 29.9        | 27.1                                    | 39.6        | 30.9        | 34.5        |  |  |
| SpaceByte                     |          | 30.7                                      | 29.8 | 33.5        | 29.5                                    | 47.8        | 34.3        | 38.1        |  |  |
| SpaceByte++                   | Byte     | 31.0                                      | 30.9 | 35.8        | 29.3                                    | 54.0        | 36.2        | 36.4        |  |  |
| H-Net (pool)                  | ·        | 30.5                                      | 31.2 | 35.4        | 29.6                                    | 53.4        | 36.1        | 37.3        |  |  |
| H-Net (space)                 |          | 30.8                                      | 31.2 | <u>38.6</u> | 29.4                                    | 54.0        | 36.8        | <u>38.2</u> |  |  |
| H-Net (1-stage)               |          | 31.2                                      | 31.1 | 35.4        | 29.9                                    | <u>54.1</u> | 36.4        | 37.2        |  |  |
| H-Net (2-stage)               |          | 30.8                                      | 32.1 | 39.3        | 30.4                                    | 57.1        | 38.0        | 39.0        |  |  |
| #FLOPs matched                | to GPT-3 | 3 XL                                      |      |             |                                         |             |             |             |  |  |
| Transformer                   | Token    | 31.6                                      | 30.7 | 28.0        | 28.5                                    | 43.0        | 32.3        | 22.2        |  |  |
| SpaceByte++                   |          | 30.9                                      | 32.1 | 40.3        | 30.6                                    | 58.5        | 38.5        | 38.5        |  |  |
| H-Net (space)                 | Dt-      | 31.2                                      | 33.2 | 41.9        | 31.8                                    | 60.7        | <u>39.8</u> | <u>40.5</u> |  |  |
| H-Net (1-stage)               | Byte     | 30.9                                      | 32.7 | 39.2        | 31.2                                    | 58.4        | 38.6        | 39.5        |  |  |
| H-Net (2-stage)               |          | 31.1                                      | 34.7 | 44.1        | 33.0                                    | 61.7        | 40.9        | 42.8        |  |  |

**Robustness to Textual Perturbations.** Table 3 evaluates model robustness on HellaSwag with various textual perturbations, following protocols from BLT (Pagnoni et al. 2024). Importantly, these are the same checkpoints trained on clean FineWeb-Edu data used to evaluate Table 2), without any form of special data mix or augmentations that may improve character-level robustness. H-Net (2-stage) demonstrates substantially improved robustness compared to all baselines, with performance gaps exceeding those observed in standard benchmarks.

**Visualization of Tokenized Positions.** In Figure 4, we provide visualizations of the boundaries dynamically drawn by H-Net (1-stage) and H-Net (2-stage). The visualization offers several insights about how the model decides boundaries.

- Single-stage behavior: H-Net (1-stage) predominantly places boundaries at whitespace characters, closely mirroring the delimiters used by SpaceByte. This indicates that H-Net learns that word boundaries represent natural semantic units in text. This convergence to spacelike boundaries, discovered purely through end-to-end training, conversely validates SpaceByte's strong empirical performance.
- *Hierarchical chunking patterns:* The first stage of H-Net (2-stage) combines spacelike boundaries with first few characters of each word. This strategy helps the model because once the initial positions of a word are identified, the remaining characters become highly predictable.
- Content-aware chunking: One might question if H-Net's chunking decisions follow static rules, such as drawing boundaries only at certain fixed bytes (e.g., whitespace). However, as shown in the figure, H-Net often merges multiple words and spacelike characters based on content (examples include the backbone, such as, and (ii)).
- *Perturbation behavior:* Figure 16 shows the same example with textual perturbations such as removing whitespaces, which more prominently demonstrates that boundaries drawn by H-Net are based on content and context. In particular, it often still chunks in between semantic words even if the space is removed.

<span id="page-14-1"></span>![](_page_14_Figure_0.jpeg)

Figure 4: Visualization of boundaries drawn by H-Net. Numbers above indicate byte value, and the colored boxes indicate positions where  $b_t = 1$ . (a) H-Net (1-stage) tends to draw boundaries at spacelike bytes, which is very similar to SpaceByte. (b) The boundaries of the first stage in H-Net (2-stage) are focused on spacelike bytes, as well as starting characters of each word. The second stage of H-Net (2-stage) chunks the text into more meaningful units, such as words or numberings (i.e., '(ii)'). We can also observe that it often chunks multiple words that form one semantic group; for example, 'the backbone' and 'such as'.

(b) H-Net (2-stage), using (3,3)-DC.

74 69 65 73 20 73 75 63 68 20 61 73 20 6c 61 6e 67 75 61 67 65 20

6e 64 20 67 65 6e 6f 6d 69 63 73 2e 20 28 69 69 29 20 46 61 73 74

a s 

ties such

nd genomics.

language

(ii) Fast

Stage 0

Stage 1

Stage 0

Stage 1

Stage 0

Stage 1

Stage 0

Stage 1

Stage 0 Stage 1

Stage 0

Stage 1

Stage 0

Stage 1

Stage 0

Stage 1

Stage 0

Stage 1

Stage 0

Stage 1

