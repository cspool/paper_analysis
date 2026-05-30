# <span id="page-7-0"></span>5 ANALYSIS

The effectiveness of gating balancing loss. Figure 5 (a) and (b) illustrate how our  $\mathcal{L}_{balance}$  function mitigates the reduction in entropy rates within gating functions, leading to a more uniform composition weight distribution. The performance comparison between MoLE and MoLE  $_{w/o}$   $_{\mathcal{L}_{balance}}$  in

<span id="page-7-1"></span>

| # Task                           | Metric  | LoRAHub         | PEMs  | MoLE |
|----------------------------------|---------|-----------------|-------|------|
| Translation                      |         |                 |       |      |
| WMT '14 En→Fr                    | BLEU    | 27.4            | 25.6  | 29.1 |
| WMT '14 Fr→En                    | BLEU    | 29.4            | 27.1  | 31.3 |
| WMT '16 En→De                    | BLEU    | 24.6            | 24.9  | 27.7 |
| WMT '16 De→En                    | BLEU    | 29.9            | 28.0  | 29.1 |
| WMT '16 En→Ro                    | BLEU    | 17.7            | 15.2  | 18.9 |
| WMT '16 Ro→En                    | BLEU    | 23.5            | 21.7  | 25.1 |
| Average                          |         | 25.4            | 24.2  | 26.9 |
| Struct to Text                   |         |                 |       |      |
| CommonGen                        | Rouge-1 | 53.7            | 48.8  | 55.1 |
|                                  | Rouge-2 | 23.1            | 22.4  | 23.1 |
|                                  | Rouge-L | 49.7            | 47.2  | 53.9 |
| DART                             | Rouge-1 | 45.3            | 46.2  | 48.8 |
|                                  | Rouge-2 | 22.6            | 18.9  | 23.5 |
|                                  | Rouge-L | 35.1            | 37.6  | 36.0 |
| E2ENLG                           | Rouge-1 | 41.1            | 40.7  | 42.0 |
|                                  | Rouge-2 | 26.3            | 24.2  | 29.0 |
|                                  | Rouge-L | 38.8            | 42.1  | 41.8 |
| WebNLG                           | Rouge-1 | 52.1            | 52.0  | 54.5 |
|                                  | Rouge-2 | 23.9            | 24.6  | 26.8 |
|                                  | Rouge-L | 45.2            | 47.8  | 49.3 |
| Average                          |         | 38.1            | 37.7  | 40.3 |
| Closed-Book QA                   |         |                 |       |      |
| ARC-c                            | EM      | 51.7            | 50.4  | 52.9 |
| ARC-e                            | EM      | 69.7            | 65.7  | 70.3 |
| NO                               | EM      | 17.3            | 16.1  | 23.5 |
| TOA                              | EM      | <del>54.5</del> | 53.9  | 54.0 |
| Average                          |         | <u>48.3</u>     | 46.5  | 50.2 |
| Big-Bench Hard (BBH)             |         |                 |       |      |
| Boolean Expressions              | EM      | 55.1            | 53.0  | 57.3 |
| Causal Judgement                 | EM      | 57.6            | 51.1  | 57.9 |
| Date Understanding               | EM      | 31.0            | 29.3  | 30.7 |
| Disambiguation                   | EM      | 46.6            | 47.2  | 49.3 |
| Penguins in a Table              | EM      | 41.4            | 39.8  | 45.0 |
| Reasoning Objects                | EM      | 35.2            | 37.5  | 33.7 |
| Ruin Names                       | EM      | 19.9            | 19.3  | 21.2 |
| Average                          |         | 38.4            | 33.2  | 42.2 |
| Natural Language Inference (NLI) |         |                 |       |      |
| ANLI-R1                          | EM      | 81.0            | 80.3  | 82.7 |
| ANLI-R1                          | EM      | 80.9            | 80.2  | 82.4 |
| ANLI-R3                          | EM      | 77.4            | 76.6  | 78.9 |
| ONLI                             | EM      | 77.6            | 78.0  | 78.1 |
| Average                          | 1.711   | 79.2            | 78.8  | 80.5 |
| 11.01.05                         |         |                 | , 5.0 | 0010 |

Table 3: Evaluation results on Translation, Struct to Text, Closed-Book QA, NLI and BBH. The **best value** is in bold and the second-best value is underlined.

Table 7 underscores the performance enhancement achieved with the inclusion of  $\mathcal{L}_{balance}$ . Additionally, we conducted an experiment wherein we solely increased the temperature  $\tau$  in Eq. 11, as an alternative to adding  $\mathcal{L}_{balance}$ . Results in Table 7 shows declining performance in MoLE variants MoLE<sup> $\tau_1$ </sup>, MoLE<sup> $\tau_2$ </sup>, MoLE<sup> $\tau_3$ </sup> ( $\tau_1 \prec \tau_2 \prec \tau_3$ ) with increasing temperature. While temperature rise addresses gating imbalance, it restricts dynamic LoRA exploration in MoLE, leading to inferior outcomes.

**Further comparison with SOTA multi-concept generation methods**. In the absence of comparable LoRA composition methods in the V&L domain, we incorporated two leading multi-concept generation algorithms that do not utilize LoRA: Custom (Kumari et al., 2023) and Textual Inversion (Gal et al., 2022a), both of which emphasize full-parameter training for enhanced results. As presented in Table 2, MoLE outperforms Textual Inversion in both image and text alignment and excels over Custom in text alignment. Furthermore, it's worth noting that our MoLE is more lightweight compared to these full-parameter training methods. These comparisons underscore the superior effectiveness of our MoLE relative to methods that involve extensive parameter tuning.

Scale to a larger number of LoRAs. We explore the performance as the number of LoRAs increases. In the NLP domain, experiments were conducted with varying numbers of LoRA (8, 24, 48, 128),

as detailed in Table [6.](#page-11-3) Our MOLE demonstrated optimal performance across these configurations, notably excelling with larger LoRA counts of 48 and 128, surpassing LoRAHub by 2.5 and 3.0, respectively. Analysis revealed that LoRAHub's optimization algorithm often zeroes out many LoRA weights in larger arrays, thus underutilizing the potential of all LoRA. Conversely, MOLE effectively overcomes this limitation. However, all methods, including MOLE, showed performance declines with an extremely large number of LoRA (128), highlighting a need for further research in this area. In the V&L domain, Table [10](#page-12-0) shows experiments with increased composed LoRAs. While typical composition involve 3-4 visual concepts, our range was 3-6 to avoid ambiguity in outputs. Results indicate that MOLE consistently outperforms other LoRA composition models in text and image alignment as the number of LoRAs increases, underscoring its robustness and superior composition capabilities.

Coarse-to-fine gating analysis. To examine the impact of different granularity levels in gating functions, we delineated four levels in MOLE: matrix-wise (MOLE, gating at the parameter matrix level), layer-wise (MOLE), block-wise (MOLE), and network-wise (MOLE), abbreviated as m-MOLE, l-MOLE, b-MOLE, and n-MOLE respectively. Table [9](#page-12-1) reveals that intermediate granularities, b-MOLE and l-MOLE, achieved the highest performance. In contrast, the coarsest level, n-MOLE, which involves minimal optimizable parameters (a single gating for the entire network), showed suboptimal outcomes. Additionally, the finest granularity, m-MOLE, underperformed, potentially due to its excessive control interfering with inherent relationships in LoRA parameters.

Generalization to new datasets. To further validate the effectiveness of our MOLE, we conducted generalization experiments. Specifically, all LoRA candidates and LoRA composition variants, including MOLE, PEMs and LoRAHub, were trained on NLI tasks (ANLI-R1, ANLI-R2, ANLI-R3, QNLI, and WNLI, among others). Subsequently, we evaluated these methods on the BBH dataset. As illustrated in Table [8,](#page-11-4) our MOLE achieves an average performance advantage of 2.4 over LoRAHub and 3.7 over PEMs, underscoring its superior generalization ability.

Flexibility of MOLE. As discussed in Section [2.1,](#page-2-2) a well-designed LoRA composition method should not only achieve effective LoRA composition but also retain the characteristics of individual LoRA. It should be versatile enough to function as a standalone LoRA generator, ensuring its practical applications are flexible and widespread. Figure [6](#page-12-2) displays a comparison of the qualitative results for the retaining ability of several composition methods, we find that our MOLE can generate images that closely resemble the original features of the LoRA experts (e.g., dog ears, the color of the backpack), while other composition methods tend to produce confusion and loss of LoRA characteristics. Besides, as shown in Figure [1,](#page-0-0) we can also degrade MOLE by masking out the LoRA experts we do not wish to use, transforming it into a MOLE that merges fewer LoRAs without affecting the composition effect of the remaining LoRAs. As shown in Figure [8,](#page-13-0) our MOLE can achieve the same flexible LoRA composition as linear arithmetic composition method without altering the weights of MOLE, while reference tuning-based composition [\(Gu et al.,](#page-9-6) [2023\)](#page-9-6) can not accomplish.

Hierarchical control analysis. MOLE aims to achieve improved LoRA composition effects through finer-grained hierarchical control. As illustrated in the Figure [7,](#page-12-3) we visualize the weight distributions assigned by the gating functions learned by MOLE at different levels in both NLP and V&L domains. We observe that MOLE adaptively assigns weights to different LoRA experts at various layers. Consequently, finer-grained weight combination methods lead to superior results.

