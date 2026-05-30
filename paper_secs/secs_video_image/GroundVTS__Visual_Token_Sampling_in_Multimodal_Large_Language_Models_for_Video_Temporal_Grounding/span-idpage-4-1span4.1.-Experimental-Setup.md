# <span id="page-4-1"></span>4.1. Experimental Setup

Dataset Preparation. We employ two datasets in the training pipeline: (a) LLaVA-Video-178K [\[65\]](#page-10-15), a large-scale video dataset that provides diverse multimodal supervision across tasks such as video captioning and VQA for pretraining and alignment; and (b) Grounding-FT, a curated dataset we construct for VTG tasks. To adapt VTG supervision to the natural language input-output format of LLMs, we design a set of instruction templates with diverse linguistic expressions and combine them with temporal grounding queries to form QA-style training pairs. Grounding-FT is derived from multiple VTG training splits covering both moment retrieval and highlight detection tasks [\[9,](#page-8-4) [20,](#page-8-5) [21\]](#page-8-6), and contains a total of 70K annotated video-query pairs. Dataset details are provided in the supplementary material. LLaVA-Video-178K is used in the first two training stages, while Grounding-FT is employed in the third stage.

Evaluation. We evaluate our model on two representative

VTG tasks, namely moment retrieval (MR) [\[4,](#page-8-0) [9\]](#page-8-4) and highlight detection (HD) [\[21\]](#page-8-6). The MR task aims to identify the start and end timestamps of the video segment corresponding to a given natural language query. Following standard practice, we conduct evaluation on Charades-STA [\[9\]](#page-8-4), ActivityNet-Captions [\[4\]](#page-8-0), and QVHighlights [\[21\]](#page-8-6), using mean intersection-over-union (mIoU) and Recall@1 (R1@t) at thresholds t ∈ {0.3, 0.5, 0.7} [\[23,](#page-8-18) [41,](#page-9-18) [55\]](#page-10-13). The HD task requires the model to output all salient moments relevant to the query in the video together with their corresponding relevance scores. We use QVHighlights [\[21\]](#page-8-6) for evaluation and adopt mean average precision (mAP) and the hit ratio of the highest-scored clip (Hit@1) as metrics [\[12,](#page-8-11) [30,](#page-9-11) [43\]](#page-9-19). Implementation Details. We construct two model variants, GroundVTS-Q and GroundVTS-I, built upon Qwen2.5VL-7B [\[3\]](#page-8-8) and InternVL3.5-8B [\[48\]](#page-10-16), respectively. Both models are trained using the three-stage strategy described in Sec. [3.4,](#page-4-0) where stages 1–3 are trained for 1, 2, and 3 epochs, respectively, with learning rates of 1×10<sup>−</sup><sup>5</sup> , 2×10<sup>−</sup><sup>4</sup> , and 1×10<sup>−</sup><sup>4</sup> . The two base models differ in their intrinsic video sampling paradigms. QwenVL employs a fixed frame-rate strategy, uniformly sampling frames over time, whereas InternVL adopts a fixed frame-count strategy, representing each video with a constant number of frames regardless of duration. During training, GroundVTS-Q uses a frame rate of 2 FPS, while GroundVTS-I samples 16 frames per video. For the VTS module, the hidden dimension D<sup>r</sup> is set to 512 for GroundVTS-Q and 128 for GroundVTS-I. The visual token sampling ratio is fixed at ρ= 0.5. Additional training settings are detailed in the supplementary material.

Table 1. Comparison with state-of-the-art methods on Charades-STA and ActivityNet-Captions test splits.

<span id="page-5-3"></span><span id="page-5-0"></span>

| Method                 |                                | Charad                         | es-STA                              |                                |                          | ActivityNo              | et-Captions                    |                                     |
|------------------------|--------------------------------|--------------------------------|-------------------------------------|--------------------------------|--------------------------|-------------------------|--------------------------------|-------------------------------------|
| Method                 | R1@.3                          | R1@.5                          | R1@.7                               | mIoU                           | R1@.3                    | R1@.5                   | R1@.7                          | mIoU                                |
| LLaVA-OV[22] arXiv' 24 | 28.8                           | 16.6                           | 5.9                                 | 19.3                           | 20.2                     | 8.6                     | 2.2                            | 13.5                                |
| TimeChat[43] CVPR' 24  | 47.7                           | 22.9                           | 12.5                                | 30.6                           | 30.2                     | 16.9                    | 8.2                            | 21.8                                |
| VTimeLLM[16] CVPR' 24  | 51.0                           | 27.5                           | 11.4                                | 31.2                           | 44.0                     | 27.8                    | 14.3                           | 30.4                                |
| Momentor[39] ICML' 24  | 42.9                           | 23.0                           | 12.4                                | 29.3                           | 42.6                     | 26.6                    | 11.6                           | 28.5                                |
| HawkEye[52] arXiv' 24  | 50.6                           | 31.4                           | 14.5                                | 33.7                           | 49.1                     | 29.3                    | 10.7                           | 32.7                                |
| ChatVTG[41] CVPR' 24   | 52.7                           | 33.0                           | 15.9                                | 34.9                           | 40.7                     | 22.5                    | 9.4                            | 27.2                                |
| NumPro[55] CVPR' 25    | 63.8                           | 42.0                           | 20.6                                | 41.4                           | 55.6                     | 37.5                    | <u>20.6</u>                    | 38.8                                |
| LLaVA-ST[23] CVPR' 25  | 63.1                           | 44.8                           | 23.4                                | 42.4                           |                          |                         |                                |                                     |
| Qwen2.5VL-7B           | 34.2                           | 18.8                           | 8.6                                 | 22.1                           | 25.3                     | 11.5                    | 4.4                            | 17.1                                |
| Qwen2.5VL-7B-G         | 45.2                           | 32.7                           | 18.7                                | 31.7                           | 40.6                     | 23.9                    | 9.9                            | 26.7                                |
| GroundVTS-Q (ours)     | <b>71.5</b> <sub>(†26.3)</sub> | <b>57.5</b> <sub>(†24.8)</sub> | <b>34.2</b> <sub>(†15.5)</sub>      | <b>50.1</b> <sub>(†18.4)</sub> | $51.3_{(\uparrow 10.7)}$ | $33.6_{(\uparrow 9.7)}$ | <b>21.4</b> <sub>(†11.5)</sub> | $\underline{36.0}_{(\uparrow 9.3)}$ |
| InternVL3.5-8B         | 35.5                           | 25.7                           | 13.2                                | 24.6                           | 22.1                     | 12.0                    | 5.6                            | 15.8                                |
| InternVL3.5-8B-G       | 59.5                           | 42.0                           | 20.2                                | 39.4                           | 35.9                     | 20.6                    | 9.0                            | 24.5                                |
| GroundVTS-I (ours)     | $61.2_{(\uparrow 1.7)}$        | 44.2 <sub>(↑2.2)</sub>         | $\underline{23.7}_{(\uparrow 3.5)}$ | 41.6 <sub>(↑2.2)</sub>         | $37.9_{(\uparrow 2.0)}$  | 22.4 <sub>(↑1.8)</sub>  | 10.3 <sub>(↑1.3)</sub>         | $25.7_{(\uparrow 1.2)}$             |

Bold denotes the best, <u>underlined</u> denotes the second-best. "-G" denotes supervised fine-tuning on the Grounding-FT dataset. "-Q" and "-I" denote our proposed models based on Qwen2.5VL-7B [3] and InternVL3.5-8B [48], respectively. ↑ indicates improvement over the corresponding "-G" baseline.

<span id="page-5-1"></span>Table 2. Comparison with state-of-the-art methods on QVHigh-lights validation split.

| Method           | M                              | IR .                     | HD                      |                                |  |  |
|------------------|--------------------------------|--------------------------|-------------------------|--------------------------------|--|--|
| Method           | R1@.5                          | R1@.7                    | mAP                     | Hit@1                          |  |  |
| SeViLA°[59]      | 54.5                           | 36.5                     |                         |                                |  |  |
| UniVTG°[30]      | <u>58.9</u>                    | 40.9                     | 27.0                    | 55.3                           |  |  |
| VTG-LLM[12]      |                                |                          | 16.5                    | 33.5                           |  |  |
| TimeChat[43]     |                                |                          | 14.5                    | 23.9                           |  |  |
| NumPro[55]       |                                |                          | 40.5                    | <u>70.7</u>                    |  |  |
| Qwen2.5VL-7B     | 8.7                            | 2.4                      | 24.9                    | 0.6                            |  |  |
| Qwen2.5VL-7B-G   | 11.0                           | 4.3                      | 34.4                    | 44.5                           |  |  |
| GroundVTS-Q      | $23.6_{(\uparrow 12.6)}$       | $12.3_{(\uparrow 8.0)}$  | $35.7_{(\uparrow 1.3)}$ | $58.8_{(\uparrow 14.3)}$       |  |  |
| InternVL3.5-8B   | 8.7                            | 3.7                      | 24.8                    | 0.32                           |  |  |
| InternVL3.5-8B-G | 31.8                           | 15.0                     | 31.9                    | 39.8                           |  |  |
| GroundVTS-I      | <b>63.6</b> <sub>(†31.8)</sub> | $40.7_{(\uparrow 25.7)}$ | 52.5 <sub>(†20.6)</sub> | <b>88.4</b> <sub>(↑48.6)</sub> |  |  |

o indicates classical expert models; other notations follow Table 1.

#### 4.2. Main Results

Moment Retrieval. As summarized in Tables 1 and 2, our proposed GroundVTS consistently outperforms existing state-of-the-art methods on multiple VTG benchmarks. On Charades-STA, GroundVTS-Q substantially outperforms the fine-tuned Qwen2.5VL-7B baseline, achieving gains of 24.8 points in R1@0.5 and 18.4 points in mIoU, reaching 57.5 R1@0.5 and 50.1 mIoU. On ActivityNet-Captions, GroundVTS-Q improves R1@0.5 by 9.7 points and mIoU by 9.3 points, further confirming the effectiveness of our sampling approach for VTG. Building upon InternVL3.5-8B, GroundVTS-I also shows stable improvements (*e.g.*, +3.5 in R1@0.7 on Charades-STA), validating the generality of our approach across diverse Vid-LLM architectures.

<span id="page-5-2"></span>Table 3. Comparison with state-of-the-art methods on NExT-GQA test splits.

| Model               | mIoU | mIoP | IoU@.5 | IoP@.5 | Acc@GQA |
|---------------------|------|------|--------|--------|---------|
| TOGA° [37]          | 24.4 | 40.5 | 21.1   | 40.6   | 24.6    |
| VideoStreaming [40] | 19.3 | 32.2 | 13.3   | 31.0   | 17.8    |
| GroundVTS-Q         | 25.8 | 37.4 | 20.4   | 35.4   | 23.2    |
| GroundVTS-I         | 16.7 | 26.5 | 11.9   | 24.3   | 18.5    |

On QVHighlights, GroundVTS-I attains 63.6 in R1@0.5 and 40.7 in R1@0.7 for moment retrieval, comparable to specialized methods such as UniVTG [30].

**Highlight Detection.** As shown in Table 2, GroundVTS-I significantly outperforms InternVL3.5-8B-G, improving mAP and Hit@1 by 20.6 and 48.6 points, respectively, to 52.5 and 88.4. It also surpasses strong methods using frame indices as auxiliary inputs, such as NumPro [55], suggesting better sensitivity to key moments in highlight detection.

**Out-of-Distribution Evaluation.** To further assess the effectiveness and generality of our method under task shift, we evaluate our models *as-is* on grounded video question answering with NExT-GQA [57], without any further training; results are shown in Table 3. GroundVTS-Q achieves the highest mIoU and remains competitive on other metrics, despite not being specifically designed or trained for this task. Moreover, the supplementary material reports two additional *as-is* evaluations: DiDeMo [2] for out-of-distribution moment retrieval, and LongVideoBench [54] for transfer to a new long-video understanding task; on both benchmarks, our models either outperform or remain competitive with recent state-of-the-art methods.

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

![](_page_6_Figure_1.jpeg)

![](_page_6_Figure_2.jpeg)

(b) Token efficiency under varying token densities.

Figure 4. Comparison between GroundVTS-Q and Qwen2.5VL-7B-G (denoted as OwenVL-G) under varying token densities.

#### 4.3. Effect of Visual Token Density

To verify the robustness of our GroundVTS with respect to visual token density, we conduct an analysis on the Charades-STA test split, comparing GroundVTS-Q with its fine-tuned base model, Qwen2.5VL-7B-G (abbreviated as QwenVL-G). We adjust the sampling ratio  $\rho$  from 0.1 to 1.0 in increments of 0.1, while keeping the dense sampling frame rate fixed at 2 FPS, to control the number of visual tokens involved in LLM inference. For a fair comparison, QwenVL-G continues to use uniform frame sampling, with the frame rate varied from 0.2 to 2 FPS in increments of 0.2.

As shown in Figure 4(a), the horizontal axis represents effective token density (FPS  $\times \rho$ ), and the vertical axis reports grounding accuracy in terms of R1@0.7. As the token density decreases, QwenVL-G degrades markedly, indicating a strong dependence on dense temporal sampling. In contrast, GroundVTS remains much more stable across the full density range, maintaining high accuracy even in sparse settings. With only half the token budget (FPS  $\times \rho$  = 1.0), GroundVTS achieves 34.2 R1@0.7, already surpassing QwenVL-G at full density (30.5 R1@0.7). Even under a more aggressive reduction (FPS  $\times \rho$  = 0.4), GroundVTS still attains 29.2 R1@0.7, exceeding QwenVL-G by 19.0 points. These results highlight the strong token efficiency and robustness of GroundVTS under sparse sampling.

Figure 4(b) illustrates token efficiency, defined as R1@0.7 divided by effective token density. When fewer visual tokens are available, GroundVTS-Q maintains higher efficiency than QwenVL-G, indicating more effective use of limited visual information. Exact values are provided in the supplementary material.

<span id="page-6-1"></span>Table 4. Ablation of different training-stage combinations for GroundVTS-Q on Charades-STA test split.

| Stage             | R1@0.3                                             | R1@0.5                         | R1@0.7                   | mIoU                           |
|-------------------|----------------------------------------------------|--------------------------------|--------------------------|--------------------------------|
| base <sup>‡</sup> | 34.2                                               | 18.8                           | 8.6                      | 22.1                           |
| None <sup>§</sup> | 8.6 <sub>(\psi25.6)</sub>                          | 5.0 <sub>(\psi13.8)</sub>      | $1.9_{(\downarrow 6.7)}$ | $5.6_{(\downarrow 16.5)}$      |
| 1                 | $31.2_{(\downarrow 3.0)}$                          | $20.5_{(\uparrow 1.7)}$        | $10.0_{(\uparrow 1.4)}$  | $20.9_{(\downarrow 1.2)}$      |
| 1, 2              | $45.8_{(\uparrow 11.6)}$                           | $28.8_{(\uparrow 10.0)}$       | $13.2_{(\uparrow 4.6)}$  | $30.1_{(\uparrow 8.0)}$        |
| 1, 3              | $49.1_{(\uparrow 14.9)}$                           | $32.5_{(\uparrow 13.7)}$       | $15.2_{(\uparrow 6.6)}$  | $32.4_{(\uparrow 10.3)}$       |
| 2, 3              | $69.4_{(\uparrow 35.2)}$                           | <u>53.0</u> (†34.2)            | $30.5_{(\uparrow 21.9)}$ | $47.4_{(\uparrow 25.3)}$       |
| 1, 2, 3           | <b>71.5</b> <sub>(<math>\uparrow</math>37.3)</sub> | <b>57.5</b> <sub>(↑38.7)</sub> | $34.2_{(\uparrow 25.6)}$ | <b>50.1</b> <sub>(↑28.0)</sub> |

<sup>&</sup>lt;sup>‡</sup> Arrowed values indicate absolute changes relative to the base model (Qwen2.5VL-7B). § "None" uses a randomly initialized VTS module.

## 4.4. Effect of the Progressive Optimization Strategy

Table 4 summarizes the effect of different training stages for GroundVTS-Q. Using an untrained VTS module (the "None" setting) causes a sharp drop across all metrics, showing that query-conditioned token sampling must be properly learned. Stage 1 (VTS Warm-up) largely recovers the base-model performance, indicating that VTS can be integrated without disrupting the original pipeline. Adding Stage 2 (Joint LoRA Adaptation) further improves performance, bringing gains of +11.6 in R1@0.3 and +8.0 in mIoU over the base model. Adding Stage 3 (Grounding Fine-tuning) yields the best results, reaching 71.5/57.5/34.2 R1@0.3/0.5/0.7 and 50.1 mIoU. The (1, 3) and (2, 3) variants remain below the full setting, confirming the importance of Stage 2 for large-scale adaptation to non-uniform token distributions and Stage 1 for stable initialization.

## 4.5. Ablation Study

We conduct ablation experiments on two key components of GroundVTS in Table 5: the visual token sampling strategy and the positional encoding used for temporal reasoning. All variants are evaluated under a matched token budget, equivalent to FPS = 2.0 and  $\rho = 0.5$ .

**Sampling Strategy.** We compare our query-guided token-level sampling with three alternatives: (a) *Uniform sampling*, implemented by evaluating Qwen2.5VL-7B-G at 1.0 FPS; (b) *Random sampling*, where 50% of visual tokens are randomly discarded; and (c) *Frame-level query selection*, where visual tokens within each frame are average-pooled to estimate frame-query relevance, and the top 50% frames are retained with all their tokens. Both the token-level and frame-level variants are trained with the same three-stage procedure, while the random variant is initialized from the token-level model after Stages 1 and 2 and trained only in Stage 3 with random dropping.

As shown in Table 5, our token-level VTS achieves the best performance on both datasets. On Charades-STA, it

Table 5. Ablation on sampling strategies and positional encoding (PE) in GroundVTS.

<span id="page-7-0"></span>

| VTS          | PE           | Sampling Methods |             | Charade     | s-STA       |             | ActivityNet-Captions |             |             |             |  |
|--------------|--------------|------------------|-------------|-------------|-------------|-------------|----------------------|-------------|-------------|-------------|--|
| V 1 3        | FE           | Sampling Methods | R1@0.3      | R1@0.5      | R1@0.7      | mIoU        | R1@0.3               | R1@0.5      | R1@0.7      | mIoU        |  |
| $\checkmark$ | ✓            | Token-Level      | 71.5        | 57.5        | 34.2        | 50.1        | 51.3                 | 33.6        | 21.4        | 36.0        |  |
| $\checkmark$ | $\checkmark$ | Frame-Level      | <u>61.7</u> | <u>44.9</u> | <u>23.3</u> | <u>41.6</u> | <u>43.7</u>          | <u>27.5</u> | <u>15.0</u> | <u>30.7</u> |  |
| _            | $\checkmark$ | Uniform          | 42.6        | 28.5        | 15.0        | 29.3        | 36.1                 | 19.5        | 7.5         | 23.4        |  |
| _            | $\checkmark$ | Random           | 54.9        | 35.0        | 16.3        | 35.7        | 40.3                 | 23.4        | 12.1        | 27.7        |  |
| $\checkmark$ | _            | Token-Level      | 15.1        | 7.0         | 2.7         | 9.5         | 22.2                 | 11.2        | 5.2         | 16.3        |  |

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

Figure 5. Qualitative comparison of temporal grounding predictions among GroundVTS-Q, Qwen2.5VL-7B-G, and Qwen2.5VL-7B.

improves mIoU over frame-level selection by 8.5 points (50.1 vs. 41.6) and reaches 57.5 R1@0.5. On ActivityNet-Captions, it also performs best, improving mIoU by 5.3 points over the frame-level variant (36.0 vs. 30.7). By contrast, both uniform and random sampling degrade performance, confirming the importance of query-guided fine-grained sampling for temporal grounding; random sampling nevertheless outperforms uniform sampling, possibly because it acts as data augmentation.

Effect of Positional Encoding. To assess the role of positional encoding, we remove position embeddings from GroundVTS while keeping training and inference settings fixed. As shown in Table 5, performance collapses on both datasets. On Charades-STA, mIoU drops from 50.1 to 9.5 and R1@0.5 from 57.5 to 7.0, with similarly severe degradation on ActivityNet-Captions. These results confirm the importance of temporal positional information in Ground-VTS, and validate our design choice of retaining the original relative positional embeddings for the selected tokens.

**Additional ablations** on training data and relevance estimation are provided in the supplementary material.

#### 4.6. Qualitative Study

Figure 5 shows a qualitative comparison on a Charades-STA example with the query "a person takes a book off a shelf." The red curve denotes the normalized token density produced by VTS, with higher values indicating stronger query relevance. GroundVTS-Q assigns most tokens to the early part of the video (roughly 0–13 s), which fully covers the ground-truth interval (6.2–12.0 s), while suppressing nearly all tokens in later frames. Based on these sampled tokens, GroundVTS-Q predicts 6.0–12.0 s, closely matching the ground truth. In contrast, Qwen2.5VL-7B-G predicts an earlier and less precise segment (4.5–10.3 s), while the base Qwen2.5VL-7B misses the target moment entirely. This shows that VTS focuses on relevant temporal regions for grounding. More results are provided in the supplement.

#### 5. Conclusion

In this paper, we present GroundVTS, a query-guided visual token sampling framework for video temporal grounding. Its core module, VTS, can be seamlessly integrated into mainstream Vid-LLMs via a progressive optimization strategy to better capture fine-grained temporal cues. Experiments show that GroundVTS consistently improves instruction-tuned base models and outperforms recent state-of-the-art methods. Further analyses confirm that Ground-VTS improves token utilization and maintains prediction stability across varying input densities.

**Acknowledgements**. The research of Liuyi Wang is supported in part by the National Natural Science Foundation

