# <span id="page-16-2"></span><span id="page-16-1"></span>E. Additional Experimental Results

We additionally present experimental results supporting the design of WorldMM, including ablation studies on backbone configurations (Sec. E.1) and the impact of temporal scales (Sec. E.2).

#### <span id="page-16-5"></span>E.1. Generalization to Different Backbones

To evaluate the flexibility and robustness of WorldMM across different backbone models, we conduct experiments with a diverse set of configurations. Specifically, in addition to the setup based on the GPT-5 model series and VLM2Vec-V2, we further incorporate Gemini 3 Flash [9] and Qwen3-VL-Embedding-2B [17]. As shown in Tab. 11, WorldMM demonstrates strong robustness to backbone selection, with the Gemini-based variant even outperforming others on EgoLifeQA. These results highlight that WorldMM generalizes well across different backbone architectures and can be seamlessly integrated with a wide range of state-of-the-art models without requiring architecture-specific modifications.

<span id="page-16-7"></span>Table 11. Performance of WorldMM with various backbones.

| Model                         | EgoLifeQA | LVBench | VideoMME (L) |
|-------------------------------|-----------|---------|--------------|
| WorldMM-Gemini + Qwen3-VL-Emb | 67.4      | 61.5    | 74.9         |
| WorldMM-Gemini + VLM2Vec-V2   | 68.2      | 61.7    | 75.8         |
| WorldMM-GPT + Qwen3-VL-Emb    | 66.0      | 61.4    | 75.8         |
| WorldMM-GPT + VLM2Vec-V2      | 65.6      | 61.9    | 76.6         |

## <span id="page-16-6"></span>E.2. Impact of Temporal Scales

While multiscale episodic memory improves overall performance, we verify that these gains result from the multiscale architecture rather than specific temporal constraints. The temporal scales used in our experiments are chosen based on empirical statistics of real-world event durations. To assess the sensitivity of WorldMM to these specific values, we introduce perturbations to the temporal scales and report the performance on EgoLifeQA in Tab. 12. The results demonstrate that WorldMM maintains consistent performance across these variations, indicating that the improvements stem from the multiscale memory design itself, rather than a reliance on precisely calibrated temporal windows

<span id="page-16-8"></span>Table 12. Performance with different episodic timescales.

| Temporal Scale | Acc  |
|----------------|------|
| 20s/2m/5m/50m  | 65.2 |
| 30s/3m/10m/1h  | 65.6 |
| 1m/5m/15m/1.5h | 64.8 |

## <span id="page-16-3"></span>F. Qualitative Results

In this section, we qualitatively analyze WorldMM's memory construction (Sec. F.1) and its multi-turn reasoning and refinement capabilities (Sec. F.2).

