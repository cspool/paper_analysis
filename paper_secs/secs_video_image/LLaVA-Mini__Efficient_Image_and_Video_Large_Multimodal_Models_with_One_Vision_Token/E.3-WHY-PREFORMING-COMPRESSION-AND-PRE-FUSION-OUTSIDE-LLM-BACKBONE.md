# E.3 WHY PREFORMING COMPRESSION AND PRE-FUSION OUTSIDE LLM BACKBONE?

LLaVA-Mini performs compression and modality pre-fusion before the LLM backbone. The motivation for conducting these processes outside the LLM backbone, rather than conducting at the L th layer within the LLM, stems from two key considerations:

- Vision representations after the L th layers contain contextual information, which hinders the compression module: After the vision tokens are fed into the LLM, the early layers cause the visual representations to carry contextual information. Applying query-based compression on top of these representations makes it difficult for the compression module to distinguish between different vision tokens.
- The inter-layer operations within the LLM may not be compatible with existing acceleration frameworks: One of the main motivations for placing the compression and pre-fusion modules outside the LLM backbone in LLaVA-Mini is to keep the LLM backbone unchanged. This design allows for compatibility with nearly all existing LLM acceleration technologies and frameworks, further enhancing efficiency.

<span id="page-21-1"></span>Table 12: Comparison of performing compression and pre-fusion outside or within LLM backbone.

|                                                               |                |           | Performance |      |      |
|---------------------------------------------------------------|----------------|-----------|-------------|------|------|
| Methods                                                       | #Vision Tokens | FLOPs (T) | VQAv2       | GQA  | MMB  |
| LLaVA-Mini                                                    | 1              | 1.96      | 77.6        | 60.9 | 65.6 |
| LLaVA-Mini (perform compression<br>and pre-fusion within LLM) | 1              | 1.84      | 76.3        | 60.1 | 64.5 |

We also conduct a comparison between LLaVA-Mini and LLaVA-Mini (compression and pre-fusion within LLM) in Table [12.](#page-21-1) The results demonstrate that the configuration of LLaVA-Mini is more advantageous. We will incorporate this result and the architectural motivation into the manuscript as per your recommendation.

