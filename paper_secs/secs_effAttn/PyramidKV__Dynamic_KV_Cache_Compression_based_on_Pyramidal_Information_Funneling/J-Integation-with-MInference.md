# J Integation with MInference

We would like to clarify that PyramidKV and MInference Jiang et al. (2024) are complementary approaches addressing different aspects of KV cache optimization. Specifically:

 MInference focuses on accelerating the generation of KV caches during the prefilling stage of LLM inference. In contrast, PyramidKV targets efficient KV cache management during LLM decoding.

To evaluate their respective strengths, we compared PyramidKV and MInference on Longbench using a KV cache size of 128. The results demonstrated the superior performance of PyramidKV.

Furthermore, we demonstrate that MInference and PyramidKV can be seamlessly integrated to achieve highly efficient inference while maintaining performance comparable to full attention. The results of MInference combined with PyramidKV, evaluated on Longbench with a KV cache size of 128, as PyramidKV + MInference hybrid approach.

|                                    | Single | -Docum                  | ent QA                  | Multi                   | -Documen                | t QA                    | Su                      | mmariza                 | tion                    | Few                  | v-shot Le               | arning                  | Synth        | etic C                                 | Code              | Avg.             |
|------------------------------------|--------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|----------------------|-------------------------|-------------------------|--------------|----------------------------------------|-------------------|------------------|
| Stra.                              | NrtvQA | Qasper                  | MF-en                   | HotpotQA                | 2WikiMQA                | Musique                 | GovReport               | OMSum                   | MultiNew                | TREC                 | TriviaQA                | SAMSum                  | PCount       | PRE LCC                                | RB-P              |                  |
| PyramidKV<br>MInference<br>M. + P. |        | 20.61<br>30.63<br>31.74 | 38.28<br>40.41<br>39.98 | 43.23<br>44.28<br>43.10 | 31.62<br>35.22<br>35.21 | 20.94<br>20.65<br>21.60 | 21.27<br>28.43<br>27.41 | 22.69<br>23.35<br>23.06 | 22.83<br>26.75<br>26.76 | 71<br>72.00<br>73.00 | 90.48<br>87.90<br>88.03 | 39.86<br>42.78<br>43.36 | 5.83<br>6.30 | 69.25 56.9<br>64.00 58.7<br>64.00 58.5 | 4 50.16<br>6 5.06 | 5 39.31<br>38.86 |

Table 7: Comparison between PyramidKV, MInference and MInference-PyramidKV hybrid method.

In summary, we demonstrate that PyramidKV outperforms MInference on Longbench. Furthermore, when integrated with MInference, PyramidKV enhances its performance even further.

### K Comparison with PyramidInfer

Our work differs from PyramidInfer in two key aspects:

- **Decay Strategy**: While PyramidInfer Yang et al. (2024) employs a geometric decay strategy, our method adopts an arithmetic decay strategy. We argue that the relatively stable and linear nature of arithmetic decay better aligns with the behavior of the attention mechanism. This strategy is derived from empirically observed attention patterns, aiming to closely match them. Notably, our approach also achieves superior results, as demonstrated in the experimental results presented in the table
- Token Selection: PyramidInfer discards tokens in earlier layers, preventing them from being reconsidered in later layers. In contrast, our method allows previously discarded tokens to be re-evaluated in higher layers, recognizing that these tokens may still hold relevance at different stages of the model's processing.
- Pyramidal Information Funneling Pattern: A key contribution of our work lies in identifying and leveraging the pyramidal information funneling phenomenon within attention mechanisms. Through in-depth analysis, we observe that attention tends to disperse in earlier layers and progressively concentrates on crucial tokens in higher layers. This insight forms the foundation of our arithmetic decay strategy, ensuring that our method aligns more naturally with these intrinsic patterns.

Despite some similarities between the two approaches, these differences lead to significantly distinct outcomes. As shown in Table 8, our method consistently outperforms PyramidInfer, highlighting the effectiveness of our design choices.

|            | Single   | -Docum | ent QA | Mult   | i-Documer | ıt QA    | Su    | ımmariza | tion  | Few-  | shot Lear | ning  | Syntl | netic    | Code       | Avg.    |
|------------|----------|--------|--------|--------|-----------|----------|-------|----------|-------|-------|-----------|-------|-------|----------|------------|---------|
| Stra.      | 4Qvari   | asper  | MF-en  | rootQA | 2WikiMQA  | , usique | Repor | MSum     | HiNew | TREC  | AQaiv:    | MSum  | Count | PRe 1    | cc RB-1    | ?       |
|            | 17,      | Qr.    | 141    | Hou    | 51N11     | Mr       | Gov.  | Qr.      | Mur   | 1. 1  | rr. 5     | PC.   | γC    |          | <b>y</b> - |         |
| Pyramidinf | er 20.42 | 12.77  | 25.21  | 35.81  | 25.83     | 16.88    | 18.27 | 21.78    | 18.52 | 51.00 | 88.54     | 35.76 | 5.61  | 69.25 53 | .21 44.12  | 2 33.94 |
| PyramidKV  | 7 21.13  | 14.18  | 30.26  | 35.12  | 23.76     | 16.17    | 18.33 | 21.65    | 19.23 | 58.00 | 88.31     | 37.07 | 5.23  | 69.50 52 | .61 45.7   | 4 34.76 |

<span id="page-19-0"></span>Table 8: Comparison between PyramidKV and Pyramidinfer.

