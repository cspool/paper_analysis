# E. Implementation Details

For all our experiments, we use the popular Huggingface models with the recently released KVPress library [\(Jegou &](#page-9-9) [Jeblick,](#page-9-9) [2024\)](#page-9-9).

<span id="page-12-0"></span>![](_page_12_Figure_1.jpeg)

Figure 14: Performance of Llama-3.1-8B-Instruct using several KV Cache compression methods on individual tasks from the Ruler dataset (with length 8192) as compression ratio evolves. We report prompt compression methods using dotted lines for comparison.

| Text Sample (Context)                                                                                                                                                                                                                                                                                  | Q-Filters | K-Norm      | Streaming-LLM |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|-------------|---------------|
| One of the show's first longest-running storylines was<br>the rivalry between a young manicurist Jill Foster Ab<br>bott (Brenda Dickson, Jess Walton) and wealthy so<br>cialite, Katherine Chancellor (Jeanne Cooper). []<br>After much investigation, it is revealed that Kay is Jill's<br>biological | mother    | father      | father        |
| Both extreme right-wing leaders taught and practised<br>the theology of Christian Identity, a belief system which<br>the FBI includes on its watch list as an extremist reli<br>gion. [] Here, the group trained an estimated 1,500<br>of like-minded Christian                                        | Identity  | fundamental | fundamental   |
| The Viral Fever<br>[] TVF debuted their platform, releasing the final<br>two episodes of Pitchers on TVFPlay. [] TVF claims<br>to have worked with over 150 brands. [] The show<br>has been on hold as writer Biswapati Sarkar focuses<br>on writing web series, including the sequel to TV            | F         | _show       | _show         |

Table 2: Next-token generation examples for different KV Cache Compression methods, applied to Wikipedia article. Passages in bold correspond to useful information that is necessary to resolve the ambiguity in the choice of the next token.