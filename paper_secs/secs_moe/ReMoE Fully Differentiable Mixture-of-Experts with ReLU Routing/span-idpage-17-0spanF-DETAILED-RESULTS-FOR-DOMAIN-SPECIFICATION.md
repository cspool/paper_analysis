# <span id="page-17-0"></span>F DETAILED RESULTS FOR DOMAIN SPECIFICATION

Figure [11](#page-18-0) shows the average routed tokens ratio of MoE and ReMoE across all layers. ReMoE demonstrates significantly stronger domain specialization compared to MoE, where certain experts are more frequently activated for specific domains. This suggests that ReMoE is better at learning and exploiting the unique characteristics of different domains, allowing it to allocate computational resources more effectively. In contrast, MoE exhibits a more uniform expert activation across domains, indicating less differentiation in its expert specialization.

<span id="page-18-0"></span>![](_page_18_Figure_1.jpeg)

Figure 11: Detailed results of average routed tokens ratio for MoE and ReMoE in different domains.

We further analyze the experts in Layer 5 of ReMoE and observe that certain highly related, domainspecific vocabularies are consistently routed to the same expert. To investigate this, we calculate the routing probabilities of different tokens based on their IDs, defined as the ratio of the number of times a specific expert is utilized to the total occurrences of the token. The results are summarized in Table [12.](#page-19-1)

Our findings reveal that the vocabularies exhibit clear specialization, reflecting domain-specific characteristics. For example, Expert 1, which is more frequently assigned to natural language domains (e.g., Books, C4), tends to route tokens such as husband, wife, and lover. In contrast, Expert 6, which is associated with non-natural language domains (e.g., Arxiv, Github, StackExchange), predominantly routes code-related tokens like variable, env, and HEAD.

<span id="page-19-1"></span>

| Expert ID | Routed Tokens With High Probability                                                                                                                      |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0         | End(100%);<br>folding(100%);<br>Fill(100%);<br>FILE(100%);<br>NULL(100%);<br>byte(100%);<br>Release(99.36%);<br>Del(99.80%)                              |
| 1         | husband(100%);<br>ife(100%);<br>baby(100%);<br>human(100%);<br>lover(99.60%);<br>).(99.86%);<br>),(99.71%);<br>)(98.425%)                                |
| 2         | invest(100%);<br>Fortune(100%);<br>exec<br>(100%); 0000(100%);<br>Sorry(100%);<br>bye(97.82%);<br>If(97.74%);<br>®(97.63%)                               |
| 3         | Conversely(100%);<br>Methods(100%);<br>flower(100%);<br>Blossom(99.93%);<br>Argentina(100%);<br>Georgian(100%);<br>Uruguay(98.90%);<br>African<br>(100%) |
| 4         | Spring(100%);<br>Summer(100%)<br>Autumn(100%);<br>Winter(100%);<br>seasons(99.02%);<br>Temperature<br>(100%); hot(97.98%);<br>cold(100%)                 |
| 5         | `e(100%);<br>˚a(98.59%);<br>æ(99.80%);<br>Æ(97.67%)                                                                                                      |
| 6         | ]);(100%);<br>gif(100%);<br>size(100%);<br>variable(100%);<br>env(100%);<br>begin(97.95%);<br>HEAD(97.94%);<br> (97.83%)                                 |
| 7         | Kuala(100%);<br>Tus(100%);<br>Lama(100%);<br>Riley(98.94%)                                                                                               |

Table 12: Routed tokens with high probability for experts in Layer 5 of ReMoE

