# C IMPROVED MMTOK

Since LLaVA-1.5 does not fine-tune the vision tower and also does not mask padding patches, we explicitly exclude padding patches from the candidate token set and fix an overflow bug that wasted one token. As shown in Table [15,](#page-16-0) these changes substantially improve accuracy while using fewer tokens. For a fair comparison, we report the results without the fix in the main text.

<span id="page-16-0"></span>

| Method                                                     | GQA<br>Acc. ↑ | MMB<br>Acc. ↑                 | MME<br>P+C ↑ | POPE<br>F1 ↑ | SEED-I<br>Acc. ↑ | Avg<br>↑ |  |  |  |  |  |  |
|------------------------------------------------------------|---------------|-------------------------------|--------------|--------------|------------------|----------|--|--|--|--|--|--|
|                                                            |               | Vanilla Baseline (576 tokens) |              |              |                  |          |  |  |  |  |  |  |
| LLaVA-1.5-7B                                               | 61.9          | 64.7                          | 1862         | 85.9         | 66.14            | 100.0%   |  |  |  |  |  |  |
| 32 Tokens                                                  |               |                               |              |              |                  |          |  |  |  |  |  |  |
| 58.59<br>1625<br>82.95<br>59.81<br>MMTok<br>55.95<br>91.0% |               |                               |              |              |                  |          |  |  |  |  |  |  |
| MMTok++                                                    | 56.61         | 58.76                         | 1636         | 83.44        | 59.85            | 91.6%    |  |  |  |  |  |  |
|                                                            |               |                               | 16 Tokens    |              |                  |          |  |  |  |  |  |  |
| MMTok                                                      | 53.31         | 54.30                         | 1551         | 79.79        | 56.67            | 86.4%    |  |  |  |  |  |  |
| MMTok++                                                    | 54.05         | 54.98                         | 1581         | 80.79        | 57.13            | 87.5%    |  |  |  |  |  |  |
|                                                            | 8 Tokens      |                               |              |              |                  |          |  |  |  |  |  |  |
| MMTok                                                      | 49.06         | 49.06                         | 1355         | 78.46        | 52.74            | 79.8%    |  |  |  |  |  |  |
| MMTok++                                                    | 50.80         | 49.31                         | 1395         | 79.75        | 53.59            | 81.4%    |  |  |  |  |  |  |
|                                                            |               |                               | 4 Tokens     |              |                  |          |  |  |  |  |  |  |
| MMTok                                                      | 43.93         | 36.94                         | 1290         | 74.84        | 48.10            | 71.4%    |  |  |  |  |  |  |
| MMTok++                                                    | 45.08         | 40.21                         | 1294         | 76.36        | 49.34            | 73.6%    |  |  |  |  |  |  |
|                                                            |               |                               | 2 Tokens     |              |                  |          |  |  |  |  |  |  |
| MMTok                                                      | 40.58         | 25.69                         | 1122         | 68.95        | 42.89            | 62.1%    |  |  |  |  |  |  |
| MMTok++                                                    | 42.18         | 31.36                         | 1237         | 72.97        | 45.27            | 67.3%    |  |  |  |  |  |  |
|                                                            |               |                               | 0 Tokens     |              |                  |          |  |  |  |  |  |  |
| Baseline                                                   | 37.65         | 19.33                         | 971          | 44.64        | 37.03            | 50.2%    |  |  |  |  |  |  |

Table 15: Evaluate MMTok++ on LLaVA-1.5-7B with Extremely Less Token Budgets.

