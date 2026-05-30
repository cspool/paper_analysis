# <span id="page-18-0"></span>C GLUE Dataset Statistics

We present the dataset statistics of GLUE [Wang et al.](#page-16-3) [\(2019\)](#page-16-3) in the following table.

| Corpus                              | Task                                  | #Train | #Dev | #Test | #Label | Metrics               |  |  |  |  |
|-------------------------------------|---------------------------------------|--------|------|-------|--------|-----------------------|--|--|--|--|
|                                     | Single-Sentence Classification (GLUE) |        |      |       |        |                       |  |  |  |  |
| CoLA                                | Acceptability                         | 8.5k   | 1k   | 1k    | 2      | Matthews corr         |  |  |  |  |
| SST                                 | Sentiment                             | 67k    | 872  | 1.8k  | 2      | Accuracy              |  |  |  |  |
| Pairwise Text Classification (GLUE) |                                       |        |      |       |        |                       |  |  |  |  |
| MNLI                                | NLI                                   | 393k   | 20k  | 20k   | 3      | Accuracy              |  |  |  |  |
| RTE                                 | NLI                                   | 2.5k   | 276  | 3k    | 2      | Accuracy              |  |  |  |  |
| QQP                                 | Paraphrase                            | 364k   | 40k  | 391k  | 2      | Accuracy/F1           |  |  |  |  |
| MRPC                                | Paraphrase                            | 3.7k   | 408  | 1.7k  | 2      | Accuracy/F1           |  |  |  |  |
| QNLI                                | QA/NLI                                | 108k   | 5.7k | 5.7k  | 2      | Accuracy              |  |  |  |  |
|                                     | Text Similarity (GLUE)                |        |      |       |        |                       |  |  |  |  |
| STS-B                               | Similarity                            | 7k     | 1.5k | 1.4k  | 1      | Pearson/Spearman corr |  |  |  |  |

Table 10: Summary of the GLUE benchmark.

GLUE includes two single-sentence classification tasks: SST-2 [\(Socher et al.,](#page-15-10) [2013\)](#page-15-10) and CoLA [\(Warstadt et al.,](#page-16-7) [2019\)](#page-16-7), and three similarity and paraphrase tasks: MRPC [\(Dolan and Brockett,](#page-14-10) [2005\)](#page-14-10), STS-B [\(Cer et al.,](#page-13-3) [2017\)](#page-13-3), and QQP. GLUE also includes four natural language inference tasks in GLUE: MNLI [\(Williams et al.,](#page-16-8) [2018\)](#page-16-8), QNLI [\(Rajpurkar et al.,](#page-15-2) [2016\)](#page-15-2), RTE [\(Dagan et al.,](#page-13-4) [2007;](#page-13-4) [Bar-Haim et al.,](#page-13-5) [2006;](#page-13-5) [Giampiccolo et al.,](#page-14-11) [2007;](#page-14-11) [Bentivogli et al.,](#page-13-6) [2009\)](#page-13-6), and WNLI [\(Levesque et al.,](#page-14-12) [2012\)](#page-14-12).

