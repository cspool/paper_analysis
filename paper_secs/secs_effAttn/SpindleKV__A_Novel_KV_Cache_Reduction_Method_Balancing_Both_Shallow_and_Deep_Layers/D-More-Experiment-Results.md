# **D** More Experiment Results

#### <span id="page-12-3"></span>D.1 Accuracy on Long Context Tasks

The detail *LongBench* results of LLaMA2-7b-chat and LLaMA3-8b-instruct are depicted in Table 10 and Table 11. Overall, SpindleKV outperforms baselines cross various reserve ratio of KV cache.

#### **D.2** LongBench Results On More Baselines

We also compared SpindleKV with three additional KV cache compression techniques on LongBench: H2O, SnapKV, and StreamingLLM. The average scores across 16 datasets for LLaMA3-8B-Instruct are showed in Table 8, which demonstrate the superiority of SpindleKV as a promising solution for KV cache compression compared to other method.

![](_page_12_Figure_11.jpeg)

<span id="page-12-4"></span>Figure 7: Cosine similarity in Key.( $\theta = 0.9$ )

![](_page_12_Figure_13.jpeg)

<span id="page-12-5"></span>Figure 8: Cosine similarity in Value.( $\theta = 0.6$ )

<span id="page-12-6"></span>

| Abbreviation | Full Name            |
|--------------|----------------------|
| Na.QA        | narrativeqa          |
| Qasp         | qasper               |
| Mu.QA        | multifieldqa_en      |
| Ho.QA        | hotpotqa             |
| Wi.QA        | 2wikimqa             |
| Musq         | musique              |
| Gv.Rp        | gov_report           |
| QMSm         | qmsum                |
| M.New        | multi_news           |
| TREC         | trec                 |
| Tr.QA        | triviaqa             |
| SASm         | samsum               |
| PCnt         | passage_count        |
| Pa.Rt        | passage_retrieval_en |
| Lcc          | lcc                  |
| RB.P         | repobench-p          |

Table 7: Abbreviation to Dataset Mapping

<span id="page-13-0"></span>

| Reserve Ratio | H2O   | SnapKV | StreamingLLM | SpindleKV |
|---------------|-------|--------|--------------|-----------|
| 20.0%         | 36.12 | 39.56  | 37.46        | 40.10     |
| 40.1%         | 37.34 | 40.48  | 39.29        | 41.13     |

<span id="page-13-1"></span>Table 8: Accuracy of *LongBench* on LLaMA3-8b-instruct with more baselines.

| Model              | FullKV        | SpindleKV with 40% Cache |
|--------------------|---------------|--------------------------|
| LLaMA3-8B-Instruct | 22.16 token/s | 18.39 token/s            |
| Mistral-7B         | 22.48 token/s | 18.47 token/s            |

Table 9: Decoding speed (token/s) comparison between FullKV and SpindleKV with 40% cache.

## D.3 Inference Speed

We measured the latency on LLaMA3-8B-Instruct and Mistral-7B using a context length of 4096 and a generation length of 1000 on a signal 3090 GPU. The inference speed (token/s) are showed in Table [9,](#page-13-1) our method does not introduce significant additional time overhead during inference, which is consistent with the analysis presented in our paper.

<span id="page-14-0"></span>

| Methods      | Ratio |             | Single-Document QA |       |       | Multi-Document QA |       |             | Summarization |       |       | Few-shot Learning |       |      | Synthetic        |             | Code              | AVG.  |
|--------------|-------|-------------|--------------------|-------|-------|-------------------|-------|-------------|---------------|-------|-------|-------------------|-------|------|------------------|-------------|-------------------|-------|
|              |       | Na.QA       | Qasp               | Mu.QA | Ho.QA | Wi.QA             | Musq  | Gv.Rp       | QMSm          | M.New | TREC  | Tr.QA             | SASm  | PCnt | Pa.Rt            | Lcc         | RB.P              |       |
| FullKV       | 100%  | 24.50       | 31.50              | 39.36 | 43.70 | 36.23             | 21.60 | 28.50       | 23.40         | 26.39 | 74.00 | 90.48             | 42.82 | 4.77 | 69.50            | 59.21       | 54.02             | 41.87 |
| PyramidInfer | 39.3% | 19.18       | 26.47              | 22.67 | 38.92 | 27.64             | 18.25 | 25.02       | 21.52         | 24.05 | 66.50 | 90.38             | 40.68 | 1.58 | 49.50            | 48.88       | 48.78             | 35.62 |
| PyramidKV    |       | 40.6% 24.73 | 20.75              | 35.56 | 44.00 | 32.74             | 20.77 | 23.47       | 22.62         | 22.10 | 72.00 | 90.33             | 40.61 |      | 5.77 69.25       |             | 58.28 54.85 39.86 |       |
| SpindleKV    | 39.1% | 23.87       | 26.40              | 39.02 | 44.38 | 36.02             | 22.12 | 26.10       | 23.28         | 24.24 | 72.00 | 90.43             | 41.49 |      | 5.23 69.50       |             | 59.37 54.68 41.13 |       |
| PyramidInfer | 31.3% | 19.58       | 23.23              | 21.60 | 36.24 | 24.45             | 16.79 | 24.31       | 21.31         | 22.78 | 62.50 | 89.74             | 40.17 | 2.20 | 49.00            | 48.04       | 49.43             | 34.46 |
| PyramidKV    | 30.5% | 23.02       | 20.24              | 33.66 | 44.50 | 30.27             | 20.95 | 22.60       | 22.77         | 21.40 | 71.50 | 90.24             | 40.47 | 5.83 |                  | 69.50 58.94 | 54.69             | 39.41 |
| SpindleKV    |       | 29.3% 24.18 | 25.71              | 37.44 | 43.42 | 34.95             | 21.97 | 25.13       | 23.52         | 23.13 | 72.00 | 90.43             | 41.47 |      | 5.24 69.50       | 59.24       | 60.04             | 41.08 |
| PyramidInfer | 26.0% | 18.80       | 21.61              | 17.34 | 33.47 | 22.29             | 13.95 | 23.32       | 21.08         | 22.01 | 61.00 | 87.85             | 40.28 | 2.25 | 32.50            | 49.57       | 50.83             | 32.38 |
| PyramidKV    | 24.2% | 24.13       | 19.70              | 33.08 | 43.32 | 30.86             | 21.08 | 22.16       | 23.02         | 20.59 | 72.00 | 90.16             | 40.00 |      | 5.37 69.50 58.48 |             | 53.92             | 39.21 |
| SpindleKV    |       | 23.6% 24.66 | 23.73              | 34.65 | 43.54 | 33.74             | 22.15 | 24.44       | 23.11         | 22.92 | 71.50 | 90.56             | 41.38 | 5.58 | 69.50            | 58.57       | 54.13             | 40.26 |
| PyramidInfer | 21.7% | 16.98       | 15.74              | 17.47 | 31.30 | 22.74             | 14.68 | 23.09       | 20.72         | 21.31 | 54.50 | 84.45             | 40.32 | 2.36 | 21.00            | 53.22       | 51.01             | 30.68 |
| PyramidKV    | 21.6% | 23.77       | 18.77              | 34.46 | 42.84 | 30.46             | 21.00 | 22.19       | 22.98         | 20.23 | 72.50 | 90.18             | 40.05 | 5.70 |                  | 69.50 57.33 | 53.83             | 39.11 |
| SpindleKV    |       | 21.2% 23.92 | 23.16              | 35.87 | 43.52 | 33.90             | 21.20 | 24.27       | 22.66         | 22.46 | 71.50 | 90.56             | 41.37 |      | 5.58 69.50       | 58.26       | 53.93             | 40.10 |
| PyramidInfer | 16.6% | 15.60       | 16.31              | 15.89 | 30.04 | 20.58             | 10.43 | 22.55       | 20.03         | 21.10 | 52.00 | 78.46             | 39.50 | 1.30 | 13.03            | 56.74       | 51.28             | 29.05 |
| PyramidKV    | 16.1% | 22.73       | 17.58              | 34.83 | 43.86 | 27.50             |       | 21.66 21.44 | 22.47         | 19.28 | 71.00 | 88.93             | 39.88 | 5.59 |                  | 69.50 56.62 | 53.39             | 38.51 |
| SpindleKV    |       | 16.0% 24.34 | 20.99              | 35.72 | 44.06 | 31.29             |       | 20.52 23.22 | 22.79         | 21.90 | 71.50 | 90.33             | 40.60 |      | 5.52 69.50       | 57.94       | 53.93             | 39.63 |

Table 10: LongBench Results of LLaMA3-8b-instruct.

<span id="page-14-1"></span>

| Methods      | Ratio  |       | Single-Document QA |       |       | Multi-Document QA |       |             | Summarization |       |       | Few-shot Learning |       |      | Synthetic  |             | Code                         | AVG.        |
|--------------|--------|-------|--------------------|-------|-------|-------------------|-------|-------------|---------------|-------|-------|-------------------|-------|------|------------|-------------|------------------------------|-------------|
|              |        | Na.QA | Qasp               | Mu.QA | Ho.QA | Wi.QA             | Musq  | Gv.Rp       | QMSm          | M.New | TREC  | Tr.QA             | SASm  | PCnt | Pa.Rt      | Lcc         | RB.P                         |             |
| FullKV       | 100%   | 18.39 | 20.14              | 35.67 | 30.92 | 25.73             | 10.64 | 25.58       | 20.98         | 26.43 | 64.00 | 83.38             | 41.02 | 5.50 | 10.00      | 60.81       | 55.12                        | 33.39       |
| PyramidInfer | 40.7%  | 15.36 | 15.40              | 19.23 | 29.14 | 24.53             | 7.49  | 21.64       | 19.66         | 22.70 | 54.00 | 81.79             | 40.71 | 4.00 | 3.50       | 54.29       | 51.98                        | 29.09       |
| PyramidKV    | 41.3%  | 18.38 | 20.99              | 35.98 | 30.76 | 25.45             |       | 10.79 23.73 | 20.88         | 25.08 | 64.00 | 83.75             | 41.17 | 6.00 | 10.50      | 60.58       |                              | 54.93 33.31 |
| SpindleKV    | 41.1%  | 18.45 | 21.23              | 36.67 | 30.80 | 25.74             |       | 10.62 24.49 | 20.66         | 25.18 | 64.00 | 84.31             | 41.11 |      | 6.00 10.00 | 60.45       |                              | 54.92 33.41 |
| PyramidInfer | 31.2%  | 13.80 | 15.27              | 17.69 | 27.69 | 26.10             | 7.27  | 20.53       | 19.42         | 22.04 | 53.50 | 77.06             | 40.50 | 2.00 | 6.50       | 52.69       | 50.46                        | 28.28       |
| PyramidKV    | 30.8%  | 17.78 | 20.49              | 36.86 | 30.55 | 26.04             | 9.93  | 22.92       | 20.97         | 24.13 | 64.00 | 83.59             | 41.06 | 6.00 | 11.00      | 60.74       | 54.15                        | 33.13       |
| SpindleKV    | 30.8%  | 18.13 | 19.95              | 36.91 | 30.77 | 26.10             | 9.91  | 23.19       | 20.95         | 24.35 | 64.00 | 83.69             | 41.37 | 6.00 | 9.00       | 59.85       | 53.77                        | 33.00       |
| PyramidInfer | 25.9%  | 14.85 | 15.19              | 15.41 | 26.82 | 24.95             | 5.67  | 19.96       | 18.66         | 21.34 | 49.00 | 76.22             | 38.83 | 4.50 | 5.00       | 51.87       | 51.71                        | 27.50       |
| PyramidKV    | 26.3%  | 17.32 | 20.99              | 36.37 | 30.86 | 25.62             | 9.80  | 22.38       | 20.57         | 23.24 | 64.00 | 83.81             | 40.82 | 6.00 |            |             | 10.50 59.95 54.40 32.91      |             |
| SpindleKV    | 26.0%  | 17.31 | 20.72              | 36.86 | 31.09 | 25.80             | 10.00 | 22.93       | 21.32         | 23.94 | 64.00 | 84.00             | 40.97 |      |            |             | 6.00 10.00 60.92 53.70 33.10 |             |
| PyramidInfer | 21.10% | 13.39 | 14.44              | 13.23 | 30.97 | 27.24             | 8.27  | 19.67       | 18.71         | 20.49 | 43.50 | 70.34             | 37.94 | 3.00 | 2.00       | 52.44       | 51.41                        | 26.69       |
| PyramidKV    | 22.3%  | 17.77 | 21.77              | 35.70 | 30.78 | 25.96             | 9.99  | 21.74       | 20.50         | 23.48 | 64.00 | 83.80             | 40.33 | 6.00 |            | 10.00 59.69 | 53.83                        | 32.83       |
| SpindleKV    | 22.1%  | 17.34 | 21.20              | 35.81 | 30.69 | 25.64             | 9.90  | 22.31       | 20.64         | 23.17 | 64.00 | 84.15             | 40.65 | 6.00 | 10.00      | 60.31       | 54.49                        | 32.89       |
| PyramidInfer | 15.9%  | 12.11 | 14.49              | 14.25 | 26.98 | 27.30             | 6.76  | 19.28       | 18.33         | 19.78 | 38.00 | 61.76             | 38.84 | 2.00 | 6.00       | 52.35       | 50.78                        | 25.56       |
| PyramidKV    | 16.8%  | 16.95 | 20.81              | 36.05 | 31.22 | 25.50             | 9.69  | 20.77       | 20.52         | 22.53 | 64.00 | 83.72             | 40.09 |      | 6.00 10.00 | 58.36       | 53.33                        | 32.47       |
| SpindleKV    | 16.7%  | 17.33 | 20.64              | 35.04 | 31.01 | 25.88             | 9.72  | 21.28       | 20.15         | 22.67 | 64.00 | 87.84             | 39.81 |      | 5.50 11.00 | 59.60       | 54.34                        | 32.86       |

Table 11: LongBench Results of LLaMA2-7b-chat.