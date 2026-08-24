# F LLM Usage

LLMs (specifically, ChatGPT and Claude) were used in the process of writing this paper for creating tables and figures, rephrasing, and proof-reading.

<span id="page-19-0"></span>

|                                     | Original | 4      | x     | 16     | ix    | 12     | 8x    | No Ctx |
|-------------------------------------|----------|--------|-------|--------|-------|--------|-------|--------|
|                                     |          | Single | Multi | Single | Multi | Single | Multi |        |
| Baseline Systems                    |          |        |       |        |       |        |       |        |
| LLMLingua2 (Qwen3-8B)               |          |        | 30.02 |        | 16.16 |        | 16.06 |        |
| ICAE (Mistral-7B)                   |          | 24.94  |       |        |       |        |       |        |
| PCC Lite (GPT2-Large & Llama3.1-8B) |          | 48.81  |       | 38.38  |       | 25.94  |       |        |
| PCC Large (Llama3.1-8B)             |          | 49.34  |       | 36.64  |       | 27.92  |       |        |
| Our Baselines                       |          |        |       |        |       |        |       |        |
| Qwen3-8B                            | 59.82    |        |       |        |       |        |       | 16.19  |
| Compression-Tokens (Causal)         |          | 51.59  | 50.01 | 41.26  | 42.99 | 34.21  | 32.50 |        |
| Compression-Tokens (Bidirectional)  |          | 53.44  | 53.99 | 44.92  | 47.15 | 33.60  | 34.27 |        |
| Mean-Pooling                        |          | 56.41  | 55.04 | 47.95  | 49.02 | 34.72  | 33.15 |        |
| Qwen3-4B                            | 58.87    |        |       |        |       |        |       | 13.74  |
| Compression-Tokens (Causal)         |          | 49.66  | 46.26 | 40.05  | 39.59 | 30.07  | 28.84 |        |
| Compression-Tokens (Bidirectional)  |          | 51.06  | 52.37 | 42.53  | 45.14 | 28.84  | 29.51 |        |
| Mean-Pooling                        |          | 55.25  | 53.77 | 45.43  | 45.86 | 30.91  | 28.38 |        |
| Qwen3-1.7B                          | 55.19    |        |       |        |       |        |       | 9.07   |
| Compression-Tokens (Causal)         |          | 36.66  | 41.64 | 35.49  | 34.64 | 24.27  | 24.04 |        |
| Compression-Tokens (Bidirectional)  |          | 46.45  | 46.72 | 36.62  | 38.93 | 24.31  | 24.33 |        |
| Mean-Pooling                        |          | 51.28  | 48.77 | 40.45  | 39.04 | 25.15  | 22.01 |        |
| Qwen3-0.6B                          | 50.85    |        |       |        |       |        |       | 4.78   |
| Compression-Tokens (Causal)         |          | 39.66  | 36.76 | 27.99  | 29.00 | 18.23  | 18.20 |        |
| Compression-Tokens (Bidirectional)  |          | 40.98  | 41.92 | 30.92  | 33.64 | 18.82  | 18.55 |        |
| Mean-Pooling                        |          | 45.82  | 43.07 | 32.50  | 33.09 | 19.05  | 16.07 |        |
| Gemma2-2B                           | 57.63    |        |       |        |       |        |       | 15.00  |
| Compression-Tokens (Causal)         |          | 47.90  | 45.98 | 39.57  | 39.74 | 32.02  | 29.66 |        |
| Compression-Tokens (Bidirectional)  |          | 49.43  | 49.40 | 40.89  | 42.70 | 31.79  | 30.43 |        |
| Mean-Pooling                        |          | 54.20  | 52.77 | 45.88  | 45.68 | 32.41  | 30.83 |        |
| Llama3.2-1B                         | 51.67    |        |       |        |       |        |       | 9.47   |
| Compression-Tokens (Causal)         |          | 41.84  | 39.03 | 33.73  | 33.38 | 24.11  | 24.64 |        |
| Compression-Tokens (Bidirectional)  |          | 43.46  | 42.96 | 35.04  | 35.46 | 24.91  | 24.56 |        |
| Mean-Pooling                        |          | 47.97  | 45.45 | 33.15  | 36.93 | 22.89  | 22.70 |        |

Table 9: Primary results with exact match (EM) as the metric.

<span id="page-20-0"></span>

|                                     | Original | 4:     | x     | 16     | ix    | 12     | 8x    | No Ctx |
|-------------------------------------|----------|--------|-------|--------|-------|--------|-------|--------|
|                                     |          | Single | Multi | Single | Multi | Single | Multi |        |
| Baseline Systems                    |          |        |       |        |       |        |       |        |
| LLMLingua2 (Qwen3-8B)               |          |        | 33.63 |        | 18.30 |        | 17.36 |        |
| ICAE (Mistral-7B)                   |          | 49.18  |       |        |       |        |       |        |
| PISCO (Llama3.1-8B)                 |          |        |       | 53.62  |       |        |       |        |
| PCC Lite (GPT2-Large & Llama3.1-8B) |          | 54.05  |       | 43.67  |       | 30.03  |       |        |
| PCC Large (Llama3.1-8B)             |          | 55.17  |       | 41.79  |       | 30.10  |       |        |
| Our Baselines                       |          |        |       |        |       |        |       |        |
| Owen3-8B                            | 68.84    |        |       |        |       |        |       | 17.98  |
| Compression-Tokens (Causal)         |          | 59.79  | 58.58 | 47.07  | 49.99 | 39.09  | 37.05 |        |
| Compression-Tokens (Bidirectional)  |          | 62.50  | 63.58 | 52.22  | 55.26 | 38.72  | 39.36 |        |
| Mean-Pooling                        |          | 65.91  | 65.06 | 55.68  | 56.77 | 39.95  | 37.80 |        |
| Qwen3-4B                            | 67.69    |        |       |        |       |        |       | 15.00  |
| Compression-Tokens (Causal)         |          | 57.12  | 54.17 | 46.06  | 45.47 | 34.83  | 33.48 |        |
| Compression-Tokens (Bidirectional)  |          | 59.35  | 60.99 | 48.92  | 52.23 | 33.20  | 34.33 |        |
| Mean-Pooling                        |          | 64.05  | 62.94 | 52.77  | 52.82 | 35.49  | 32.37 |        |
| Qwen3-1.7B                          | 64.21    |        |       |        |       |        |       | 9.85   |
| Compression-Tokens (Causal)         |          | 43.01  | 49.31 | 41.31  | 40.45 | 28.45  | 27.90 |        |
| Compression-Tokens (Bidirectional)  |          | 54.31  | 54.70 | 42.66  | 45.13 | 28.33  | 28.04 |        |
| Mean-Pooling                        |          | 60.03  | 57.28 | 46.63  | 45.07 | 28.96  | 25.82 |        |
| Qwen3-0.6B                          | 59.67    |        |       |        |       |        |       | 5.62   |
| Compression-Tokens (Causal)         |          | 46.36  | 43.62 | 33.07  | 34.30 | 21.13  | 21.71 |        |
| Compression-Tokens (Bidirectional)  |          | 48.04  | 48.84 | 36.11  | 39.50 | 22.13  | 22.00 |        |
| Mean-Pooling                        |          | 54.36  | 51.16 | 38.30  | 38.68 | 22.44  | 18.87 |        |
| Gemma2-2B                           | 66.14    |        |       |        |       |        |       | 16.80  |
| Compression-Tokens (Causal)         |          | 55.50  | 52.94 | 46.13  | 45.59 | 36.14  | 33.94 |        |
| Compression-Tokens (Bidirectional)  |          | 56.94  | 57.54 | 46.90  | 49.52 | 36.14  | 34.75 |        |
| Mean-Pooling                        |          | 62.51  | 61.28 | 52.55  | 51.90 | 36.61  | 34.93 |        |
| Llama3.2-1B                         | 60.30    |        |       |        |       |        |       | 11.09  |
| Compression-Tokens (Causal)         |          | 48.85  | 45.41 | 39.16  | 38.51 | 27.54  | 28.01 |        |
| Compression-Tokens (Bidirectional)  |          | 50.89  | 50.10 | 40.44  | 41.91 | 28.34  | 28.44 |        |
| Mean-Pooling                        |          | 56.62  | 53.50 | 38.84  | 42.76 | 25.99  | 26.19 |        |

Table 10: Primary results with accuracy as the metric.

<span id="page-21-0"></span>

|                                     | Original | 4      | x     | 16     | óχ    | 12     | 8x    | No Ctx |
|-------------------------------------|----------|--------|-------|--------|-------|--------|-------|--------|
|                                     |          | Single | Multi | Single | Multi | Single | Multi |        |
| Baseline Systems                    |          |        |       |        |       |        |       |        |
| LLMLingua2 (Qwen3-8B)               |          |        | 48.38 |        | 21.34 |        | 19.68 |        |
| ICAE (Mistral-7B)                   |          | 45.6   |       |        |       |        |       |        |
| PCC Lite (GPT2-Large & Llama3.1-8B) |          | 78.38  |       | 67.63  |       | 40.22  |       |        |
| PCC Large (Llama3.1-8B)             |          | 79.56  |       | 62.93  |       | 41.23  |       |        |
| Our Baselines                       |          |        |       |        |       |        |       |        |
| Qwen3-8B                            | 86.48    |        |       |        |       |        |       | 20.31  |
| Compression-Tokens (Causal)         |          | 77.11  | 74.89 | 57.05  | 62.12 | 44.56  | 42.35 |        |
| Compression-Tokens (Bidirectional)  |          | 80.00  | 81.23 | 64.80  | 69.27 | 44.30  | 43.86 |        |
| Mean-Pooling                        |          | 83.76  | 82.75 | 71.37  | 71.19 | 44.65  | 43.19 |        |
| Qwen3-4B                            | 85.75    |        |       |        |       |        |       | 17.72  |
| Compression-Tokens (Causal)         |          | 74.23  | 71.31 | 57.49  | 56.88 | 38.95  | 37.10 |        |
| Compression-Tokens (Bidirectional)  |          | 77.24  | 79.28 | 60.71  | 65.49 | 38.19  | 38.41 |        |
| Mean-Pooling                        |          | 83.19  | 81.54 | 68.20  | 67.47 | 39.96  | 37.54 |        |
| Qwen3-1.7B                          | 83.65    |        |       |        |       |        |       | 12.66  |
| Compression-Tokens (Causal)         |          | 54.25  | 64.50 | 49.09  | 49.98 | 31.78  | 30.10 |        |
| Compression-Tokens (Bidirectional)  |          | 72.92  | 73.39 | 53.07  | 57.36 | 31.58  | 31.17 |        |
| Mean-Pooling                        |          | 79.56  | 77.17 | 59.01  | 58.30 | 32.36  | 29.90 |        |
| Qwen3-0.6B                          | 81.55    |        |       |        |       |        |       | 7.91   |
| Compression-Tokens (Causal)         |          | 61.67  | 57.52 | 40.29  | 41.60 | 22.06  | 22.45 |        |
| Compression-Tokens (Bidirectional)  |          | 64.21  | 67.05 | 42.81  | 46.68 | 22.09  | 22.36 |        |
| Mean-Pooling                        |          | 74.00  | 70.60 | 48.62  | 48.14 | 22.40  | 20.45 |        |
| Gemma2-2B                           | 84.58    |        |       |        |       |        |       | 16.41  |
| Compression-Tokens (Causal)         |          | 70.61  | 69.06 | 55.75  | 56.37 | 37.89  | 35.66 |        |
| Compression-Tokens (Bidirectional)  |          | 74.16  | 75.41 | 57.77  | 61.75 | 37.72  | 37.00 |        |
| Mean-Pooling                        |          | 81.67  | 80.01 | 66.38  | 65.64 | 38.96  | 36.71 |        |
| Llama3.2-1B                         | 81.16    |        |       |        |       |        |       | 11.27  |
| Compression-Tokens (Causal)         |          | 62.65  | 59.34 | 46.41  | 45.49 | 28.58  | 28.44 |        |
| Compression-Tokens (Bidirectional)  |          | 64.48  | 65.61 | 48.99  | 51.39 | 29.09  | 28.82 |        |
| Mean-Pooling                        |          | 74.91  | 71.40 | 47.36  | 53.66 | 28.02  | 27.91 |        |

Table 11: SQuAD  $F_1$ .

<span id="page-22-0"></span>

|                                     | Original | 4      | x     | 16     | ix    | 12     | 8x    | No Ctx |
|-------------------------------------|----------|--------|-------|--------|-------|--------|-------|--------|
|                                     |          | Single | Multi | Single | Multi | Single | Multi |        |
| Baseline Systems                    |          |        |       |        |       |        |       |        |
| LLMLingua2 (Qwen3-8B)               |          |        | 53.54 |        | 29.32 |        | 26.27 |        |
| ICAE (Mistral-7B)                   |          | 50.01  |       |        |       |        |       |        |
| PCC Lite (GPT2-Large & Llama3.1-8B) |          | 68.55  |       | 59.38  |       | 43.93  |       |        |
| PCC Large (Llama3.1-8B)             |          | 70.08  |       | 59.05  |       | 46.46  |       |        |
| Our Baselines                       |          |        |       |        |       |        |       |        |
| Qwen3-8B                            | 84.67    |        |       |        |       |        |       | 26.65  |
| Compression-Tokens (Causal)         |          | 78.85  | 78.32 | 68.00  | 72.65 | 64.14  | 59.74 |        |
| Compression-Tokens (Bidirectional)  |          | 80.24  | 81.45 | 73.30  | 76.77 | 63.26  | 62.44 |        |
| Mean-Pooling                        |          | 83.30  | 82.08 | 77.66  | 78.41 | 63.88  | 63.77 |        |
| Qwen3-4B                            | 84.12    |        |       |        |       |        |       | 23.16  |
| Compression-Tokens (Causal)         |          | 76.48  | 75.56 | 68.85  | 69.80 | 59.08  | 55.78 |        |
| Compression-Tokens (Bidirectional)  |          | 78.13  | 79.41 | 71.11  | 74.60 | 58.38  | 56.87 |        |
| Mean-Pooling                        |          | 82.20  | 80.77 | 75.45  | 76.02 | 59.29  | 58.74 |        |
| Qwen3-1.7B                          | 80.95    |        |       |        |       |        |       | 18.75  |
| Compression-Tokens (Causal)         |          | 66.69  | 70.98 | 63.82  | 63.84 | 51.48  | 48.78 |        |
| Compression-Tokens (Bidirectional)  |          | 73.73  | 74.93 | 66.06  | 68.50 | 52.08  | 50.77 |        |
| Mean-Pooling                        |          | 78.64  | 76.11 | 68.76  | 68.78 | 50.86  | 49.44 |        |
| Qwen3-0.6B                          | 77.35    |        |       |        |       |        |       | 14.74  |
| Compression-Tokens (Causal)         |          | 66.62  | 65.76 | 55.88  | 57.79 | 43.16  | 39.58 |        |
| Compression-Tokens (Bidirectional)  |          | 67.24  | 69.28 | 58.44  | 61.79 | 43.80  | 41.66 |        |
| Mean-Pooling                        |          | 73.00  | 69.58 | 61.13  | 61.08 | 42.76  | 40.45 |        |
| Gemma2-2B                           | 82.55    |        |       |        |       |        |       | 25.18  |
| Compression-Tokens (Causal)         |          | 75.62  | 75.54 | 69.18  | 70.42 | 61.91  | 59.03 |        |
| Compression-Tokens (Bidirectional)  |          | 76.55  | 77.60 | 69.64  | 73.51 | 61.67  | 60.46 |        |
| Mean-Pooling                        |          | 80.93  | 79.85 | 75.10  | 74.90 | 62.36  | 61.64 |        |
| Llama3.2-1B                         | 77.96    |        |       |        |       |        |       | 19.34  |
| Compression-Tokens (Causal)         |          | 69.68  | 68.22 | 63.36  | 62.96 | 52.27  | 49.81 |        |
| Compression-Tokens (Bidirectional)  |          | 70.74  | 71.01 | 64.47  | 66.19 | 53.51  | 51.79 |        |
| Mean-Pooling                        |          | 74.38  | 72.69 | 62.75  | 66.59 | 51.05  | 50.86 |        |

Table 12: HotpotQA  $F_1$ .

<span id="page-23-0"></span>

|                                     | Original | 4x     |       | 16x    |       | 128x   |       | No Ctx |
|-------------------------------------|----------|--------|-------|--------|-------|--------|-------|--------|
|                                     |          | Single | Multi | Single | Multi | Single | Multi |        |
| Baseline Systems                    |          |        |       |        |       |        |       |        |
| LLMLingua2 (Qwen3-8B)               |          |        | 27.33 |        | 14.35 |        | 10.69 |        |
| ICAE (Mistral-7B)                   |          | 32.65  |       |        |       |        |       |        |
| PCC Lite (GPT2-Large & Llama3.1-8B) |          | 50.29  |       | 34.16  |       | 16.05  |       |        |
| PCC Large (Llama3.1-8B)             |          | 50.72  |       | 32.56  |       | 16.18  |       |        |
| Our Baselines                       |          |        |       |        |       |        |       |        |
| Qwen3-8B                            | 68.00    |        |       |        |       |        |       | 10.93  |
| Compression-Tokens (Causal)         |          | 59.62  | 58.28 | 46.58  | 49.32 | 33.41  | 29.37 |        |
| Compression-Tokens (Bidirectional)  |          | 61.44  | 62.13 | 51.48  | 55.68 | 34.17  | 33.61 |        |
| Mean-Pooling                        |          | 65.89  | 64.74 | 58.17  | 58.38 | 34.81  | 32.21 |        |
| Qwen3-4B                            | 67.12    |        |       |        |       |        |       | 10.40  |
| Compression-Tokens (Causal)         |          | 57.18  | 55.08 | 46.69  | 43.77 | 30.39  | 27.84 |        |
| Compression-Tokens (Bidirectional)  |          | 58.37  | 60.74 | 48.84  | 52.41 | 29.77  | 30.22 |        |
| Mean-Pooling                        |          | 65.22  | 63.38 | 56.14  | 55.42 | 32.66  | 28.99 |        |
| Qwen3-1.7B                          | 64.42    |        |       |        |       |        |       | 7.57   |
| Compression-Tokens (Causal)         |          | 41.97  | 49.85 | 40.15  | 39.49 | 25.42  | 23.64 |        |
| Compression-Tokens (Bidirectional)  |          | 55.68  | 56.49 | 44.87  | 47.28 | 25.16  | 26.15 |        |
| Mean-Pooling                        |          | 60.34  | 58.56 | 48.95  | 48.78 | 26.91  | 23.60 |        |
| Qwen3-0.6B                          | 61.13    |        |       |        |       |        |       | 7.76   |
| Compression-Tokens (Causal)         |          | 48.09  | 46.30 | 34.68  | 36.05 | 20.10  | 20.69 |        |
| Compression-Tokens (Bidirectional)  |          | 48.04  | 50.67 | 38.85  | 40.35 | 21.29  | 21.07 |        |
| Mean-Pooling                        |          | 56.48  | 53.12 | 42.47  | 42.21 | 21.29  | 19.06 |        |
| Gemma2-2B                           | 66.47    |        |       |        |       |        |       | 10.34  |
| Compression-Tokens (Causal)         |          | 56.17  | 55.78 | 47.16  | 46.68 | 31.17  | 29.72 |        |
| Compression-Tokens (Bidirectional)  |          | 59.20  | 59.51 | 48.59  | 51.43 | 31.50  | 31.91 |        |
| Mean-Pooling                        |          | 64.29  | 63.44 | 56.94  | 55.71 | 33.42  | 31.79 |        |
| Llama3.2-1B                         | 61.67    |        |       |        |       |        |       | 9.03   |
| Compression-Tokens (Causal)         |          | 48.57  | 45.60 | 38.44  | 37.12 | 23.03  | 23.14 |        |
| Compression-Tokens (Bidirectional)  |          | 52.24  | 49.98 | 40.65  | 42.08 | 23.59  | 23.72 |        |
| Mean-Pooling                        |          | 57.97  | 55.15 | 38.58  | 46.23 | 19.53  | 23.17 |        |

Table 13: NarrativeQA  $F_1$ .

<span id="page-24-0"></span>

|                                     | Original 4x |        | x 16: |        | x     | 128x   |       | No Ctx |
|-------------------------------------|-------------|--------|-------|--------|-------|--------|-------|--------|
|                                     |             | Single | Multi | Single | Multi | Single | Multi |        |
| Baseline Systems                    |             |        |       |        |       |        |       |        |
| LLMLingua2 (Qwen3-8B)               |             |        | 65.65 |        | 46.46 |        | 52.55 |        |
| ICAE (Mistral-7B)                   |             | 70.63  |       |        |       |        |       |        |
| PCC Lite (GPT2-Large & Llama3.1-8B) |             | 86.43  |       | 78.03  |       | 72.50  |       |        |
| PCC Large (Llama3.1-8B)             |             | 86.64  |       | 77.13  |       | 74.08  |       |        |
| Our Baselines                       |             |        |       |        |       |        |       |        |
| Qwen3-8B                            | 89.65       |        |       |        |       |        |       | 53.79  |
| Compression-Tokens (Causal)         |             | 89.67  | 89.15 | 88.92  | 85.50 | 79.36  | 77.55 |        |
| Compression-Tokens (Bidirectional)  |             | 90.44  | 89.41 | 87.07  | 87.95 | 75.90  | 78.17 |        |
| Mean-Pooling                        |             | 87.94  | 86.52 | 84.38  | 86.32 | 79.74  | 75.89 |        |
| Qwen3-4B                            | 90.46       |        |       |        |       |        |       | 43.49  |
| Compression-Tokens (Causal)         |             | 88.49  | 83.28 | 82.95  | 80.42 | 72.27  | 70.84 |        |
| Compression-Tokens (Bidirectional)  |             | 91.59  | 90.83 | 86.10  | 87.53 | 67.43  | 74.52 |        |
| Mean-Pooling                        |             | 85.50  | 88.72 | 83.68  | 85.32 | 71.04  | 67.50 |        |
| Owen3-1.7B                          | 89.20       |        |       |        |       |        |       | 25.08  |
| Compression-Tokens (Causal)         |             | 74.89  | 83.83 | 80.55  | 73.91 | 61.72  | 64.02 |        |
| Compression-Tokens (Bidirectional)  |             | 85.62  | 86.41 | 76.15  | 80.34 | 61.46  | 61.67 |        |
| Mean-Pooling                        |             | 85.83  | 82.75 | 80.61  | 75.94 | 63.03  | 53.77 |        |
| Owen3-0.6B                          | 81.55       |        |       |        |       |        |       | 9.87   |
| Compression-Tokens (Causal)         |             | 78.27  | 73.57 | 62.14  | 64.79 | 48.97  | 49.69 |        |
| Compression-Tokens (Bidirectional)  |             | 81.58  | 78.38 | 69.22  | 74.16 | 50.24  | 54.05 |        |
| Mean-Pooling                        |             | 81.45  | 77.88 | 69.08  | 70.41 | 52.45  | 43.08 |        |
| Gemma2-2B                           | 90.69       |        |       |        |       |        |       | 54.06  |
| Compression-Tokens (Causal)         |             | 89.75  | 85.06 | 82.73  | 79.19 | 77.64  | 73.71 |        |
| Compression-Tokens (Bidirectional)  |             | 88.52  | 87.14 | 85.32  | 84.63 | 78.15  | 73.59 |        |
| Mean-Pooling                        |             | 89.09  | 86.99 | 84.95  | 85.27 | 75.30  | 74.29 |        |
| Llama3.2-1B                         | 82.13       |        |       |        |       |        |       | 31.14  |
| Compression-Tokens (Causal)         |             | 84.40  | 79.82 | 75.47  | 74.28 | 65.03  | 67.58 |        |
| Compression-Tokens (Bidirectional)  |             | 83.72  | 84.18 | 78.94  | 73.03 | 64.83  | 68.57 |        |
| Mean-Pooling                        |             | 84.87  | 83.62 | 73.95  | 76.02 | 62.02  | 60.28 |        |

Table 14: TriviaQA Verified  $F_1$ .

<span id="page-25-0"></span>

|                                     | Original | al 4x  |       | 16x    |       | 128x   |       | No Ctx |
|-------------------------------------|----------|--------|-------|--------|-------|--------|-------|--------|
|                                     |          | Single | Multi | Single | Multi | Single | Multi |        |
| Baseline Systems                    |          |        |       |        |       |        |       |        |
| LLMLingua2 (Qwen3-8B)               |          |        | 32.79 |        | 19.56 |        | 18.76 |        |
| ICAE (Mistral-7B)                   |          | 27.34  |       |        |       |        |       |        |
| PCC Lite (GPT2-Large & Llama3.1-8B) |          | 42.51  |       | 35.36  |       | 26.44  |       |        |
| PCC Large (Llama3.1-8B)             |          | 44.09  |       | 33.39  |       | 27.52  |       |        |
| Our Baselines                       |          |        |       |        |       |        |       |        |
| Qwen3-8B                            | 60.44    |        |       |        |       |        |       | 19.15  |
| Compression-Tokens (Causal)         |          | 47.26  | 45.97 | 36.35  | 39.16 | 32.42  | 31.15 |        |
| Compression-Tokens (Bidirectional)  |          | 51.46  | 51.51 | 40.05  | 42.51 | 32.54  | 32.69 |        |
| Mean-Pooling                        |          | 53.71  | 53.32 | 42.55  | 45.07 | 32.52  | 31.36 |        |
| Qwen3-4B                            | 57.04    |        |       |        |       |        |       | 17.38  |
| Compression-Tokens (Causal)         |          | 45.07  | 43.44 | 34.61  | 34.34 | 28.95  | 26.25 |        |
| Compression-Tokens (Bidirectional)  |          | 45.35  | 47.93 | 36.32  | 38.23 | 28.17  | 27.88 |        |
| Mean-Pooling                        |          | 51.47  | 48.49 | 40.36  | 39.09 | 28.84  | 27.31 |        |
| Qwen3-1.7B                          | 46.62    |        |       |        |       |        |       | 14.86  |
| Compression-Tokens (Causal)         |          | 30.09  | 34.01 | 29.29  | 29.47 | 22.27  | 22.82 |        |
| Compression-Tokens (Bidirectional)  |          | 37.92  | 37.00 | 29.72  | 31.44 | 22.43  | 21.05 |        |
| Mean-Pooling                        |          | 42.14  | 39.78 | 32.33  | 32.46 | 22.01  | 22.18 |        |
| Qwen3-0.6B                          | 39.04    |        |       |        |       |        |       | 10.97  |
| Compression-Tokens (Causal)         |          | 29.69  | 28.58 | 23.55  | 23.95 | 18.80  | 19.46 |        |
| Compression-Tokens (Bidirectional)  |          | 29.16  | 33.06 | 25.47  | 26.99 | 19.60  | 18.80 |        |
| Mean-Pooling                        |          | 33.84  | 32.70 | 26.11  | 26.21 | 19.52  | 18.08 |        |
| Gemma2-2B                           | 51.45    |        |       |        |       |        |       | 16.76  |
| Compression-Tokens (Causal)         |          | 39.83  | 40.80 | 33.93  | 35.23 | 28.83  | 29.06 |        |
| Compression-Tokens (Bidirectional)  |          | 40.76  | 41.96 | 34.73  | 35.52 | 29.47  | 27.27 |        |
| Mean-Pooling                        |          | 45.61  | 44.88 | 37.14  | 36.88 | 28.70  | 28.94 |        |
| Llama3.2-1B                         | 39.75    |        |       |        |       |        |       | 14.30  |
| Compression-Tokens (Causal)         |          | 30.58  | 29.42 | 26.27  | 26.66 | 21.04  | 21.63 |        |
| Compression-Tokens (Bidirectional)  |          | 31.71  | 30.30 | 25.77  | 28.88 | 23.74  | 20.71 |        |
| Mean-Pooling                        |          | 34.84  | 33.60 | 27.18  | 27.29 | 21.21  | 20.46 |        |

Table 15: AdversarialQA  $F_1$ .

<span id="page-26-0"></span>

|                                     | Original | 1 4x   |       | 16x    |       | 128x   |       | No Ctx |
|-------------------------------------|----------|--------|-------|--------|-------|--------|-------|--------|
|                                     |          | Single | Multi | Single | Multi | Single | Multi |        |
| Baseline Systems                    |          |        |       |        |       |        |       |        |
| LLMLingua2 (Qwen3-8B)               |          |        | 27.42 |        | 15.32 |        | 7.56  |        |
| ICAE (Mistral-7B)                   |          | 28.42  |       |        |       |        |       |        |
| PCC Lite (GPT2-Large & Llama3.1-8B) |          | 46.31  |       | 33.25  |       | 18.14  |       |        |
| PCC Large (Llama3.1-8B)             |          | 46.77  |       | 31.17  |       | 17.95  |       |        |
| Our Baselines                       |          |        |       |        |       |        |       |        |
| Qwen3-8B                            | 56.77    |        |       |        |       |        |       | 7.49   |
| Compression-Tokens (Causal)         |          | 49.69  | 48.77 | 40.37  | 41.71 | 30.94  | 28.40 |        |
| Compression-Tokens (Bidirectional)  |          | 51.65  | 51.68 | 44.91  | 45.85 | 31.40  | 31.07 |        |
| Mean-Pooling                        |          | 55.36  | 53.87 | 48.95  | 48.63 | 31.79  | 29.12 |        |
| Qwen3-4B                            | 56.14    |        |       |        |       |        |       | 6.59   |
| Compression-Tokens (Causal)         |          | 47.85  | 46.50 | 40.73  | 40.49 | 28.86  | 27.16 |        |
| Compression-Tokens (Bidirectional)  |          | 49.63  | 50.74 | 42.98  | 44.60 | 27.72  | 28.07 |        |
| Mean-Pooling                        |          | 54.74  | 53.24 | 46.93  | 47.01 | 29.95  | 26.25 |        |
| Qwen3-1.7B                          | 54.75    |        |       |        |       |        |       | 5.09   |
| Compression-Tokens (Causal)         |          | 37.53  | 43.23 | 36.08  | 35.38 | 24.47  | 22.71 |        |
| Compression-Tokens (Bidirectional)  |          | 46.33  | 47.36 | 39.28  | 39.72 | 24.81  | 23.81 |        |
| Mean-Pooling                        |          | 52.04  | 50.67 | 42.90  | 42.55 | 25.16  | 22.02 |        |
| Qwen3-0.6B                          | 51.54    |        |       |        |       |        |       | 4.82   |
| Compression-Tokens (Causal)         |          | 42.04  | 39.38 | 32.91  | 31.36 | 20.07  | 19.73 |        |
| Compression-Tokens (Bidirectional)  |          | 43.34  | 43.74 | 34.15  | 35.77 | 21.12  | 19.16 |        |
| Mean-Pooling                        |          | 48.26  | 46.29 | 38.13  | 37.81 | 21.24  | 17.03 |        |
| Gemma2-2B                           | 56.00    |        |       |        |       |        |       | 7.10   |
| Compression-Tokens (Causal)         |          | 48.10  | 46.86 | 41.66  | 40.33 | 29.30  | 27.74 |        |
| Compression-Tokens (Bidirectional)  |          | 49.36  | 49.83 | 42.30  | 43.72 | 29.85  | 28.81 |        |
| Mean-Pooling                        |          | 54.39  | 53.39 | 47.82  | 47.83 | 31.16  | 28.91 |        |
| Llama3.2-1B                         | 52.26    |        |       |        |       |        |       | 5.94   |
| Compression-Tokens (Causal)         |          | 41.97  | 40.07 | 35.08  | 35.24 | 22.50  | 23.14 |        |
| Compression-Tokens (Bidirectional)  |          | 44.58  | 44.02 | 36.36  | 38.77 | 23.84  | 23.88 |        |
| Mean-Pooling                        |          | 49.90  | 46.92 | 33.84  | 39.59 | 17.67  | 21.17 |        |

Table 16: ParaphraseRC *F*<sub>1</sub>.