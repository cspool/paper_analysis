# D Additional evaluation

In this section, we provide additional experimental results to demonstrate that

- MagicPIG can support longer context lengths and a wide range of LLMs (Appendix [D.1\)](#page-19-0).
- MagicPIG can scale up with 70B level LLM (Appendix [D.2\)](#page-19-1).
- MagicPIG can perform well in reasoning benchmarks (Appendix [D.3\)](#page-19-2).
- <span id="page-19-0"></span>• MagicPIG improves decoding throughput with various hyper-parameters (K, L). (Appendix [D.4\)](#page-19-3).

### D.1 Longer Contexts

Following the setups of Table [3,](#page-10-1) we evaluate two additional models, MegaBeam-Mistral-7B-512K[4](#page-19-4) and Llama3- 8B-Prolong-512K [\(Gao et al.,](#page-13-18) [2024\)](#page-13-18) with context lengths extended to 256K. The results are shown in Table [4.](#page-19-5)

<span id="page-19-5"></span>Table 4 Synthesized tasks on RULER [\(Hsieh et al.,](#page-13-11) [2024\)](#page-13-11). MagicPIG preserves high accuracy with extended context lengths and different models. Config and cost are defined as in Table [1.](#page-9-0)

| Methods                  | Config    | 16K  | 32K  | 64K  | 96K  | 128K | 256K | Avg. | Cost1 | Cost2 | Costtotal. |
|--------------------------|-----------|------|------|------|------|------|------|------|-------|-------|------------|
| MegaBeam-Mistral-7B-512K | Full      | 91.7 | 88.1 | 83.5 | 83.7 | 83.5 | 82.5 | 85.5 | 0.00  | 1.00  | 1.00       |
| MagicPIG                 | (10,150)  | 89.8 | 86.5 | 81.7 | 80.7 | 81.6 | 79.0 | 83.2 | 0.00  | 0.02  | 0.02       |
| MagicPIG                 | (9,120)   | 90.7 | 88.5 | 82.9 | 82.4 | 82.3 | 80.1 | 84.5 | 0.00  | 0.04  | 0.04       |
| MagicPIG                 | (8,75)    | 90.6 | 86.4 | 82.8 | 81.6 | 82.3 | 80.8 | 84.1 | 0.00  | 0.05  | 0.05       |
| Quest                    | (16,0.04) | 83.3 | 83.2 | 79.3 | 78.6 | 78.5 | 78.5 | 80.2 | 0.06  | 0.04  | 0.10       |
| Llama3-8B-Prolong-512K   | Full      | 93.5 | 90.8 | 85.1 | 83.5 | 81.7 | 78.4 | 85.5 | 0.00  | 1.00  | 1.00       |
| MagicPIG                 | (10,150)  | 88.0 | 86.4 | 81.3 | 78.8 | 77.3 | 71.1 | 80.5 | 0.00  | 0.02  | 0.02       |
| MagicPIG                 | (10,170)  | 89.0 | 88.7 | 82.8 | 80.0 | 77.7 | 73.7 | 82.0 | 0.00  | 0.025 | 0.025      |
| MagicPIG                 | (9,120)   | 91.4 | 88.2 | 82.4 | 80.4 | 79.2 | 75.2 | 82.8 | 0.00  | 0.04  | 0.04       |
| MagicPIG                 | (8,75)    | 91.4 | 88.6 | 83.1 | 80.5 | 79.1 | 73.9 | 82.8 | 0.00  | 0.05  | 0.05       |
| Quest                    | (16,0.04) | 84.9 | 83.7 | 78.7 | 78.6 | 76.3 | 72.3 | 79.2 | 0.06  | 0.04  | 0.10       |

### <span id="page-19-1"></span>D.2 Scaling up to larger models

We evaluate MagicPIG for meta-llama/Llama-3.1-70B-Instruct [\(Dubey et al.,](#page-13-0) [2024\)](#page-13-0) to demonstrate that our approach can work well with larger LLMs in Table [5.](#page-19-6)

<span id="page-19-6"></span>Table 5 Synthesized tasks from RULER [\(Hsieh et al.,](#page-13-11) [2024\)](#page-13-11). MagicPIG preserves high accuracy with low computation for 70B level models. 4 layers {0,16,32,48} are preserved. Config and cost are defined as in Table [1.](#page-9-0)

| Methods                | Config   | 16K  | 32K  | 64K  | 96K  | Avg. | Cost1 | Cost2 | Costtotal. |
|------------------------|----------|------|------|------|------|------|-------|-------|------------|
| Llama-3.1-70B-Instruct | Full     | 96.4 | 94.6 | 89.2 | 80.8 | 90.3 | 0.00  | 1.00  | 1.00       |
| MagicPIG               | (10,150) | 94.7 | 93.5 | 87.5 | 79.3 | 88.8 | 0.00  | 0.02  | 0.02       |
| MagicPIG               | (9,110)  | 95.7 | 93.5 | 88.4 | 79.4 | 89.3 | 0.00  | 0.034 | 0.034      |
| MagicPIG               | (9,120)  | 95.5 | 94.1 | 88.8 | 80.6 | 89.8 | 0.00  | 0.04  | 0.04       |

### <span id="page-19-2"></span>D.3 Reasoning

In mathematical reasoning tasks infini igsm [\(Zhou,](#page-16-1) [2024a,](#page-16-1)[b\)](#page-16-2), MagicPIG consistently outperforms Quest [\(Tang](#page-15-2) [et al.,](#page-15-2) [2024\)](#page-15-2) across all complexity (in terms of operators). We also find TopK attention suffers from significant performance degradation while Oracle Sampling can maintain high accuracy.

<span id="page-19-4"></span><span id="page-19-3"></span><sup>4</sup><https://huggingface.co/aws-prototyping/MegaBeam-Mistral-7B-512k>

Table 6 Tasks from infini igsm [\(Zhou,](#page-16-1) [2024a](#page-16-1)[,b\)](#page-16-2). MagicPIG preserves high accuracy for reasoning tasks. Config and cost for MagicPIG and Quest are defined as in Table [1.](#page-9-0) Config denotes the ratio of selected tokens for TopK and sampled tokens for oracle sampling. For oracle sampling, massive duplication exists in sampled tokens, so Cost<sup>2</sup> is significantly lower than the ratio of sampled tokens Theorem [3.3.](#page-5-0)

| Task                   | Methods               | Config    | 2-Ops | 4-Ops | 5-Ops | Cost1 | Cost2 | Costtotal. |
|------------------------|-----------------------|-----------|-------|-------|-------|-------|-------|------------|
|                        | Llama-3.1-8B-Instruct | Full      | 87.4  | 71.4  | 26.8  | 0.00  | 1.00  | 1.00       |
|                        | MagicPIG              | (10,300)  | 83.1  | 67.2  | 20.7  | 0.00  | 0.06  | 0.06       |
|                        | MagicPIG              | (10,220)  | 79.8  | 58.9  | 17.9  | 0.00  | 0.04  | 0.04       |
|                        | MagicPIG              | (10,150)  | 68.3  | 43.5  | 11.7  | 0.00  | 0.02  | 0.02       |
|                        | TopK                  | 0.06      | 78.6  | 62.9  | 20.8  | 0.50  | 0.06  | 0.56       |
| 4K close (Zhou, 2024a) | TopK                  | 0.04      | 76.2  | 59.0  | 19.2  | 0.50  | 0.04  | 0.54       |
|                        | TopK                  | 0.02      | 71.5  | 44.0  | 11.3  | 0.50  | 0.02  | 0.52       |
|                        | Oracle Sampling       | 0.3       | 88.1  | 72.4  | 27.6  | 0.50  | 0.02  | 0.52       |
|                        | Oracle Sampling       | 0.1       | 88.5  | 69.2  | 26.2  | 0.50  | 0.01  | 0.51       |
|                        | Oracle Sampling       | 0.02      | 83.1  | 57.9  | 11.9  | 0.50  | 0.005 | 0.505      |
|                        | Quest                 | (16,0.06) | 55.8  | 23.2  | 5.2   | 0.06  | 0.06  | 0.12       |
|                        | Llama-3.1-8B-Instruct | Full      | 80.2  | 68.8  | 26.0  | 0.00  | 1.00  | 1.00       |
|                        | MagicPIG              | (10,300)  | 78.6  | 61.5  | 25.2  | 0.00  | 0.06  | 0.06       |
|                        | MagicPIG              | (10,220)  | 72.2  | 60.7  | 20.4  | 0.00  | 0.04  | 0.04       |
|                        | MagicPIG              | (10,150)  | 67.1  | 44.0  | 11.9  | 0.00  | 0.02  | 0.02       |
|                        | TopK                  | 0.06      | 70.2  | 61.1  | 22.3  | 0.50  | 0.06  | 0.56       |
| 8K close (Zhou, 2024b) | TopK                  | 0.04      | 66.9  | 55.2  | 20.6  | 0.50  | 0.04  | 0.54       |
|                        | TopK                  | 0.02      | 64.7  | 47.2  | 15.9  | 0.50  | 0.02  | 0.52       |
|                        | Oracle Sampling       | 0.3       | 80.0  | 67.3  | 26.2  | 0.50  | 0.02  | 0.52       |
|                        | Oracle Sampling       | 0.1       | 76.6  | 64.1  | 25.4  | 0.50  | 0.01  | 0.51       |
|                        | Oracle Sampling       | 0.02      | 79.0  | 60.3  | 20.4  | 0.50  | 0.005 | 0.505      |
|                        | Quest                 | (16,0.06) | 54.8  | 30.0  | 11.1  | 0.06  | 0.06  | 0.12       |

<span id="page-20-1"></span>Table 7 System performance for MagicPIG using Llama-3.1-8B-Instruct with a 96K context length under varying hyper-parameter configurations. We report the decoding latency (time between tokens, TBT) when the batch size is 1, the maximum throughput, and the throughput with a latency constraint of 200ms (Throughput200ms in the table). Config and cost are defined as in Table [1.](#page-9-0) The number with <sup>∗</sup> means hit the memory limit of CPU.

| Config   | TBT (ms) | Max Throughput (tokens/sec) | Throughput200ms<br>(tokens/sec) | Costtotal. |
|----------|----------|-----------------------------|---------------------------------|------------|
| (11,300) | 17.38    | 41.68∗                      | 40.84                           | 0.02       |
| (10,220) | 14.07    | 32.29∗                      | 26.66                           | 0.04       |
| (10,170) | 16.79    | 46.52∗                      | 39.90                           | 0.025      |
| (10,150) | 18.31    | 53.78                       | 48.89                           | 0.02       |
| (9,120)  | 13.93    | 32.50                       | 26.60                           | 0.04       |
| (8,75)   | 12.47    | 27.43                       | 21.17                           | 0.05       |

## D.4 System performance

In this section, we evaluate the system performance (latency, throughput) of MagicPIG under different hyper-parameter configurations. We use Llama-3.1-8B-Instruct [\(Dubey et al.,](#page-13-0) [2024\)](#page-13-0) with 96K contexts as an example.

