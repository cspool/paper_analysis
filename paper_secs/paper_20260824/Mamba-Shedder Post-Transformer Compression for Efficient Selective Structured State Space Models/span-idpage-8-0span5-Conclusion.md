# <span id="page-8-0"></span>5 Conclusion

Selective structure state space models have become an efficient alternative to Transformer-based models. In this paper, we propose Mamba-Shedder and investigate structured pruning strategies to remove elements from Mamba and hybrid models and reduce model size, accelerating inference. The results demonstrate that selective structured state space architectures have several redundancies that

<span id="page-9-3"></span>

| Model       | Method                 | Ratio<br>(Block, Width) | Additional<br>Pruned SSMs | Lambada<br>PPL (↓) | Average<br>Accuracy |
|-------------|------------------------|-------------------------|---------------------------|--------------------|---------------------|
|             | Dense<br>Mamba-Shedder | -<br>10.27%             | 0 / 54<br>18 / 54         | 4.01<br>5.18       | 67.2<br>65.9        |
| Zamba2-2.7B | Mamba-Shedder w/ tune  | 10.27%                  | 18 / 54                   | 4.58-0.60          | 67.0+1.1            |
|             | Mamba-Shedder          | 15.48%                  | 18 / 54                   | 7.43               | 61.3                |
|             | Mamba-Shedder w/ tune  | 15.48%                  | 18 / 54                   | 5.88-1.55          | 64.4+3.1            |

Table 11: Results of the compressed Mamba2-2.7B and Zamba2-2.7B models with recovery tuning.

<span id="page-9-4"></span>

| Model           | Method                | Num. of Pruned<br>Hymba Blocks | Average<br>Accuracy |
|-----------------|-----------------------|--------------------------------|---------------------|
|                 | Dense                 | 0 / 32                         | 63.8                |
| Hymba-1.5B-Base | Mamba-Shedder         | 7 / 32                         | 61.7                |
|                 | Mamba-Shedder w/ tune | 7 / 32                         | 63.7+2.0            |

Table 12: Results of the compressed Hymba-1.5B-Base model with recovery tuning. *Average Accuracy* is calculated over HellaSwag, PIQA, ARC-e, ARC-c, and WinoGrande tasks (Table [4\)](#page-7-0).

can be removed without significantly affecting the model's performance.

