# <span id="page-15-0"></span>G FAIRNESS

To enhance memory capacity, MoM applies sparse activation to the key and value projections. Although these projections constitute a small portion of the overall model parameters, this inevitably increases the parameter count. Due to differing linear model structures, aligning both parameter count and memory capacity exactly is challenging. Thus, to ensure fairness, we conduct comparisons from two perspectives: equal activated parameter count and equal memory capacity.

### G.1 EQUAL ACTIVATED PARAMETER COUNT

To ensure a fair comparison of parameters, we reduced the MLP hidden ratio to 2 and retrained the MoM model using the same training configurations as in Section 4.1. Both MoM and Gated Deltanet were set with 400M activated parameters. Although the smaller hidden ratio might impact the model's commonsense knowledge, we tested on commonsense reasoning tasks and recall-intensive benchmarks. MoM consistently outperformed Gated Deltanet in both tests, further validating the effectiveness of the MoM approach. The results are presented in Table [9](#page-16-2)

<span id="page-16-2"></span>

| Model          | Params | ARC-e<br>acc↑ | ARC-c<br>accn↑ | Hella.<br>accn↑ | Lamb.<br>acc↑ | PIQA<br>acc↑ | Wino.<br>acc↑ | Avg.  |
|----------------|--------|---------------|----------------|-----------------|---------------|--------------|---------------|-------|
| Gated DeltaNet | 400M   | 46.04         | 23.55          | 35.18           | 27.01         | 66.05        | 50.83         | 41.44 |
| MoM            | 400M   | 47.10         | 23.72          | 35.43           | 26.88         | 64.64        | 51.22         | 41.50 |
| Model          | Params | FDA           | SWDE           | SQUAD           | NQ            | TriviaQA     | Drop          | Avg.  |
| Gated DeltaNet | 400M   | 20.53         | 23.24          | 28.55           | 14.98         | 44.91        | 16.48         | 24.78 |
| MoM            | 400M   | 24.16         | 25.59          | 29.46           | 15.36         | 46.15        | 18.35         | 26.51 |

Table 9: Comparison with the Same Activated Parameters. MoM and Gated DeltaNet with 400M activated parameters are tested.

### G.2 EQUAL MEMORY CAPACITY

To ensure a fair comparison of memory capacity, we also compared the single extended memory model with the MoM model. Notably, the single extended memory model has more parameters than the activated parameters in the MoM due to the extension of the v dimension. MoM expands memory more elegantly and significantly outperforms in both recall and commonsense tasks. This comparison result is presented in Table [4.](#page-6-2)

