# **B.2** For Recovery Stage

We follow [19] in our recovery stage. We set the rank d to 8 in our experiment. The learning rate is set to 1e-4 with 100 warming steps. The batch size of training is selected from  $\{64, 128\}$  and the AdamW optimizer is employed in our experiment. The best training epoch we found is 2 epochs, as training with more epochs even has a negative impact on the model performance. We run our experiment on a single GPU with 24GB memory, using approximately 2.5 hours if RTX4090 is utilized. All the linear module is taken into account for efficient tuning. An ablation experiment for this is shown in Table 11.

<span id="page-15-0"></span>Table 11: Ablation: Tuning different modules in the recovery stage

| Module | WikiText↓ | PTB↓  |
|--------|-----------|-------|
| ALL    | 17.36     | 29.99 |
| - MLP  | 17.64     | 30.63 |
| - MHA  | 17.62     | 30.23 |

## C More Analysis

### C.1 More Data for Recovery

Despite our primary experiments being conducted using 50k samples, we remain convinced that the inclusion of additional data could substantially enhance the recovery process, albeit at a considerably higher computational cost. Consequently, we conduct an experiment aimed at model recovery with more data, employing a dataset comprising 2.59 million samples [57]. The results are detailed in Table 12. From the results, it is evident that the performance of the compressed model closely approximates that of the base model, exhibiting only a marginal performance decrease of 0.89%.

Table 12: Model Recovery: 50k samples vs. 2.59M samples

<span id="page-15-1"></span>

| Model      | #Samples   | BoolQ | PIQA  | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA  | Average |
|------------|------------|-------|-------|-----------|------------|-------|-------|-------|---------|
| LLaMA-7B   | -          | 73.18 | 78.35 | 72.99     | 67.01      | 67.45 | 41.38 | 42.40 | 63.25   |
| LLaMA-5.4B | 50k [47]   | 64.62 | 77.20 | 68.80     | 63.14      | 64.31 | 36.77 | 39.80 | 59.23   |
| LLaMA-5.4B | 2.59M [57] | 76.57 | 77.37 | 66.60     | 65.82      | 70.62 | 40.70 | 38.80 | 62.36   |

## C.2 Pruning vs. Quantization

Here, we conduct a comparative analysis of different compression techniques and illustrate that these techniques can be effectively combined with little performance degradation. We have chosen LLM.int8() [8] as a representative example of quantization methods. Our results show that LLM.int8() outperforms LLM-Pruner while LLM-Pruner enhances latency, reduces parameter size. When these two techniques are applied in tandem, they collectively reduce memory consumption and expedite inference, offering a balanced approach that combines the benefits of both methods.

Table 13: Pruning and Quantization on LLaMA-7B

| Pruning Ratio           | #Param | Memory     | Latency | BoolQ | PIQA  | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA  | Average |
|-------------------------|--------|------------|---------|-------|-------|-----------|------------|-------|-------|-------|---------|
| LLaMA-7B                | 6.74B  | 12884.5MiB | 69.32s  | 73.18 | 78.35 | 72.99     | 67.01      | 67.45 | 41.38 | 42.40 | 63.25   |
| LLM.int8()              | 6.74B  | 6777.7MiB  | 76.20s  | 73.36 | 78.18 | 73.01     | 66.93      | 67.47 | 40.87 | 41.80 | 63.09   |
| LLaMA-5.4B              | 5.47B  | 10488.4MiB | 58.55s  | 76.57 | 77.37 | 66.60     | 65.82      | 70.62 | 40.70 | 38.80 | 62.36   |
| LLaMA-5.4B + LLM.int8() | 5.47B  | 5444.37MiB | 63.10s  | 76.39 | 76.71 | 66.62     | 66.46      | 70.54 | 40.19 | 39.20 | 62.30   |

### C.3 Global Pruning vs. Local Pruning

we present a comparative analysis between global pruning and local pruning, where the pruning ratio is 20% and the base model is LLaMA-7B. Global pruning refers to ranking all groups in the model together, whereas local pruning involves only ranking groups within the same module for pruning. The outcome of global pruning leads to varying widths across different layers and modules, whereas local pruning ensures uniformity across all layers.

Based on our experimental findings, we observed a slight advantage of local pruning over global pruning. We think this is because of the varying magnitudes in different layers or modules, which makes the importance scores incomparable between groups across different layers.

Table 14: Results of global pruning and local pruning

| Method                        | WikiText2↓ | PTB↓   BoolQ  | PIQA  | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA   Average |
|-------------------------------|------------|---------------|-------|-----------|------------|-------|-------|----------------|
| Element <sup>1</sup> - local  | 19.09      | 34.21   57.06 | 75.68 | 66.80     | 59.83      | 60.94 | 36.52 | 40.00   56.69  |
| Element <sup>1</sup> - global | 20.84      | 32.86 63.15   | 73.23 | 63.31     | 66.38      | 55.85 | 35.49 | 38.00 56.49    |

### <span id="page-16-1"></span>C.4 Overfitting Phenomena in Post-Training

We present a comprehensive analysis of the overfitting issue in the recovery stage, as previously mentioned in Figure 5. Here the results cover all 9 datasets across various training steps. Based on the findings presented in Table 15, a noticeable trend emerges: the accuracy or generation quality initially shows improvement but subsequently experiences a slight decline. This pattern suggests that the recovery process is completed within a short period. And given that the training corpus is domain-constrained, more training epochs can result in overfitting to the specific dataset while potentially compromising the original capabilities of the language model.

Table 15: The PPL and Accuracy on different training steps

<span id="page-16-2"></span>

| Step | WikiText2↓   | PTB↓  | BoolQ | PIQA  | HellaSwag | WinoGrande   | ARC-e | ARC-c        | OBQA  | Average      |
|------|--------------|-------|-------|-------|-----------|--------------|-------|--------------|-------|--------------|
| 0    | 19.09        | 34.21 | 57.06 | 75.68 | 66.80     | 59.83        | 60.94 | 36.52        | 40.00 | 56.69        |
| 200  | 18.10        | 30.66 | 64.62 | 77.20 | 68.80     | 63.14        | 64.31 | 36.77        | 39.80 | 59.24        |
| 400  | 17.69        | 30.26 | 63.00 | 76.66 | 68.75     | 63.54        | 64.39 | 37.20        | 40.60 | 59.16        |
| 600  | 17.69        | 30.57 | 66.24 | 76.28 | 68.52     | 63.85        | 64.48 | 37.37        | 41.00 | <u>59.68</u> |
| 800  | <u>17.64</u> | 30.57 | 65.05 | 76.22 | 68.38     | 63.77        | 63.64 | 37.29        | 40.80 | 59.31        |
| 1000 | 17.67        | 30.60 | 66.39 | 76.17 | 68.24     | <u>64.17</u> | 63.05 | 37.37        | 41.60 | 59.57        |
| 1200 | 17.74        | 30.75 | 65.75 | 76.28 | 68.28     | 63.77        | 63.30 | 37.63        | 41.20 | 59.46        |
| 1400 | 17.88        | 30.85 | 64.34 | 76.28 | 68.31     | 63.85        | 63.47 | <u>37.80</u> | 41.20 | 59.32        |

