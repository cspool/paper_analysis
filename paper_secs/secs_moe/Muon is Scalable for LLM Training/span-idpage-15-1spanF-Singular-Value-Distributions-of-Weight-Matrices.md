# <span id="page-15-1"></span>F Singular Value Distributions of Weight Matrices

We visualize the singular value distributions of weight matrices by plotting a line graph of its singular values in descending order for each matrix, normalized by the largest one. As shown in Figures [9](#page-17-1) and [10,](#page-18-0) we find that, for most of the weight matrices, the singular value distributions of them optimized by Muon are more flattened than that of AdamW, which further confirms the hypothesis that Muon can provide a more diverse spectrum of updates.

<span id="page-16-0"></span>![](_page_16_Figure_2.jpeg)

Figure 7: Training dynamics comparison between Moonlight and Moonlight-A

Table 10: Comparison of different models on various benchmarks.

<span id="page-16-1"></span>

|         | Benchmark (Metric)                                                               | Moonlight                       | Larger Training Compute Model  |                                 |                                  |  |
|---------|----------------------------------------------------------------------------------|---------------------------------|--------------------------------|---------------------------------|----------------------------------|--|
|         | Activated Param <sup>†</sup> Total Params <sup>†</sup> Training Tokens Optimizer | 2.24B<br>15.29B<br>5.7T<br>Muon | 7.38B<br>7.38B<br>15T<br>AdamW | 8.32B<br>8.32B<br>8T<br>Unknown | 6.83B<br>6.83B<br>18T<br>Unknown |  |
| English | MMLU<br>MMLU-pro<br>BBH<br>TriviaQA <sup>‡</sup>                                 | 70.0<br>42.4<br>65.2<br>66.3    | 66.7<br>37.1<br>57.7<br>70.3   | 71.3<br>44.7<br>68.2            | 74.2<br>45.0<br>70.4<br>60.0     |  |
| Code    | HumanEval<br>MBPP                                                                | 48.1<br>63.8                    | 37.2<br>47.6                   | 37.8<br>62.2                    | 57.9<br>74.9                     |  |
| Math    | GSM8K<br>MATH                                                                    | 77.4<br>45.3                    | 57.2<br>20.3                   | 70.7<br>37.7                    | 85.4<br>49.8                     |  |

<sup>&</sup>lt;sup>†</sup> The reported parameter counts exclude the embedding parameters. <sup>‡</sup> We test all listed models with the full set of TriviaQA.

<span id="page-17-0"></span>![](_page_17_Figure_2.jpeg)

Figure 8: The GSM8k performance of our Moonlight model optimized with Muon and other comparable models.

<span id="page-17-1"></span>![](_page_17_Figure_4.jpeg)

Figure 9: Distribution of singular values for each weight matrix in the attention layers. We use WC to denote the weight matrices at each layer that compress the hidden states to the shared latent spaces for keys and values, WV to denote the weight matrices up-projecting the values from the latent space, WO to denote the output projection matrices, and WKR, WKC, WQR and WQC to denote the projection matrices for the part of keys and queries with and without RoPE respectively. We set the spines of each line graph red if the corresponding weight matrix optimized by Muon has a lower singular entropy than AdamW.

<span id="page-18-0"></span>![](_page_18_Figure_2.jpeg)

Figure 10: Distribution of singular values for each weight matrix in the feed-forward network (FFN) layers. We use WI, WV and WO to denote the weight matrices involved in the FFN layer with SwiGLU activation function, where WI represents the input projection to the Swish<sup>1</sup> function, WV represents the extra input projection interacting with Swish<sup>1</sup> activations, and WO represents the output projection. We use E0, E2, E3 to denote three arbitrarily selected expert models and SE to denote the weights in the shared expert model. We use RW to denote the weights in the router. We set the spines of each line graph red if the corresponding weight matrix optimized by Muon has a lower singular entropy than AdamW.