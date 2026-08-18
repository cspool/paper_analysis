# G. Comparison with Fixed-precision Designs

To isolate the effect of composability, we compare fixed-precision UNICORE GEMM units with baselines built using (i) Synopsys DesignWare IP (DW IP) and (ii) a modified Tender design without composability, all under a weight-stationary dataflow. UNICORE achieves the highest compute density across all fixed-precision settings: 6.46 at W4A4 (2.78× and  $1.32\times$  over DesignWare and Tender), 3.42 at W8A8 ( $1.94\times$  and  $1.19\times$ ), and 2.05 at W16A16 ( $2.05\times$  and  $1.90\times$ ), as shown in Figure 21a. These results indicate that the

<span id="page-12-0"></span>TABLE III: WikiText-2 perplexity (PPL) of FPMA with different kinds of compensation. FG: Fine-Grained Compensation; CG: Coarse-Grained Compensation. UNICORE adapts both FG and CG to maintain accuracy, without DynFP quantization.

| Method  | Bits | OPT-6.7B | Llama-2-7B | Llama-3-8B | Qwen3-8B | Qwen3-14B |
|---------|------|----------|------------|------------|----------|-----------|
| FP16    | 16   | 10.86    | 5.47       | 6.14       | 9.72     | 8.64      |
| UniCore | 16   | 10.88    | 5.48       | 6.17       | 9.69     | 8.64      |
| FP8     | 8    | 10.98    | 5.50       | 6.20       | 9.76     | 8.71      |
| FPMA+CG | 8    | 11.02    | 5.52       | 6.23       | 9.84     | 8.77      |
| UniCore | 8    | 10.98    | 5.50       | 6.20       | 9.77     | 8.72      |
| FP4     | 4    | 11.15    | 5.81       | 7.05       | 10.37    | 9.14      |
| FPMA    | 4    | 1.1E+4   | 3.4E+4     | 3.6E+5     | 4.9E+6   | 1.5E+6    |
| UniCore | 4    | 11.15    | 5.81       | 7.05       | 10.37    | 9.14      |

<span id="page-12-1"></span>![](_page_12_Figure_2.jpeg)

Fig. 21: Normalized compute density and power of standalone fixed-precision GEMM variants across different precisions.

UNICORE's advantage stems from the lightweight S-FPMA datapath rather than composability alone.

We further include fixed-width mixed-precision baselines BitMoD [7] and AxCore [50]. Figure 21a shows that using the fixed-precision UNICORE W16A16 GEMM unit (2.05) as the A16-compatible reference, UNICORE achieves 1.60× BitMoD (1.28) and 0.50× AxCore (4.07) under W4A16. The gap stems from AxCore specializing in fixed weight-only mixed precision, whereas UNICORE supports composable precision with joint W+A quantization. Under the same W4 setting, UNICORE reaches 1.59× AxCore with comparable accuracy (Table I), highlighting the advantage of lowering activation precision. In addition, as shown in Figure 21b, the fixed-width UNICORE GEMM also maintains competitive or lower normalized power across precisions. Across fixed-precision settings, UNICORE reduces normalized power by 35.6%–45.0% over DesignWare and 11.1%–44.4% over Tender.

### H. Ablation Study on Quantization Group Sizes

We evaluate the perplexity results of UNICORE across multiple quantization group sizes and compare it with INT, M-ANT, and BitMoD. Table IV reports results with group-wise quantization on both weights and activations. As expected, smaller groups improve all methods by providing finer-grained scaling, while UNICORE maintains the best PPL across all group sizes. These results show that UNICORE's accuracy advantage is robust to quantization granularity.

