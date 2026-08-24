# I Full Experimental Results on Out-of-Domain Benchmarks

Figure 11 illustrates the performance of various methods on out-of-domain benchmarks, including GPQA [17], LSAT [29, 23], and MMLU [8]. Across all benchmarks, LASER, LASER-D and LASER-DE consistently demonstrate significant improvements in both accuracy and efficiency. Notably,

<span id="page-18-2"></span>> **[图片提取文字 (无描述)]:**
> Easy Level Medium Level Hard Level 14000 17000 4000 -LASER - DLT = 1024 LASER - DLT = 1024 12000 3500 -16000 LASER - DLT = 2048 LASER - DLT = 2048 Target Length LASER - DLT = 4096 LASER - DLT = 4096 Length LASER - DELT = 1024 LASER - DE<sub>LT = 1024</sub> 10000 Target Le LASER - DELT = 2048 LASER - DELT = 2048 15000 Target --- LASER - DE<sub>LT</sub> = 4096 — LASER — DE<sub>LT</sub> = 4096 8000 Adaptive Adaptive Adaptive 1500 14000 LASER - DLT = 1024 6000 -LASER - DLT = 2048 1500 LASER - DLT = 4096 13000 LASER - DELT = 1024 4000 1000 -LASER - DELT = 2048 LASER - DELT = 4096 2000 + 12000 -100 200 300 400 500 600 700 100 200 300 400 500 600 700 100 200 300 400 500 600 700 Step Step Step
![](_page_18_Figure_0.jpeg)

Figure 10: Dynamics of adaptive target lengths during the training process of LASER-D and LASER-DE. The figure shows how the adaptive target length  $L_A$  changes over training iterations for problems of different difficulty levels (easy, medium, hard). For easy problems, the model selects short target lengths; for medium problems, it gradually decreases from higher initial values; and for hard problems, it maintains consistently high target lengths near the context window limit. This demonstrates the methods' ability to adaptively allocate computational resources based on problem complexity, unlike fixed-length approaches.

<span id="page-18-1"></span>Table 6: Full results of accuracy (%) with average token usage for each dataset and different methods. The base model is DeepSeek-R1-Distill-Qwen-1.5B. "Original" denotes the original model.  $T_k$  is the truncation method with context window k. "Group" denotes the Efficient Reasoning [2] with different  $\alpha$ . Due to the space limit, we only show three most representative results of truncation method here.

|                                           |             | Accuracy (%) |      |                   |       |             | Generation Length (tokens) |      |                   |       |  |  |
|-------------------------------------------|-------------|--------------|------|-------------------|-------|-------------|----------------------------|------|-------------------|-------|--|--|
|                                           | MATH<br>500 | AIME         | AMC  | Olympiad<br>Bench | Avg.  | MATH<br>500 | AIME                       | AMC  | Olympiad<br>Bench | Avg.  |  |  |
| Original                                  | 83.9        | 28.9         | 71.6 | 43.3              | 56.9  | 5042        | 15956                      | 8202 | 11510             | 10177 |  |  |
| $T_{10240}$                               | 82.7        | 26.9         | 73.1 | 44.1              | 56.7  | 2056        | 5458                       | 3036 | 3405              | 3489  |  |  |
| $T_{8192}$                                | 81.8        | 24.8         | 70.9 | 43.9              | 55.35 | 1795        | 4465                       | 2560 | 2841              | 2915  |  |  |
| $T_{7168}$                                | 81.8        | 23.3         | 68.6 | 43.0              | 54.18 | 1553        | 3726                       | 2251 | 2323              | 2463  |  |  |
| $T_{6144}$                                | 80.9        | 20.2         | 66.2 | 42.1              | 52.35 | 1351        | 2821                       | 1917 | 1947              | 2009  |  |  |
| $T_{4096}$                                | 77.7        | 19.2         | 62.2 | 38.5              | 49.4  | 1054        | 2481                       | 1484 | 1564              | 1646  |  |  |
| $T_{2048}$                                | 73.2        | 15.8         | 56.9 | 35.9              | 45.45 | 721         | 1029                       | 936  | 1084              | 943   |  |  |
| $Group_{\alpha=0.4}$                      | 74.6        | 25.0         | 69.2 | 43.1              | 53.0  | 1069        | 4747                       | 2162 | 2536              | 2629  |  |  |
| $Group_{\alpha=0.2}$                      | 78.1        | 28.1         | 68.0 | 44.4              | 54.7  | 1135        | 5628                       | 2635 | 2944              | 3085  |  |  |
| $Group_{\alpha=0.1}$                      | 77.0        | 29.0         | 69.5 | 44.9              | 55.1  | 1228        | 6301                       | 2808 | 3271              | 3402  |  |  |
| $Group_{\alpha=0.05}$                     | 74.4        | 30.2         | 65.5 | 43.1              | 53.3  | 1193        | 4839                       | 2457 | 2703              | 2798  |  |  |
| L1-Max-1024                               | 76.4        | 15.0         | 59.4 | 39.1              | 47.5  | 661         | 1303                       | 933  | 938               | 959   |  |  |
| L1-Max-4096                               | 79.7        | 20.0         | 65.0 | 41.0              | 51.4  | 875         | 1718                       | 1159 | 1229              | 1245  |  |  |
| $LASER_{L_T=2048}$                        | 83.6        | 29.2         | 71.6 | 44.1              | 57.1  | 1913        | 4815                       | 2493 | 2767              | 2895  |  |  |
| $LASER_{L_T=4096}$                        | 83.9        | 31.0         | 74.1 | 45.7              | 58.7  | 1914        | 5915                       | 3136 | 3579              | 3636  |  |  |
| $LASER_{L_T=8192}$                        | 85.6        | 31.5         | 75.9 | 47.7              | 60.2  | 2736        | 6589                       | 4162 | 4547              | 4509  |  |  |
| LASER-D <sub><math>L_T</math>=1024</sub>  | 83.0        | 30.6         | 72.8 | 43.7              | 57.5  | 1362        | 4991                       | 2556 | 2837              | 2862  |  |  |
| Laser-D <sub><math>L_T</math>=2048</sub>  | 82.2        | 31.0         | 73.3 | 46.2              | 58.2  | 1623        | 5158                       | 2572 | 2960              | 3059  |  |  |
| Laser-D <sub><math>L_T</math>=4096</sub>  | 84.2        | 34.2         | 75.3 | 47.3              | 60.3  | 1872        | 5750                       | 2981 | 3474              | 3520  |  |  |
| LASER-DE <sub><math>L_T=1024</math></sub> | 82.1        | 33.8         | 72.2 | 43.7              | 58.0  | 1350        | 4794                       | 2254 | 2654              | 2763  |  |  |
| LASER-DE <sub><math>L_T</math>=2048</sub> | 83.9        | 31.5         | 75.3 | 46.4              | 59.3  | 1456        | 5263                       | 2679 | 2971              | 3092  |  |  |
| LASER-DE $_{L_T=4096}$                    | 83.5        | 35.0         | 73.3 | 46.0              | 59.5  | 1949        | 5789                       | 3080 | 3488              | 3577  |  |  |

<span id="page-18-0"></span>these improvements extend even to the knowledge-intensive MMLU benchmark, highlighting the robust generalization capabilities of our proposed methods.

### J Visualization Details

In this appendix, we provide details about the visualization of different reward functions depicted in Table 2. These visualizations illustrate how different methods calculate rewards based on response length.

<span id="page-19-0"></span>> **[图片提取文字 (无描述)]:**
> Model Performance on GPQA Model Performance on LSAT 26 36 25 Acc 24.62% Accuracy (%) Accuracy (%) Acc 33.62% 23 Original Model Original Model --- Truncation Truncation 33 -+- Group-based -+- Group-based --- LASER --- LASER --- LASER-D 22 --- LASER-D - LASER-DE --- LASER-DE 15421 3000 4000 5000 7000 11200.8 4000 6000 8000 2000 6000 2000 **Average Tokens** Average Tokens (a) GPQA (b) LSAT Model Performance on MMLU Model Performance Across All Benchmarks 37 48 Acc 34.59% 35 47 Accuracy (%) Accuracy (%) 33 Acc 45.52% 31 Original Model Original Model --- Truncation Truncation 44 29 -+- Group-based -+- Group-based --- LASER - LASER --- LASER-D --- LASER-D 43 - LASER-DE LASER-DE 27 4000 2691.7 2000 5000 0 1000 3000 9771.2 6000 **Average Tokens Average Tokens** (c) MMLU (d) Average
![](_page_19_Figure_0.jpeg)

Figure 11: Performance on out-of-domain benchmarks including GPQA [17], LSAT [29, 23], and MMLU [8].

#### J.1 Visualization Parameters

Each visualization captures the relationship between response length and reward value with the following specifications:

- **X-axis**: L(y) represents the response length, ranging from 0 to 20 tokens.
- Y-axis: Reward value, with different ranges depending on the method.
- Line styles: Solid lines represent rewards for correct responses (blue), while dashed lines represent rewards for incorrect responses (red).
- Target length  $(L_T)$ : Set to 10 tokens for all methods.

The visualizations were generated using a high-resolution grid of 400 points between 0 and 20 tokens.

#### J.2 Unified Reward Formulation

Each method can be represented using the unified reward formula:

$$\hat{R}(x,y) = C(y) + \lambda(y) \cdot S(y)$$

We implement the specific components for each method in this simulation as follows. Note that the paramters are only used for better visualization which are different from the practical experiments.

#### **Vanilla Truncation**

$$\begin{split} &C(y) = 0 \\ &\lambda(y) = 1 \\ &S(y) = \begin{cases} R(x,y) & \text{if } L(y) \leq L_T \\ \rho & \text{if } L(y) > L_T \end{cases} \end{split}$$

where  $L_T = 10$  and  $\rho = 0$ .

#### **ThinkPrune**

$$C(y) = 0$$

$$\lambda(y) = 1$$

$$S(y) = \begin{cases} R(x, y) & \text{if } L(y) \le L_A \\ \rho & \text{if } L(y) > L_A \end{cases}$$

where  $L_A \in \{10, 7.5, 5\}$ .

### **Efficient Reasoning**

$$\begin{split} &C(y) = R(x,y) \\ &\lambda(y) = \mathbb{I}(R) \\ &S(y) = -\alpha \cdot \sigma \left( \frac{L(y) - Mean(y)}{STD(L)} \right) \end{split}$$

where  $\mu = 10$  and  $\sigma = 2$ .

#### Kimi-k1.5

$$\begin{split} &C(y) = R(x,y) \\ &\lambda(y) = 1 \\ &S(y) = \begin{cases} 0.5 - \frac{L(y) - L_{\min}}{L_{\max} - L_{\min}} & \text{if } \mathbb{I}(R) = 1 \\ &\min\left(0,\ 0.5 - \frac{L(y) - L_{\min}}{L_{\max} - L_{\min}}\right) & \text{if } \mathbb{I}(R) = 0 \end{cases} \end{split}$$

where  $L_{min} = 2.5$  and  $L_{max} = 20$ .

### L1-Exact

$$C(y) = R(x, y)$$

$$\lambda(y) = 1$$

$$S(y) = -\alpha \cdot |L(y) - L_T|$$

where  $\alpha = 0.03$  and  $L_T = 10$ .

#### L1-Max

$$C(y) = 0$$

$$\lambda(y) = \mathbb{I}(R)$$

$$S(y) = \text{clip}(\alpha \cdot (L(y) - L_T) + \delta, 0, 1)$$

where  $\alpha = 0.03$  and  $L_T = 10$ .

