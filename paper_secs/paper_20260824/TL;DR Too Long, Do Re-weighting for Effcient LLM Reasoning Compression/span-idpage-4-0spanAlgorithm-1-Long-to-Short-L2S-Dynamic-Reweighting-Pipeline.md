# <span id="page-4-0"></span>**Algorithm 1** Long-to-Short (L2S) Dynamic Reweighting Pipeline

```
Require: Domain data \mathscr{D}_{\text{system-1}}, \mathscr{D}_{\text{system-2}}, \mathscr{D}_{\text{dev}}; training steps T; batch size b; step size \eta; smooth-
   ing parameter c \in [0,1] (e.g., c = 10^{-4} in our implementation)
   Initialize proxy weights \theta_0
   Initialize mixture weights \alpha_0 = (1/2, 1/2)
   for t = 1 to T do
         Let |x| denote token length of example x (with |x| \le L)
         Compute benefit of fine-tuning with System-1 data: \lambda_{\text{system-1}} and System-2 data \lambda_{\text{system-2}}
         Update weights (entrywise exponential): \alpha'_{t} \leftarrow \alpha_{t-1} \cdot \exp(\eta \cdot \lambda_{t})
Renormalize and smooth: \alpha_{t} \leftarrow (1-c) \frac{\alpha'_{t}}{\sum_{i=1}^{k} \alpha'_{t}[i]} + cu
         Update proxy model weights \theta_t using L(\theta_{t-1}, \alpha_t) (e.g., via Adam, Adafactor)
   end for
   return \frac{1}{T}\sum_{t=1}^{T} \alpha_t
```

of which,  $\phi_{sys-1,\theta}$  can be regarded as a metric for measuring the efficiency of the System-1 models.  $\phi_{\text{SVS}-2,\theta}$  can be regarded as an accuracy metric. In this way, the overall optimization objective is to minimize the gap between the model and the efficiency upper bound of System-1, as well as the reasoning capability upper bound of System-2, while simultaneously optimizing the model parameters to maximize both reasoning performance and efficiency.

$$\phi_{sys-1,\text{bound}} = -\mathbb{E}_{dev}[T(M_s(x))]$$
(3)

$$\phi_{sys-2,\text{bound}} = \mathbb{E}_{dev}[C(M_l(x))] \tag{4}$$

**Setup for System-1/2 Mixed Data.** Since System-1 can provide fast and intuitive answers to simple problems, we use the short CoT model to modulate the data for the System-1 model. Since System-2 is designed to execute slow, logical reasoning for challenging problems, we employ the long CoT model to sample prompts from S1 [37], retaining only the correct responses. Finally, we obtain  $D_{system-1} = < Simple Question, Short CoT > instruction pairs. For the harder problems within$ the System-1 domain, we used the long CoT model for sampling, resulting in a large amount of  $D_{system-2} = \langle Hard \ Question, Long \ CoT \rangle$  instruction data.

### 3.2 Long-to-Short Data-Reweighting Tuning.

Step 1: Estimate the ideal upper bounds of efficiency and performance. During training, we aim to continuously adjust the ratio of System-1 and System-2 data in the post-training phase, ensuring that the model retains the reasoning capabilities of the original long CoT model while achieving the efficiency of the short CoT model. Therefore, we set the accuracy upper bound,  $\phi_{svs-2.bound}$ , of the model obtained through mixed training to match the accuracy of the original long CoT model, while setting the token lower bound,  $\phi_{sys-1,bound}$ , of the mixed model to correspond to the data lower bound of the short CoT model we constructed.

$$\phi_{\text{sys-2, bound}} = \phi_{\text{sys-2},L} = \hat{\mathbb{E}}_{\text{dev}}[C^L(x)] = \frac{1}{K} \sum_{i=1}^K \mathbb{1}[\text{Correct}(y_i^L)]$$
 (5)

$$\phi_{\text{sys-1, bound}} = \phi_{\text{sys-1, short}} = -\hat{\mathbb{E}}_{\text{dev}}[T^S(x)] = -\frac{1}{K} \sum_{i=1}^K \text{Token}(y_i^S)$$
 (6)

Step 2: Thinking Compression Post-Train with dynamic System-1/2 reasoning weights dynamically evaluate the utility of System-1 and System-2 reasoning data during training, and, guided by the performance of a reference model, adjust the sampling ratio between the two data types in real time to optimize training effectiveness.

$$\lambda_{\text{sys-1}} = \max\left(\frac{\phi_{\text{sys-1, bound}} - \phi_{\text{sys-1}, \theta_{\text{proxy}}}}{\phi_{\text{sys-1}, \theta_s} - \phi_{\text{sys-1}, \theta_l}}, 0\right)$$

$$\lambda_{\text{sys-2}} = \max\left(\frac{\phi_{\text{sys-2, bound}} - \phi_{\text{sys-2}, \theta_{\text{proxy}}}}{\phi_{\text{sys-2}, \theta_l} - \phi_{\text{sys-2}, \theta_s}}, 0\right)$$
(8)

$$\lambda_{\text{sys-2}} = \max\left(\frac{\phi_{\text{sys-2, bound}} - \phi_{\text{sys-2},\theta_{\text{proxy}}}}{\phi_{\text{sys-2},\theta_{\text{l}}} - \phi_{\text{sys-2},\theta_{\text{s}}}}, 0\right)$$
(8)

<span id="page-5-0"></span>

| Model                   |       |       | Ac   | curacy |      |          | Generation Length |       |       |       |      |       | A.C.R.  |       |
|-------------------------|-------|-------|------|--------|------|----------|-------------------|-------|-------|-------|------|-------|---------|-------|
|                         | ASDiv | GSM8K | MATH | AIME   | AMC  | Minerva  | Avg.              | ASDiv | GSM8K | MATH  | AIME | AMC   | Minerva |       |
| 7B Models               |       |       |      |        |      |          |                   |       |       |       |      |       |         |       |
| R1-Distill-Qwen         | 86.8  | 89.4  | 86.8 | 42.9   | 81.5 | 46.0     | 72.2              | 769   | 554   | 2861  | 6820 | 4510  | 3347    | _     |
| TALE-EP                 | 80.4  | 89.1  | 84.3 | 40.0   | 80.0 | 42.3     | 69.3              | 509   | 450   | 1994  | 6520 | 3892  | 2242    | 22.3% |
| ConciseCoT              | 86.0  | 89.5  | 86.2 | 41.7   | 79.6 | 46.0     | 71.5              | 532   | 457   | 2330  | 6587 | 4245  | 3347    | 12.7% |
| Avg. Merging            | 92.8  | 70.1  | 58.6 | 0.05   | 39.6 | 29.8     | 48.4              | 622   | 8552  | 8540  | 8501 | 8542  | 8544    | 3.2%  |
| Task-Arithmetic-Merging | 83.3  | 84.6  | 74.6 | 20.0   | 63.5 | 39.6     | 61.0              | 321   | 383   | 907   | 2500 | 1311  | 794     | 61.3% |
| Ties-Merging            | 74.4  | 69.7  | 59.8 | 13.6   | 42.5 | 23.2     | 47.2              | 1114  | 2475  | 4086  | 6767 | 5195  | 4306    | 0.1%  |
| Ties-Dare-Merging       | 75.9  | 72.3  | 65.4 | 14.6   | 45.6 | 24.3     | 49.6              | 1036  | 2073  | 2934  | 5483 | 3698  | 2938    | 8.3%  |
| Overthink               | 86.6  | 89.6  | 87.2 | 38.7   | 79.6 | 45.2     | 71.1              | 773   | 555   | 2898  | 6766 | 4558  | 3407    | 0.1%  |
| ThinkPrune              | 90.6  | 92.1  | 91.0 | 43.3   | 86.2 | 45.6     | 74.8              | 653   | 587   | 2379  | 6207 | 3739  | 2762    | 12.6% |
| CoT-Valve*              | 59.4  | 88.4  | 84.2 | 41.2   | 80.6 | 41.9     | 65.9              | 140   | 514   | 2144  | 6397 | 4278  | 2172    | 26.8% |
| TLDR                    | 93.0  | 87.7  | 87.4 | 41.2   | 83.1 | 44.5     | 72.8              | 147   | 253   | 1556  | 6368 | 3386  | 1451    | 44.9% |
| Δ                       | +6.2  | -1.7  | +0.6 | -1.7   | +1.6 | -1.5     | +0.7              | -622  | -301  | -1305 | -452 | -1124 | -1896   | -     |
|                         |       |       |      |        | 14.  | B Models |                   |       |       |       |      |       |         |       |
| R1-Distill-Qwen         | 80.5  | 92.5  | 86.4 | 43.4   | 79.6 | 48.2     | 71.7              | 476   | 679   | 2951  | 6701 | 4584  | 3270    | _     |
| TALE-EP                 | 77.5  | 92.4  | 85.4 | 49.2   | 80.3 | 50.0     | 72.5              | 369   | 555   | 2248  | 6551 | 4179  | 2731    | 15.4% |
| ConciseCoT              | 74.0  | 92.4  | 85.6 | 51.6   | 82.3 | 47.1     | 72.2              | 369   | 555   | 2066  | 6267 | 3878  | 2605    | 18.8% |
| Avg. Merging            | 94.8  | 90.3  | 73.0 | 10.8   | 55.0 | 44.1     | 61.3              | 167   | 366   | 5158  | 6364 | 5668  | 1084    | 30.5% |
| Task-Arithmetic-Merging | 86.5  | 86.5  | 74.2 | 13.3   | 55.3 | 36.0     | 58.6              | 238   | 368   | 870   | 2813 | 1411  | 1050    | 60.2% |
| Ties-Merging            | 79.6  | 91.3  | 82.6 | 25.4   | 72.5 | 37.1     | 64.8              | 242   | 542   | 1919  | 5913 | 3158  | 1850    | 31.8% |
| Ties-Dare-Merging       | 80.7  | 91.8  | 84.8 | 25.4   | 75.3 | 34.9     | 65.4              | 274   | 467   | 1870  | 5747 | 3182  | 1877    | 33.0% |
| Overthink               | 79.3  | 92.3  | 88.0 | 45.8   | 82.8 | 45.6     | 72.3              | 451   | 679   | 2893  | 6700 | 4464  | 3715    | 1.6%  |
| ThinkPrune              | 80.6  | 93.7  | 89.0 | 50.8   | 88.7 | 50.7     | 75.6              | 379   | 563   | 2177  | 5778 | 3327  | 2234    | 22.8% |
| CoT-Valve&              | 72.9  | 92.0  | 87.0 | 45.0   | 83.5 | 47.8     | 71.4              | 204   | 576   | 2652  | 6686 | 4392  | 2833    | 16.7% |
| TLDR                    | 88.0  | 90.9  | 86.6 | 43.3   | 83.8 | 48.7     | 73.5              | 158   | 240   | 2092  | 6403 | 3839  | 2177    | 35.8% |
| Δ                       | +8.0  | -1.6  | +0.2 | -0.1   | +4.2 | +0.5     | +2.1              | -318  | -439  | -859  | -298 | -745  | -1093   | -     |

Table 1: Performance comparison of TLDR with baselines. The accuracy is measured by sampling multiple responses from the LLMs and taking the average to reduce variance. \* denotes the CoTValve [33] result that we reproduced using the official dataset. Δ refers to TLDR in comparison with *Original*. Math' and Minerva' refer to MATH500 and MinervaMath datasets, respectively. A.C.R. means the token compression ratio computed by Eq. 9. In the table: yellow represents *prompt-based* methods; green highlights Merging-based methods; red indicates *SFT-based* and *RL-based* methods.

<span id="page-5-1"></span>

| Model              |       |      | Generation Length |      |         |      |       |      |      |      |         |      |
|--------------------|-------|------|-------------------|------|---------|------|-------|------|------|------|---------|------|
|                    | GSM8K | MATH | AIME              | AMC  | Minerva | Avg. | GSM8K | MATH | AIME | AMC  | Minerva | Avg. |
| 7B Models          |       |      |                   |      |         |      |       |      |      |      |         |      |
| Original Model     | 89.4  | 86.8 | 42.9              | 81.5 | 46.0    | 69.3 | 554   | 2861 | 6820 | 4510 | 3347    | 3618 |
| -MixChain-Z-GSM8K& | 88.4  | 84.2 | 41.2              | 80.6 | 41.9    | 67.3 | 514   | 2144 | 6397 | 4278 | 2172    | 3101 |
| -Static-Mixture    | 87.1  | 84.8 | 39.7              | 73.1 | 35.5    | 64.0 | 236   | 1221 | 5322 | 2560 | 1544    | 2177 |
| -TLDR              | 87.7  | 87.4 | 41.2              | 83.1 | 41.0    | 68.1 | 253   | 1556 | 6368 | 3386 | 1434    | 2599 |

Table 2: Performance comparison of TLDR with static baselines. The accuracy is measured by sampling multiple responses from the LLMs and taking the average to reduce variance. & denotes the CoT-Valve [33] result that we reproduced using the official dataset. Math and Minerva mean MATH500 and MinervaMath datasets.

