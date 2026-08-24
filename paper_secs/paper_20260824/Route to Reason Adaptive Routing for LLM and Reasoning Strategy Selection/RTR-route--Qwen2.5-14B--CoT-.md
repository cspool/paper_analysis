# *RTR* route −−→ *Qwen2.5-14B + CoT* :

To solve this problem, we need to determine the current age of Raymond and then figure out how long ago his son was born. First, determine Raymond's current age. Samantha is currently 31 years old. Raymond was born 6 years before Samantha. Therefore, Raymond is currently 31 + 6 = 37 years old. Next, determine when Raymond's son was born. Raymond had a son when he was 23 years old. Raymond is currently 37 years old. The difference between Raymond's current age and the age when he had his son is 37 - 23 = 14 years. Therefore, Raymond's son was born 14 years ago. 14 . (Total 170 tokens)

Figure 11: An easy case from GSM8K. The fixed-best model-strategy pair incurs a token cost of 635. RTR predicts a high cost for this option and instead routes the question to a lightweight non-reasoning model (Qwen2.5-14B), which answers correctly using only 170 tokens.

<span id="page-14-0"></span>

|                                        |                 |          | Routi     | ng Table        |             |          |               |
|----------------------------------------|-----------------|----------|-----------|-----------------|-------------|----------|---------------|
| Question                               | Model           | Strategy | Pred Perf | Pred Ans Length | Route Score | GT Label | GT Ans Length |
|                                        | DeepSeek-R1-14B | CoD      | 0.979     | 619.3           | 0.964       | True     | 216.0         |
|                                        | QwQ-32B         | CoD      | 0.994     | 1414.0          | 0.959       | True     | 1756.0        |
|                                        | Qwen2.5-14B     | CoT      | 0.967     | 453.3           | 0.956       | True     | 333.0         |
| Divide the                             | QwQ-32B         | CoT      | 0.995     | 2051.1          | 0.943       | True     | 1963.0        |
|                                        | DeepSeek-R1-14B | CoT      | 0.980     | 1815.3          | 0.934       | True     | 3387.0        |
| reciprocal of $\frac{5}{6}$ by the the | DeepSeek-R1-7B  | CoD      | 0.946     | 720.0           | 0.928       | True     | 196.0         |
| reciprocal of $\frac{5}{6}$ .          | DeepSeek-R1-14B | Vanilla  | 0.976     | 1978.9          | 0.927       | True     | 2273.0        |
| reciprocal of $\frac{1}{6}$ .          | QwQ-32B         | Vanilla  | 0.988     | 2726.9          | 0.920       | True     | 2834.0        |
|                                        | Qwen2.5-7B      | CoT      | 0.922     | 468.6           | 0.910       | True     | 400.0         |
|                                        | DeepSeek-R1-7B  | CoT      | 0.957     | 2367.1          | 0.898       | False    | 17000.0       |
|                                        | DeepSeek-R1-7B  | Vanilla  | 0.954     | 2493.2          | 0.892       | True     | 1972.0        |
|                                        |                 |          |           |                 |             |          |               |

### ? Question (Math):

Divide the reciprocal of  $\frac{5}{6}$  by the reciprocal of the reciprocal of  $\frac{5}{6}$ . (**Ground Truth:**  $\frac{36}{25}$ )

### $\clubsuit$ Fixed Best $\rightarrow$ QwQ-32B + CoT:

<think> Okay, let's tackle this problem step by step. The question says: "Divide the reciprocal of 5/6 by the reciprocal of the reciprocal of 5/6." Hmm, that sounds a bit convoluted, but let's break it down. First, I need to recall what a reciprocal is. The reciprocal of a fraction is just flipping the numerator and denominator. So, the reciprocal of 5/6 would be 6/5. Got that part. Now, the problem mentions "the reciprocal of the reciprocal of 5/6." Let me parse that. The first reciprocal of 5/6 is 6/5. Then the reciprocal of that would be flipping 6/5 back to 5/6. So, the reciprocal of the reciprocal of 5/6 is just 5/6 again. Wait, so the second part simplifies back to the original fraction? That makes sense because taking the reciprocal twice cancels out the operation. So now the problem is asking me to divide the first reciprocal (which is 6/5) by the second term, which we just found out is 5/6. So mathematically, (More Thinking Steps...)

First, we need to find the reciprocal of  $\frac{5}{6}$ , which is  $\frac{6}{5}$ . Next, we determine the reciprocal of the reciprocal of  $\frac{5}{6}$ . The reciprocal of  $\frac{6}{5}$  is  $\frac{5}{6}$ . We then need to divide the first result by the second result. This means we need to compute  $\frac{6}{5} \div \frac{5}{6}$ . To divide by a fraction, we multiply by its reciprocal:

$$\frac{6}{5} \div \frac{5}{6} = \frac{6}{5} \times \frac{6}{5} = \frac{6 \times 6}{5 \times 5} = \frac{36}{25}$$

Thus, the final answer is  $\boxed{\frac{36}{25}}$  . (Total 1963 tokens)

