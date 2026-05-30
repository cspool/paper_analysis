## τ-leaping for Discrete Diffusion Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

τ-leaping（tau-leaping）最初来自化学动力学中的Gillespie算法扩展，用于加速离散状态随机过程的模拟。在离散扩散模型中，τ-leaping是一种近似加速策略：精确反向扩散过程每步仅修改1个token（逐个token去噪），需要L步完成长度L的序列——极其低效。τ-leaping允许在单步中同时更新多个[MASK]位置的token，大幅减少所需步数。具体地，对于吸收态（masked）扩散的反向过程，τ-leaping近似为：

$$q_{s|t}(x_s^i | x_t) = \begin{cases} 1, & x_t^i \neq [\text{MASK}], x_s^i = x_t^i \\ \frac{s}{t}, & x_t^i = [\text{MASK}], x_s^i = [\text{MASK}] \\ \frac{t-s}{t} q_{0|t}(x_s^i | x_t), & x_t^i = [\text{MASK}], x_s^i \neq [\text{MASK}] \end{cases}$$

其中t和s为扩散时间步（s < t），x_t为当前带噪声序列，q_{0|t}为模型预测的干净数据分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LLaDA中τ-leaping推理（baseline，无KV Cache）：

```
Input: prompt p0, total_length L, num_steps T (e.g. 128)
   x = [p0; [MASK] * (L - |p0|)]       # 全[MASK]初始化
   timesteps = linspace(1, 0, T+1)      # T=128等分
   
   for step in range(T):
       t = timesteps[step]              # 当前噪声水平
       s = timesteps[step+1]            # 目标噪声水平
       
       logits = model(x)                # full bidirectional forward
       probs = softmax(logits)
       
       for each masked position i:
           if x[i] == [MASK]:
               # 以 (t-s)/t 概率解码为预测token，否则保持[MASK]
               with prob (t-s)/t:
                   x[i] = argmax(probs[i])
               else:
                   x[i] = [MASK]        # 概率 s/t
```

核心问题：多个[MASK] token在单步中并行解码时，采样假设条件独立——product of marginals: q(X|E) = Π_j p_j(X_{i_j}|E)——但真实数据分布p(X|E)包含token间条件依赖。当每步解码大量token时，独立性假设的偏离会导致质量下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

τ-leaping在LLaDA和Dream的官方实现中通过schedule控制每步解码的token数量。LLaDA默认最优策略是每步1 token（非并行），因为τ-leaping越激进（每步越多token）质量越差。Fast-dLLM通过confidence过滤改进τ-leaping：不是随机解码掩码token，而是先计算置信度，仅在高置信度时并行解码多个token，从而在加速的同时保持质量。τ-leaping步数T的选择：T越大→每步修改越少token→质量好但慢；T越小→每步修改越多token→快但质量差。LLaDA默认T=128，Fast-dLLM通过减少实际NFE（number of function evaluations）加速推理。

涉及论文标题：
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
