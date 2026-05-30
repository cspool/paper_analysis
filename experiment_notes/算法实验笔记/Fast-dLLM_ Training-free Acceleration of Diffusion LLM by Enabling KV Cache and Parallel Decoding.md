## Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现两个核心算法组件：(1) **Block-wise Approximate KV Cache**：针对Diffusion LLM的双向注意力特性，设计分块生成的近似KV Cache机制（PrefixCache和DualCache），利用相邻推理步之间KV激活的高余弦相似度（≈1），在块内复用缓存的Key和Value，块完成后统一更新所有token的KV Cache。(2) **Confidence-Aware Parallel Decoding**：提出基于置信度阈值的自适应并行解码策略（threshold strategy和factor strategy），仅解码置信度超过阈值/满足因子约束的token，从理论上证明了高置信度条件下贪婪并行解码等价于贪婪顺序解码（Theorem 1: (n+1)ε ≤ 1时二者argmax相同），在保证生成质量的同时实现最多13.3×并行解码加速。

  实验比较：(i) 消融实验：LLaDA baseline vs +Cache vs +Parallel vs +Cache+Parallel (Fast-dLLM) 四组对比，评估GSM8K/MATH/HumanEval/MBPP上accuracy和吞吐量(tok/s)；(ii) Dream模型上的泛化验证；(iii) Cache变体：PrefixCache vs DualCache vs No Cache（Table 4）；(iv) 并行解码策略：threshold vs factor vs 固定token-per-step baseline（Figure 5, Table 11）；(v) Cache block size消融(4/8/16/32)（Figure 4）；(vi) 不同生成长度(256/512/1024)和不同shot数(5-shot/8-shot)下的可扩展性（Table 4-5）；(vii) LLaDA-V多模态模型MathVista/MathVerse评估（Table 3, 9-10）；(viii) 不同batch size (1-32)下PrefixCache vs LLaDA vs LLaMA吞吐对比（Figure 9）；(ix) LLaDA vs LLaDA-1.5对比（Table 12）。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB GPU（单卡），所有实验不使用任何推理加速框架（如vLLM/TensorRT-LLM）。prefill length=256 tokens（batch size scaling实验），生成长度16/32/64/256/512/1024。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaDA-Instruct（7B）、LLaDA-1.5（增强版）、Dream-Base（7B）、LLaDA-V（多模态vision-language变体）
  - Benchmark：GSM8K（5-shot数学推理）、MATH（4-shot竞赛数学）、HumanEval（0-shot代码生成）、MBPP（3-shot代码生成）、MathVista（视觉数学推理）、MathVerse（视觉数学推理）
  - 评估框架：lm-eval（标准化评估库），吞吐量指标为平均每秒生成的输出token数，计算完整序列到<eos>

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/NVlabs/Fast-dLLM（NVIDIA官方，Apache 2.0许可，ICLR 2026录用）。v1目录为本文训练无关加速方法。

  算法pipeline（以PrefixCache + Threshold策略，块大小B=32，阈值τ=0.9为例）：

  ```
  输入: pθ (MDM模型), prompt p0, 答案长度L, 块数K=⌈L/B⌉, 块大小B, 每块步数T, 阈值τ
   1: x ← [p0; [MASK]×L]                                    // 初始化：prompt后接全MASK序列
   2: 首次forward pass计算全序列attention，缓存prefix K/V       // KV Cache Init
   3: for k = 1 to K do                                      // 逐块解码
   4:     s ← |p0| + (k-1)B,  e ← |p0| + kB                  // 当前块起止索引
   5:     for t = 1 to T do
   6:         复用缓存prefix K/V，对x[s:e)（及prefix）计算attention  // 仅计算当前块attention
   7:         对x[s:e)中每个[MASK]位置i: c_i = max_v pθ(X_i=v|x)  // 置信度=最大softmax概率
   8:         找出所有c_i > τ的位置，解码argmax token              // 置信度阈值过滤
   9:         若所有c_i ≤ τ，解码max c_i token                    // 保底：防止死循环
  10:        if x[s:e)全部非MASK: break                         // 当前块完成
  11:     end for
  12:     重新计算全序列attention，更新prefix KV Cache            // 块间Cache更新（复杂度与解码融合）
  13: end for
  14: return x
  ```

  DualCache变体：额外在第2步缓存suffix（全[MASK]位置）的K/V，第6步仅需对当前块B×B的query-key计算自注意力。
  Factor策略：替换第8步为：排序{c_i}按降序{c^(1), c^(2), ..., c^(m)}，找最大n使(n+1)(1-c^(n)) < f，解码top-n tokens。

  张量计算对比（PrefixCache vs 无Cache，单step）：
  - 无Cache: Q × K_full^T, 其中Q ∈ R^{B×d}, K_full ∈ R^{(|p|+L)×d} → 计算量O(B·(|p|+L)·d)
  - PrefixCache: Q × [K_prefix||K_rest]^T, K_prefix ∈ R^{|p|×d}缓存复用, 仅需Q×K_rest^T（rest含当前块+suffix的自注意+交叉注意）→ 减少重复的Q×K_prefix^T计算
  - DualCache: 进一步缓存K_suffix/V_suffix, Q×K_block仅需B×B块内自注意力 → 计算量O(B²·d)
