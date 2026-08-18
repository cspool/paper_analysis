## Mantissa Truncation（尾数截断）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
直接丢弃浮点数低位尾数比特以降低表示位宽的压缩手段，区别于量化：不改变数值表示（仍为浮点）、不做 round/scale 变换，仅保留高 w_t 位 mantissa、低位清零。两个关键性质：(1) 重建开销极低——zero-padding 即可恢复为完整浮点布局；(2) 截断后的草稿模型是目标模型的严格比特子集（比特级包含关系），使 self-speculative decoding 中"验证数据补齐即完全恢复原模型"成立，从而无需存储任何独立草稿参数、显存低于原始 BF16 格式。BF16 = 1 符号 + 8 指数 + 7 尾数；截断 4 位后有效尾数剩 3 位（指数另行压缩）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
W_draft_mant = (W_mant >> t) << t           # 保留高 t 位（默认 t=4），低位清零
# BF16 mantissa 7-bit → 截断后 3-bit 有效；低位在 decoder 拼接时恒为 0
# draft 前向：mantissa concatenator 把高/低位拼回 BF16 布局，dynamic shifter 对齐
```
本文用法：默认配置权重 40% 剪枝 + 4-bit 截断、KV cache 4-bit 截断（可直接迁移到其他模型）；设计空间探索（图 7）显示剪枝+截断联合使用比单独使用任一者的接受率-压缩率曲线更鲁棒（Deepseek-R1-Distillated-Llama-8B，γ=5）；超参由目标函数 J = α / (S_w(1−w_p)(B−w_t) + S_kv(1−kv_p)(B−kv_t)) 的 grid search 确定（剪枝 30–60%、截断 0–5 bit，8 样本 dev set，A100 约 5 分钟）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现即移位与掩码操作（PyTorch bitwise）；硬件上由 Cassandra decoder 的 mantissa concatenator + dynamic shifter 完成拼接与指数对齐。与量化的对比：量化改变数值表示、可能引入反量化 scale 乘法（低 batch decode 下该开销不可忽略，W8A8 仅 1.3×）；截断无此开销但单用压缩率有限，本文将其与剪枝+指数压缩组合以弥补。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
