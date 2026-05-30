## Calibration with Positional Interpolation for KV Cache Quantization (位置插值校准)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PM-KVQ 提出的校准策略，用于解决短上下文校准数据无法准确估计 long-CoT LLM 的 Key Cache 分布的问题。根源在 RoPE：低频通道（`θ_i = θ^{-2i/d}` 随 i 增大指数衰减）的周期可达数万 token（DeepSeek-R1-Qwen-7B 最低频通道周期 54,410 tokens）。短校准数据（512-2048 tokens）仅覆盖正弦曲线的一个小片段，导致 channel-wise reparameterization factor `λ_i = (max_m K_{m,i})^α` 校准偏差。

PM-KVQ 在校准的 RoPE 中引入位置缩放：`cos(s·mθ_i)` 替代 `cos(mθ_i)`。s=4 使 2048 token 模拟 8192 token 的位置分布，低频通道在短数据中展露出完整周期。消融：2048+s=4 的 pass@1 (48.33%) 与 8192 无插值 (48.33%) 持平，超 2048 无插值 (46.67%)。

从算法pipeline角度拆解术语：校准脚本中修改 RoPE 计算 `angle = s * position_m * θ_i`（仅校准阶段）。校准完成后产出的 λ_i 和 per-block Fbit 在推理时直接使用（推理恢复 s=1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现无需修改模型结构——仅校准脚本中乘 s。s 的选择受校准长度和 target 长度影响：PM-KVQ 用 s=4（2048→8192），s 过大引入位置失真（s=16 降至 46.67%）。该方法继承自 Positional Interpolation (Chen et al. 2023) 但首次应用于 KV Cache 量化校准。代码开源：https://github.com/thu-nics/PM-KVQ。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs
