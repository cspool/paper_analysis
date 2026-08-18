## Top-score Expert Prefetching（top-score 专家预取 / 高分组专家定向预取）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-score Expert Prefetching 是 SMoE 的在线预取策略：只预取下一层 gate score 最高的（top-score）专家，把 PCIe 加载与当前层 GPU 计算重叠，隐藏加载延迟。与 HybriMoE/MoE-Infinity 等"预测多层、预取全部缺失专家"的做法相反，SMoE 只预取少量高影响专家，理由是 PCIe 远比计算慢，盲目预取会造成带宽压力、常来不及加载而瓶颈化流水线；只预取少量 top-score 专家显著降低 PCIe 压力，且 top-score 专家即使预测不完全准确也因输出影响大而保证预测收益。预测在 GPU 上完成（GPU 相对 CPU/PCIe 更空闲，预测开销不占 TPOT）：用 GPU 中已缓存的未共享专家 + 常驻共享专家生成 hidden state → 用下一层 KV cache 完成 attention → 计算下一层 gate 分数预测 top-score 专家；cache eviction 保证高分专家驻留使预测更准。实测预取命中率约 82%（预测为 top-score 的专家确实 top-score），非 top-score 时 95% 概率仍为 active。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# 层 i 计算期间（PCIe 空闲窗口）并行预取层 i+1
h = shared_experts(x_i) + cached_experts(x_i)   # GPU 上，仅用驻留参数
h_attn = Attention_next_layer(h, kv_cache_{i+1}) # 用下一层 KV cache 完成 attention
scores_{i+1} = gate_next_layer(h_attn)           # 预测下一层全部专家分数
top_score_{i+1} = TopK(scores_{i+1}, k_prefetch) # 只预取 top-score 专家
PCIe: load top_score_{i+1}  → GPU               # 与层 i 的专家计算重叠
# 预测错误时：立即清出加载队列、插入正确专家，回退 baseline 无性能惩罚
```
配合流程：预取开始越早（层 i 计算一开始就预测）能预取越多专家（C5），但要以预测准确性为前提（C6）；预取目标与 expert-cache router 的替换结果联动——被替换掉的 low-score 专家不需要加载，进一步减少预取量（Fig.2 例子中预取量从 3 降到 1）。消融（+Pre）相对 +CR 再降 TPOT 14%、cache ratio +12%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 SMoE 开源运行时（https://github.com/goingshr/SMoE）中由 config 的 if_prefetch 字段开启，GPU 预测在解码循环内自动执行、经 PCIe 异步加载。代表性预取工作对比：MoE-Infinity 用历史请求的 activation matrix 预测（需历史数据、记录不全时精度下降）；HybriMoE 预测多层全部缺失专家（带宽压力大）；SMoE 只预测单层 top-score 子集（Fig.8 展示了与传统方法、正常流程的对比）。预取距离（prefetch distance）、预取剪枝（prefetch pruning）等是该方向的其他变体研究点。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution
