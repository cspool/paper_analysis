## Data-Driven Expert Predictor (数据驱动的专家预测器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Data-Driven Expert Predictor 是运行在 wafer-scale GPU 的 Global Command Processor 上的轻量级预测算法，利用 cross-token heatmap（历史 expert selection 条件概率）在 MoE kernel launch 时预测下一 token 的热门 experts，生成 duplication guidance（cp_en bits）指导各 die 的 hardware-managed HBM 自动缓存远程热门 expert。Predictor 是 Insight 2（cross-hierarchy memory management）在 token-level 的具体实现。其设计理念是：与其在 kernel 执行后被动响应 cache miss，不如在 kernel launch 时利用 temporal correlation 主动预测并 prefetch/cache。

从kernel调度角度拆解术语：
Predictor 在 kernel 调度流程中的嵌入位置：

```
Wafer-scale GPU MoE kernel 执行流程（含 Predictor）:
==================================================
Phase 1: Global CP (kernel launch 时)
------------------------------------------
1. 读取当前 batch 的 expert selection E_curr
   （来自上一层 MoE kernel 的输出或 prefill traces）

2. 运行 Predictor:
   for each expert e_id in E_curr:
       row = cross_token_heatmap[e_id]  // 从 Heatmap Cache 读取
       top_n = argsort(row)[-n:]        // top-n most likely next experts
   E_pred = union of all top_n results  // 合并去重

3. 对每个 die d:
   E_die = 该 die 当前计算的 experts
   for e in E_pred ∩ E_die:
       if e not already in die d's local HBM:
           set cp_en[e] = 1  // 标记为应缓存

4. 将 cp_en bits 打包发送到各 die 的 Local CP
   Local CP 配置到 D2D controller 的 PDU Prediction Table

Phase 2: Per-die execution (kernel 执行期间)
------------------------------------------
5. SM 请求 expert 数据:
   if is_local[expert]:
       ATU 翻译远程地址到本地地址 → 从本地 HBM/LLC 读取
   else:
       D2D XY routing → 从远程 die 读取
       返回时 PDU 检查 cp_en[expert]:
           if cp_en[expert] == 1:
               写入本地 HBM + LLC
               更新 ATU entry
               设置 is_local[expert] = 1
```

Predictor 的输出不是硬性约束（即不等同于 "必须用这些 expert"），而是"如果这些 expert 被远程访问，值得本地缓存"的 soft guidance。这种方式天然容忍预测错误——预测错误仅浪费少量本地 HBM 空间，不影响计算正确性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 Global CP 上的软件算法（运行于 A76-class ARM core），不增加专用硬件。输入 heatmap 存储在 Global CP DRAM（50 MB full）中，运行时仅需要 0.5 MB on-chip cache 缓存一层 heatmap。
- 每 token prediction 的额外开销极低（仅查表 + 取 top-n + 组合 cp_en），完全隐藏在 kernel launch 的常规 overhead 中。
- 在 Dojo 5×5 配置上的效果：Pred Only（仅 predictor, 无 task allocation）在 DeepSeek V3 上实现 3.0× throughput 提升，hop count 降低 4.5×。Allo+Pred（task allocation + predictor）进一步实现 7.0× throughput 和 >213× hop count 降低。
- 局限：Predictor 效果取决于 cross-token correlation 的强度——Llama 4 受益最多（top 20% candidates cover 80% mass），DeepSeek V3 受益最少（47%）。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

---
