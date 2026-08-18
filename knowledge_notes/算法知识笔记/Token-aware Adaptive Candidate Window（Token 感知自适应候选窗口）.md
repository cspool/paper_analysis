## Token-aware Adaptive Candidate Window（Token 感知自适应候选窗口）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STEP 的在线自适应机制：为每个 MoE 层维护奖励分 r_i 与候选窗口大小 d_i（d_i 同时控制"用多少步收集投票"与"当选专家保留多久"），按实际预取准确率动态调整窗口大小，解决固定窗口的调参困境——短窗口上下文不足、预测不可靠，长窗口易误判、过度预取浪费带宽。规则：窗口结束评估该窗口预取准确率——准确率 > th_s(75%) 则 r_i+1，r_i 累计达奖励阈值 τ（窗口 1/2 时 τ=4、窗口 4 时 τ=3、窗口 ≥8 时 τ=3，表 I）则窗口翻倍并重置 r_i；准确率 < th_f(40%) 则窗口减半并重置；介于 th_f 与 th_s 之间则窗口不变并重置。窗口缩到 1 时停用实际预取（收益有限、防带宽浪费），但继续统计投票，准确率回升后重新激活。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# per layer l，每个解码窗口结束
acc = prefetch_accuracy(window)          # 该窗口预取命中率
if acc > th_s(0.75):
    r_i += 1
    if r_i >= τ: d_i *= 2; r_i = 0       # 连续高准确 → 翻倍窗口（Fig.8a：4→8）
elif acc < th_f(0.40):
    d_i //= 2; r_i = 0                   # 低准确 → 减半（Fig.8b：8→4）
else:
    r_i = 0                              # 平均准确 → 保持（Fig.8c）
if d_i == 1: disable_prefetch()          # 仅统计投票，不实际预取
```
Annotations：r_i=奖励分、d_i=窗口大小、th_s=75%（Good Candidate Accuracy）、th_f=40%（Poor Candidate Accuracy）、τ=奖励阈值。设计动机（Fig.4/5/14）：专家选择的时间连续性在早期解码 step 弱、后期才稳定（Fig.14b：token 0–20、100–120 少预取，200–220、300–320 积极预取）；不同层时间模式不同（Fig.14b 按层独立调窗）；Fig.14a 显示自适应窗口的预取准确率与生成质量始终优于任何固定窗口（4/6/8/16）。Fig.14c 定量：预取准确率 >75% 时增加预取专家数显著降延迟，<40% 时过度预取反而增加延迟——这是 th_s/th_f 取值的依据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每层独立维护 (r_i, d_i)，在窗口边界依据命中率反馈更新；报告 Window Size 为运行时所有层 d_i 的平均（Table II-IV 中窗口 6/8 等）。使用场景：与"临时共享专家选举"同一框架内协作——窗口长度决定投票跨度与当选专家保留时长，两者共享同一套投票统计。效果：消融（Fig.13）中在 spatial allocation + prefetch 基础上再加自适应窗口把加速从 1.52× 提到 2.22×；实验对比固定窗口（Fig.14a）证明自适应在 LongBench 上命中率与 Rouge-L 双优。边界条件：窗口=1 时预取禁用（保留统计、可重新激活），当选专家整窗口固定避免频繁换出。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
