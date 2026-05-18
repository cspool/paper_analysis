## Headroom-based Token-level Scheduling

术语是什么？

Headroom-based Token-level Scheduling是SLINFER提出的token级LLM推理调度机制：将多个model instance的prefill/decode迭代按token粒度交错调度到共享硬件上，每次调度cycle选择headroom最短的instance执行一个iteration。Headroom定义为请求距离SLO violation的剩余时间余量：headroom = ST + TTFTSLO + TPOTSLO × O − CT，其中ST为请求到达时间、O为已生成token数、CT为当前时间。负headroom表示SLO已violate。该机制使instance间精密共享compute资源，避免idle的同时保证SLO compliance。

从系统架构角度拆解术语：

Headroom-based scheduling在SLINFER中的运转流程：
1. **Quantification**：对每种硬件类型（CPU/GPU）和每个模型，通过sampling profiling获取：(a) prefill time：与input length近似线性相关→linear interpolation (O(log Lmax) samples)；(b) decode time：与batch size和average token length相关→2D linear interpolation (O(log Lmax · log Bmax) samples)。Estimator偏差：TTFT 5.9%, TPOT 3.9%。
2. **Scheduling Cycle**：每个cycle内，scheduler计算所有instance中最短headroom请求，选择该instance执行一次iteration（prefill或decode）。在Figure 14例子中：Instance-2请求headroom=1.9s最短→scheduler选其执行0.2s decode→headroom更新为1.9−0.2+0.25=1.95s→下一cycle重新比较。
3. **Headroom Update**：每次iteration完成后，该请求的O增1→headroom减少(iteration_time − TPOTSLO)。若iteration_time < TPOTSLO则headroom增加（提前完成），反之减少。
4. **SLO Check**：任何时刻headroom < 0表示SLO violation，scheduler通过shadow validation预防而非事后检测。

与Continuous Batching的对比：(1) CB以batch为单位调度，headroom scheduling以token iteration为单位；(2) CB在request-level混合prefill+decode，headroom scheduling区分prefill/decode iteration分别调度；(3) CB的调度目标是最大化throughput，headroom scheduling的目标是最小化SLO violation风险。

术语一般如何实现？如何使用？

实现要点：
- **Profiling阶段**：对每种(硬件类型, 模型)对，运行sampling benchmark收集不同input length和batch size下的prefill/decode time，构建interpolation table。O(log Lmax · log Bmax) = 几百个sample，数分钟内完成。
- **在线Scheduling**：每iteration后更新所有instance的headroom，O(N)扫描选最小值。SLINFER实测scheduling overhead与cluster规模无关（每node独立决策），shadow validation随node数轻微增长（更多candidate instance需probe）。
- **Shadow Validation**（见独立条目）在添加新请求前使用headroom预测来防止SLO violation。
- **Precision overestimation**：量化误差和runtime波动→SLINFER对每iteration时间overestimate 10%作为安全margin。

涉及论文标题：
- Towards Resource-Efficient Serverless LLM Inference with SLINFER
