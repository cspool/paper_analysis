## CoE Offline Profiler (CoE 离线性能分析器)

术语解释
CoE Offline Profiler 是 CoServe 在系统初始化前执行的离线性能分析工具，通过运行 microbenchmarks 自动确定最优系统配置——包括最优内存分配（加载多少 expert 到 GPU）和最优 executor 数量。

术语是什么？
Offline Profiler 在部署前对每个设备执行一次，生成三类配置信息：
1. **Expert Performance Metrics**：通过 microbenchmarks 测量最大 batch size（平均延迟 plateaus 时的 batch）、执行延迟参数 K/B、加载延迟、显存占用（normalized to memory score）。同架构 expert 仅 profile 一次（计算复杂度相同）。
2. **Expert Information**：路由规则（用户提供）+ 专家使用概率（从路由规则推算或从小样本数据统计）。
3. **Memory Allocation**：通过 sliding decay window 方法在 GPU 上搜索最优加载专家数量，平衡 expert loading vs batch 推理的内存使用。

从系统架构角度拆解术语：
Sliding Decay Window 搜索最优专家数量的流程：
```
1. 构建 Expert Usage CDF: 按使用概率降序排列专家 → 累积曲线
2. 初始化窗口：[lower=0, upper=initial_window_size]
3. decay_factor = 1 - initial_window_size / 100
4. Repeat:
     window_size *= decay_factor
     load N = upper_bound experts
     用小样本数据运行推理 → 测量 throughput
     若 throughput 开始下降（实际偏离线性预测超过 error_margin）:
       break
     否则: 继续滑动窗口
5. 在停止窗口内随机选 N 作为最优 expert 数量
6. 剩余内存分配给 batch 推理的中间结果
```

术语一般如何实现？如何使用？
- 每个设备（GPU 和 CPU 独立）执行一次，生成配置后供 online 阶段使用
- 用户也可通过 memory scores 和 executor 数量手动配置
- 实现于 PyTorch，运行小样本数据集上的 microbenchmarks
- 结果直接影响 Expert Manager 的淘汰决策（依赖使用概率）和 Request Scheduler 的延迟预测（依赖 K/B 参数）

涉及论文标题：
- CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory
