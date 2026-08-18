## 场景感知自动化阈值搜索（Scenario-aware Pareto 多目标阈值搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
为 DiTPA 的三层冗余消除策略（朝向阈值 th、迭代跳过比例）自动选取阈值的离线搜索机制：用场景特定校准任务集（占总任务 3%）采样候选阈值组合，对成功率与速度做多目标 Pareto 优化得到前沿，再按当前场景需求沿前沿选点。场景分类（对齐 VLA 加速框架惯例，如 Sp-VLA）：latency-sensitive（如分拣机器人：动作频率 ≥200Hz、容忍 ≤2% 成功率损失，错误可由 re-planning 补救）与 mission-critical（如救援机器人：不可重规划、要求近零精度损失）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
搜索流程伪代码：
```
calib = 场景特定校准任务集（总任务数的 3%）
candidates = grid(orientation_th ∈ {0°,1°,2°,3°,4°}, skip_ratio ∈ {...})
for (ot, sr) in candidates:
    (success_rate, action_freq) = evaluate(calib, ot, sr)   # 模拟器 rollout
front = pareto_front(candidates, maximize=success_rate, maximize=action_freq)
th = pick_on_front(front, requirement)   # latency-sensitive: ≥200Hz 且损失 ≤2%；mission-critical: 损失≈0
# 部署期：同场景任务共享动作空间动力学与轨迹模式，固定朝向阈值
# 未来工作：按相邻视觉帧平均光流幅值自适应调整阈值（高动态环境）
```
结果：LIBERO-Long 属 latency-sensitive，搜索得朝向阈值 2°、迭代跳过 40%，并每 20 个跳过迭代插入完整去噪；收紧朝向阈值可得 226.68× 加速且零成功率损失，收紧迭代跳过阈值得 245.69× 且零损失（均归一化到 GPU）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：离线 Python 搜索 + LIBERO 模拟 rollout 评估（论文未开源该搜索脚本细节，论文未明确说明具体实现工具）。使用：不同机器人场景部署前的参数整定；与消融/敏感性分析配合（论文 Fig.20/21 展示阈值 0°–4° 的成功率-加速权衡曲线：>3° 成功率骤降、>2° 加速收益饱和——因为新增跳过动作多为重复抓取等无效操作）。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
