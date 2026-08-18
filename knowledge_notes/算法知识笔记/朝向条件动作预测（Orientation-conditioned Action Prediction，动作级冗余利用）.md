## 朝向条件动作预测（Orientation-conditioned Action Prediction，动作级冗余利用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
轨迹级动作冗余利用策略：机器人相邻动作具有"局部集中"特性——10 任务 × 50 初始化环境下平均朝向变化仅 2°、距离变化仅 0.5cm（55.2% 旋转变化 <2°、97.2% 平移 <1cm、最大距离变化 1.1cm）。据此，当相邻动作绝对朝向差小于阈值时直接复用当前动作、跳过下一次完整 DiT 推理。为防止"轨迹段边界误判"（相邻动作朝向差小但后续大偏差）导致误差累积，用 Skip_flag 交替强制完整推理与预测。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
伪代码（论文 Algorithm 1）：
```
输入 Act_cur，输出 Act_nxt
if Skip_flag == False:
    Act_nxt = FullDiTInference()
else:
    Rad_rel = rotation_extract(Act_cur)      # 步骤1: 取旋转向量得相对弧度
    Rad_acc += Rad_rel                       #        累积为绝对弧度
    Deg_abs = 180/pi * Rad_acc
    Deg_sim = dot(Deg_abs_i, Deg_abs_{i-1}) / (|Deg_abs_i| * |Deg_abs_{i-1}|)  # 步骤2: 朝向相似度
    Deg_diff = 180/pi * arccos(Deg_sim)      # 步骤3: 朝向差
    if Deg_diff <= th:                       # 步骤4: 阈值比较
        Act_nxt = Act_cur                    # 复用当前动作
    else:
        Act_nxt = FullDiTInference()
Skip_flag = !Skip_flag                       # 交替预测与全推理
return Act_nxt
```
预测误差边界分析：轨迹初始化与物体交互阶段才出现突变朝向，且突变可分解为多关节协同运动；跳过复用不改变整体轨迹（论文 TABLE V：轨迹长度仅 +0.62%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：软件端为 Algorithm 1；硬件端化简为无除法/三角函数的单次比较（Eq.1–3，见硬件架构层"动作预测器"条目），4 个周期完成、0.05% 功耗/0.23% 面积开销。使用：朝向阈值由自动化 Pareto 搜索确定（LIBERO-Long 场景取 2°），消除 42.28% 的动作推理（2° 阈值）；阈值 >3° 时成功率骤降（宽松约束破坏抓取等精细操作的空间定位），≤2° 时速度与成功率兼得；阈值收紧到 0° 即退化为全推理模式。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
