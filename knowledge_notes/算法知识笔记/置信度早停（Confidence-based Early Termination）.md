## 置信度早停（Confidence-based Early Termination）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 置信度早停是弹性推理在系统层的实现机制：在推理的每个时间步（或每个 spine/token 完成后）计算一个置信度分数，一旦超过预设阈值就提前终止剩余时间步，用当前输出代替完整推理输出。ELSA 的早停策略：分类任务用最大类概率（max class probability）作置信度，检测任务用检测器的 objectness score（如 YOLO 的输出）作置信度；阈值可选"保守"（保持精度）或"激进"（更大延迟缩减、轻微精度损失）。
- 它依赖一个前提：SNN 输出随时间逐步收敛（ST-BIF 等价 Q-ReLU），因此早期输出是完整输出的高质量前缀，可无损/低损提前返回。ELSA 是首个把早停粒度做到 spine/token 级的加速器——配合细粒度流水，每个 spine/token 可独立早停（Fig.20），而非像粗粒度 TBT 那样整网早停。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 早停控制流（ELSA 论文实验设置）：
```
# 分类（confidence = max class probability）
for t in 1..T:
    logits_t = 前向推理至时间步 t（累计膜/脉冲）
    p_max = max(softmax(logits_t))
    if p_max >= τ: return 输出_t          # τ=0.55：平均减 21.9% 延迟、<0.2% 精度损失
# 检测（confidence = objectness score）
for t in 1..T:
    obj = detector.objectness(t)           # YOLO 的框目标度
    if obj >= τ: return 检测结果_t         # τ=0.2：match 率 94.9%、延迟减 45.4%（1.83×）
```
- 例（ELSA Fig.1/Fig.18）：COCO2017 YOLOv2 上，FCR 最早 1.19 ms、相对完整推理 2.76×；显著目标（框面积比 0.05→0.85 on VOC2007）延迟从 2.38ms 降到 1.88ms——显著输入更快响应正是弹性推理的价值。
- Annotations：τ 是置信度阈值；分类 max class prob、检测 objectness；mismatch 定义为早停检测与最终检测类相同且 IoU>0.5；阈值越高延迟越长、mismatch 越低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：ELSA_Algorithm（PyTorch）逐时间步记录精度与置信度，产出 Fig.1b 的 accuracy-vs-latency 曲线与早停缩减表（Tab.VII）。硬件侧：ELSA 的 spine/token 流水使每个 spine/token 可独立退出（置信度高即停止该 spine 的后续时间步），ELSA Output Scheduler 调度退出事件；Fig.21 显示即使在 NoC 拥塞（注入率>0.04）下，早停带来的 cycle 缩减仍稳定 >19%。指标：Tab.VII 中 ResNet18/34/50、ViT-S 早停延迟缩减 16.6%~26.1%（保守）/ 19.3%~39.1%（激进），精度损失 <0.2%/<3.3%。Web 证据（SIREN）显示同类工作可用熵或 patience 机制替代最大概率、防抖动误停。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
