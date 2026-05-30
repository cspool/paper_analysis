## Adaptive Label Sampling（自适应标签采样 for ZSQ Object Detection）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Label Sampling 是 Task-Specific Zero-shot Quantization-Aware Training for Object Detection 论文提出的核心创新方法，用于在零样本场景下为检测任务合成带边界框标签的校准数据。其核心思想是：利用预训练全精度检测网络的知识（而非真实标注），通过交替迭代——图像优化和标签更新——逐步使合成标签收敛到teacher模型认可的高质量检测目标集合。具体步骤：(1) 初始化：随机生成高斯噪声输入x和包含单一随机目标的标签y（类别~U(0,C)，bbox中心~U(W/2,1-W/2)，宽高~U(0.2,0.8)）；(2) 循环：每固定迭代步，用teacher模型对当前x做检测推理，取conf > conf_thresh的高置信度预测作为new_tgts，计算IOU(new_tgts, 当前tgts)，添加与现有标签不重叠的新标签(max_iou < iou_thresh)，移除未被teacher重新检测到的旧标签，确保每张图至少保留一个标签；(3) 标签收敛后，固定标签y*，重新初始化高斯噪声x，用task-specific损失L_total = alpha_prior*L_prior + alpha_detect*L_detect(phi(x), y*) + L_reg优化生成最终合成图像。该方法无需任何真实标注或额外网络（如预训练生成模型），仅依赖预训练检测网络的内嵌知识。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以YOLOv5-s在MS-COCO上的Adaptive Label Sampling两阶段流程为例：

```
// Algorithm 1: Adaptive Label Sampling 核心循环
Input: current_image x, current_labels tgts, teacher_model phi(theta),
       conf_thresh, iou_thresh

1. new_tgts = phi(theta).predict(x)[conf > conf_thresh]
   // 用teacher对当前图像做检测推理，取高置信度预测

2. ious = IOU(new_tgts, tgts)
   // 计算新预测和已有标签之间的IOU矩阵

3. add_tgts = new_tgts[(max(ious, dim=1) < iou_thresh)]
   // 添加不与已有标签重叠的新标签

4. minus_tgts = (max(ious, dim=0) < iou_thresh).bool()
   // 找出未被teacher重新检测到的旧标签

5. tgts = tgts[~minus_tgts]            // 移除失效的旧标签
6. tgts = cat([tgts, add_tgts], dim=0) // 合并新旧标签


// 整体两阶段流程：
// Stage 1: Label Sampling (低分辨率 160x160)
x = GaussianNoise(3, 160, 160)
tgts = [random_single_label()]           // 按Table 6的均匀分布随机初始化
for step in range(N1):
    x = Adam(x, lr=1e-2).step(grad(L_total(x, tgts)))  // 优化图像
    if step % interval == 0:
        AdaptiveLabelSampling(x, tgts)    // 调用上述核心循环更新标签
y_star = tgts                             // 固定最终标签

// Stage 2: Image Synthesis (高分辨率 640x640)
x_new = GaussianNoise(3, 640, 640)
for step in range(2500):  // YOLOv5: 2500次迭代
    L = alpha_prior*L_prior + alpha_detect*(L_category + L_box + L_conf) + L_reg
    x_new = Adam(x_new, lr=1e-2, cosine_annealing)
// 输出：高质量task-specific合成校准集 {(x_new_i, y_star_i)}
```

关键设计决策：(1) 两阶段优于单阶段——标签持续变化导致图像优化目标不稳定（消融: 两阶段W6A6 mAP=32.1 vs 单阶段30.6）；(2) 低分辨率采样+高分辨率合成——节省计算且标签采样足够；(3) 最终标签固定后重新初始化高斯噪声——避免前期不准确标签污染图像。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在PyTorch中实现，teacher模型保持eval模式且参数冻结。超参数配置：(1) 初始随机标签：单目标，类别U(0,C)，bbox宽高U(0.2,0.8)，中心U(W/2,1-W/2)；(2) 标签采样分辨率160，合成分辨率640；(3) YOLOv5超参{alpha_detect, alpha_BN, alpha_TV, alpha_l2} = {0.5, 0.01, 0, 5e-4}，2500次迭代；(4) YOLO11超参={1e-3, 1e-3, 0, 5e-5}，3000次迭代；(5) Transformer-backbone Mask R-CNN超参={10.0, 1.0, 0, 1e-3}，4000次迭代；(6) Cutout增强提升多样性。生成2k张校准样本仅需1/60的原始训练数据量（MS-COCO 120k vs 2k），在8x RTX 4090上生成速度约256张/20分钟。消融证明该方法在完全data-free场景（无真实标签、无分布信息）下超越最优in-distribution proxy数据集2.3% mAP@W6A6，仅比使用真实标签差0.7% mAP。开源代码：https://github.com/DFQ-Dojo/dfq-toolkit。

涉及论文标题：
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

---
