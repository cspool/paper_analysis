## 2D Image Scanning (Multi-directional Scanning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
2D Image Scanning 是 VisualRWKV 针对 RNN VLM 的视觉序列处理机制。RWKV（及其他线性 RNN）本质上为 1D 因果语言序列设计，其 Scan 操作假定序列具有因果方向性。但视觉 encoder（如 CLIP ViT）生成的 visual tokens 来自 2D patch grid，天然是双向/多向的非因果序列。若直接用单向 Scan（Forward-only），模型只能从左上到右下依次处理，丢失了大量空间上下文。2D Image Scanning 通过在相邻 RWKV layers 中交替排列不同扫描方向（Forward/Backward/Upward/Downward），使不同层的 RWKV blocks 从不同方向"看"图像，从而在不增加任何参数和计算开销的情况下获得等效的 2D 空间感知能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种扫描变体的 layer 排列：
```
# Unidirectional (UniDir) - VisualRWKV-Base baseline
Layer 0: Forward Scan    Layer 1: Forward Scan    
Layer 2: Forward Scan    Layer 3: Forward Scan    ...

# Bidirectional (BiDir) - 论文最优
Layer 0: Forward Scan    Layer 1: Backward Scan   
Layer 2: Forward Scan    Layer 3: Backward Scan   ... 交替

# Multidirectional (MultiDir) - 四向交替
Layer 0: Forward         Layer 1: Backward        
Layer 2: Upward          Layer 3: Downward        ... 循环
```
Forward/Backward：按 patch 的 row-major 顺序正向/反向扫描。Upward/Downward：将 2D patch grid 转置后正向/反向扫描（按列扫描）。每种扫描方向等价于对 visual token 序列做特定的 permutation。关键实现细节：训练和推理时的扫描方向必须保持一致（论文尝试过动态重排 layer 顺序但性能不稳定，因为特定 layer 已"专业化"于处理特定方向的视觉信息）。交替扫描在 layers 间形成互补——例如 Forward 层的输出被 Backward 层看见，使 Backward 层能融合前向上下文做反向推断。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在 RWKV block 的输入处理中，根据当前 layer index 对输入的 visual token 序列做 permutation（Forward=none, Backward=flip, Upward=transpose+forward, Downward=transpose+backward）。每个 RWKV block 内部的 WKV scan 操作不变，仅仅是输入 token 顺序被重排。零参数开销，零额外 FLOPs。实验结果（Table 4）：BiDir VQA 65.62 > UniDir 51.03 (+14.59)，MultiDir 66.04 > UniDir (+15.01)。BiDir 在大多数 benchmark 上表现最好，是多向扫描中最高效的配置。该技术仅适用于 visual tokens——text instruction tokens 保持单向 Forward scan 不变（保持语言因果性）。论文确认已开源：https://github.com/howard-hou/VisualRWKV。

涉及论文标题：
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---
