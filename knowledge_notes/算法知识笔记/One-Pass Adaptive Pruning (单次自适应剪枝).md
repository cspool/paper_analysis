## One-Pass Adaptive Pruning (单次自适应剪枝)

术语是什么？
One-Pass Adaptive Pruning 是 UniQL 提出的部署策略：在云端对 LLM 进行一次性的结构化权重排序、masked LoRA 微调和量化（single pass），产出一个包含已排序通道的量化模型。部署到边缘设备后，设备端仅需根据当前系统负载（可用内存）决定剪枝率，从量化权重末尾去除指定比例的通道即可获得对应尺寸的模型。该策略的核心思想是将计算密集的排序-微调-量化"压缩"为单次 O(1) 操作，而非为每种剪枝率独立执行 O(n) 次完整压缩流程。

关键设计：
1. **隐藏维度不变**：剪枝仅缩减中间维度（$D_{int}$、$D_{hd}$、$D_s$），隐藏维度 $D_h$ 在所有剪枝率下保持不变，保证各层间的维度匹配。
2. **Hadamard 矩阵不作用于剪枝通道**：Hadamard rotations 仅融合到非剪枝侧的矩阵（如 `q_proj` 的输入 Hadamard 为 Yes，输出为 No*），避免剪枝后预融合矩阵形状不匹配。
3. **设备端 INT4 在线处理**：部署的 4-bit 权重在运行时解包（unpack from INT4）→ 按当前剪枝率去除末尾通道 → 重新打包为 INT32 → 送入 GEMM kernel。

从算法pipeline角度拆解：
```
# 云侧 (one-pass, 单张 A6000 GPU)
Step 1: Ridge leverage scores / QSVD / state-aware 排序各模块权重通道
Step 2: BI scores 计算所有目标剪枝率的层间分配 (O(1) for multiple rates)
Step 3: Masked LoRA 微调 5 epochs (每步随机采样剪枝率训练)
Step 4: GPTQ W4A16 量化 (含 embedding/output 层, head-to-toe)
Step 5: 产出单个 INT4 模型文件 (Llama-3.1-8B: 4.1GB)

# 设备侧 (每次自适应)
Input: device_memory_budget ← OS 报告当前可用内存
If device_memory_budget > 4.1GB: p = 0%    # 满精度量化
elif device_memory_budget > 3.4GB: p = 15%  # 温和剪枝
elif device_memory_budget > 3.0GB: p = 25%  # 中度剪枝
else: p = 35%                               # 激进剪枝

For each prunable layer:
    D' = D * (1 - p%)                       # 新维度
    W_int4 = load_from_file(layer)          # 加载 INT4 packed weights
    W_fp16 = unpack(W_int4)[:, :D']        # 解包，取前 D' 列
    W_int32 = repack(W_fp16)                # 重新打包
    // 加载到 GPU 并使用 Marlin/Marlin-like GEMM kernel 执行
```

术语一般如何实现？如何使用？
适用场景：动态工作负载的边缘设备（如手机、VR/AR 眼镜、Jetson Orin），操作系统在不同时刻分配给 LLM 推理的内存不同。实现需要：①云侧 GPU 48GB+（排序 + 微调 + 量化），②客户端推理框架支持动态 INT4 权重裁剪和重打包，③融合 RoPE kernel 处理排序后的位置嵌入索引。UniQL 在 Jetson Orin Nano 8GB 上验证：从 TAO-HQQ 的 W4A16 固定模型（5.7GB, 133.6ms TPOT）降至 head-to-toe 4-bit + 自适应剪枝模型（4.1GB @ 0%, 3.4GB @ 15%, 2.8GB @ 35%），TPOT 对应 77.2ms / 64.0ms / 57.7ms。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs


---
