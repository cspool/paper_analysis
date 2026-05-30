## FasterTransformer

术语是什么？
FasterTransformer 是 NVIDIA 开发的高性能 Transformer 推理库（已归档，功能整合至 TensorRT-LLM），提供针对 NVIDIA GPU 手动优化的 Transformer 层 kernel 实现，包括 Attention、FFN、LayerNorm 等核心组件的融合 kernel。与 PyTorch/HuggingFace 等通用框架不同，FasterTransformer 绕过高层次的框架开销，直接调用 cuBLAS/cuBLASLt 和自定义 CUDA kernel，实现更高的 GPU 利用率和更低的推理延迟。支持 encoder-only（BERT）、decoder-only（GPT、LLaMA）和 encoder-decoder（T5）等多种 Transformer 变体。

从编译框架角度拆解术语：
FasterTransformer 作为推理编译框架的工作流程：

1. **模型转换（Offline）**：将 HuggingFace/NeMo 格式的 PyTorch 模型权重转换为 FasterTransformer 自定义的二进制格式。Diff-MoE 在其基础上进一步将 model.bin 拆分为 per-expert 细粒度文件以实现 per-expert offloading。
2. **图构建**：定义 Transformer 推理的计算图——embedding lookup → [Per Layer: LayerNorm → Attention (QKV projection + multi-head attention + output projection) → LayerNorm → FFN/MoE (gating + expert dispatch + SwiGLU + combine)] → final LayerNorm → logits projection → sampling。
3. **Kernel 选择与执行**：每个算子使用预优化的 CUDA kernel：
   - Attention：fused multi-head attention kernel（可选用 FlashAttention）。
   - FFN：fused GEMM + activation kernel（如 `gelu`/`swiglu` + matmul 融合为一个 kernel launch）。
   - MoE：expert dispatch + parallel expert FFN + token reorder。FasterTransformer 的 MoE 实现支持 top-1/top-2 gating。
4. **内存管理**：预分配 GPU memory pool，在推理过程中复用缓冲区。Diff-MoE 在其内存管理基础上增加了三级缓存（HPC/MPC/LPC）的显存分配逻辑。
5. **输出**：生成 token ID 序列。

Diff-MoE 基于 FasterTransformer v5.2 构建（CUDA 12.4, PyTorch 1.13.0, NCCL 2.15.1, Transformer 4.31.0）。核心修改：
- 注入 per-expert 文件加载逻辑（替代整体模型加载）
- 在 gating 后插入三级缓存查找与 host→GPU 传输逻辑
- 在 MoE 层计算后插入优先级更新与 replacement 逻辑
- 增加 GRU predictor 推理通路（当前层计算完后异步触发预取）
- 不需要修改 FasterTransformer 的 attention/MoE kernel 本身

术语一般如何实现？如何使用？
FasterTransformer 通过 C++/CUDA 实现核心 kernel，Python 端提供用户接口。典型使用方式：
```python
import fastertransformer as ft

# 加载转换为 FasterTransformer 格式的模型
model = ft.GptModel.from_pretrained(model_dir, ...)

# 在线推理
output = model.generate(input_ids, max_length=128)
```

NVIDIA 已宣布 FasterTransformer 进入维护模式，推荐用户迁移到 TensorRT-LLM。TensorRT-LLM 提供了更强的图优化（如 kernel fusion、memory planning）、更灵活的 backend（支持 in-flight batching、paged attention）和更广的模型支持（包括 MoE）。

涉及论文标题：
- Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching
- Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

"Who Says Elephants Can't Run" 基于 FasterTransformer 构建了首个生产级单 GPU MoE 推理系统，核心扩展包括：(1) 添加 DeepSpeed MoE 模型格式支持；(2) 实现基于 CUB radix sort + CUTLASS Grouped GEMM 的 GPU-efficient token routing；(3) 实现 INT4/INT8 Fused GEMM+Dequantize kernel；(4) 实现 MoE decoder batch pruning；(5) 与 Triton Inference Server 集成实现云规模弹性部署。

Pre-gated MoE (ISCA '24) 基于 FasterTransformer 构建，修改内容：(1) 实现分层参数存储——MoE expert 参数 offload 到 CPU，non-MoE 参数常驻 GPU；(2) 实现 preemptive expert migration pipeline——利用 pre-gate function 输出通过独立 CUDA stream 异步迁移下一 block 激活 experts；(3) 修改 MoE block forward——插入 pre-gate Linear layer，输出传递给下一个 block；(4) 第一个 MoE block 实现双 gate（传统 gate + pre-gate），最后一个 block 无 pre-gate。
