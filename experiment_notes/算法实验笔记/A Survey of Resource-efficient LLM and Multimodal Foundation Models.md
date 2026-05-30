## A Survey of Resource-efficient LLM and Multimodal Foundation Models

- 属于算法pipeline的实现是什么？实验比较什么？
  本文为综述论文，无原创实验。综述范围覆盖资源高效算法，包括：
  (i) **高效注意力机制**（§3.1）——稀疏注意力（Longformer、BIGBIRD）、近似注意力（Linformer、Reformer、Performer）、无注意力架构（Mamba/SSM、RWKV、RetNet、Hyena），复杂度从O(T²d)降至O(Td)或O(T log T d)；
  (ii) **动态神经网络**（§3.2）——Mixture-of-Experts（Switch Transformer、GLaM、V-MoE）、Early Exiting（DeeBERT、PABEE、FREE）；
  (iii) **预训练算法**（§4.1）——训练数据缩减（去重、patch removal如MAE）、NAS（Zero-shot NAS、PASHA）、渐进式学习（StackingBERT、LiGO）、混合精度训练（Mesa、GACT）；
  (iv) **微调算法**（§4.2）——Additive Tuning（Adapter、Prompt Tuning、Prefix Tuning）、Selective Tuning（SAM、SmartFRZ）、Re-parameter Tuning（LoRA及其变体QLoRA、DoRA、PiSSA、LoRA+）；
  (v) **推理算法**（§4.3）——Speculative Decoding（2-3×加速）、KV Cache优化（H2O、FastGen、vLLM PagedAttention）、Prompt压缩（LLMLingua 20×压缩）、Long Context（StreamingLLM、LongNet）；
  (vi) **模型压缩**（§4.4）——剪枝（SparseGPT、Wanda、LLM-Pruner）、知识蒸馏（MiniLLM、GKD、Distilling Step-by-Step）、量化（GPTQ 3-4bit、AWQ、SmoothQuant、QuaRot）、低秩分解（TensorGPT 38.4×压缩）。
  论文使用flops-profiler工具（https://pypi.org/project/flops-profiler/）对GPT-2及Stable Diffusion 2.1进行FLOPs和存储开销分析（§2.1.3、§2.3.3）。

- 硬件平台是什么，配置是什么。
  论文为综述，未进行统一硬件实验。综合分析引用以下平台：NVIDIA A100/H100 GPU、TPU v4、消费级GPU、手机端（iPhone 12 CoreML、安卓NPU）、Raspberry Pi 5等。

- 模型是什么。数据集和bench分别是什么。
  综述覆盖模型：LLM（GPT-1/2/3/4、LLaMA-1/2、BERT、T5、PaLM）、ViT（ViT、DeiT、MAE、Swin Transformer）、扩散模型（Stable Diffusion 1/2）、多模态（CLIP、Flamingo、LLaVA、SAM）。
  Benchmark覆盖：GLUE、SuperGLUE、SQuAD、MMLU、HumanEval、ImageNet-1K/21K、COCO、ADE20K等。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  综述材料全部开源：https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey。该仓库包含LaTeX源码及参考文献BibTeX，但不包含可执行代码或benchmark脚本。以下以量化为例说明算法pipeline：

  量化（Quantization）算法pipeline——将FP32权重/激活转为低精度整数（§4.4.3）：
  ```
  // Weight-Only PTQ (以GPTQ为例)
  // 逐层量化，使用逆Hessian信息更新未量化权重
  for layer l in model.layers:
      W = layer.weight  // FP32, shape [d_out, d_in]
      H = inverse_hessian(W, calibration_data)  // [d_in, d_in]
      for i in range(d_in):
          // 量化第i列，使用H[i:,i:]补偿误差
          w_q[:,i] = round(W[:,i] / scale[i])  // INT4
          // 更新未量化权重以补偿量化误差
          W[:,i+1:] -= (w_q[:,i] - W[:,i]) * H[i,i+1:] / H[i,i]
  // 推理时dequantize: W_fp16 ≈ dequant(w_q) * scale
  ```

  以LoRA（§4.2.3）为例的低秩适应pipeline：
  ```
  // W_0 ∈ R^{d×k} 为预训练权重（冻结）
  // A ∈ R^{d×r}, B ∈ R^{r×k} 为可训练低秩矩阵，r << min(d,k)
  h = W_0 @ x + α/r * (B @ A) @ x
  // 训练时仅更新A, B；推理时W = W_0 + α/r * B @ A可fuse回原权重
  ```
