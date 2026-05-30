## EfficientQAT Efficient Quantization-Aware Training for Large Language Models

- **baseline方法是什么？**
  Baseline有两类：(1) PTQ方法（GPTQ、AWQ、OmniQuant、AutoRound）采用block-wise reconstruction，但仅训练少量量化参数（rounding参数、clipping阈值或步长），限制优化空间且忽略跨block交互，低比特下精度损失严重；(2) Q-PEFT方法（PEQA、QA-LoRA、QLoRA）冻结量化权重，仅训练少量连续浮点参数或LoRA adapter，低比特下无法充分恢复量化信息损失。

  Baseline全栈执行例子（GPTQ, Llama-2-7B W2G128）：
  - 算法pipeline：加载FP16预训练Llama-2-7B → 逐block执行贪心OBQ量化：对每行权重，逐列贪心选择量化误差最小的列进行量化+补偿剩余权重 → Hessian矩阵用于计算补偿量 → 仅优化rounding决策（量化后向上/向下取整），所有权重值不参与梯度优化。无端到端训练、无跨block交互。W2G128 C4 PPL=不可用（退化严重），5-task Avg Acc≈41.56%。
  - 系统框架：PyTorch + HuggingFace Transformers。GPU推理。量化模型通过标准INT2 packing存储，推理时解包为FP16执行矩阵乘法。
  - 编译框架：论文未明确说明（标准PyTorch eager mode推理，无自定义编译器）。
  - kernel调度：依赖标准cuBLAS FP16 GEMM kernel，量化权重在kernel调用前解量化为FP16。无定制化INT2/INT3 kernel优化。
  - 硬件架构：NVIDIA A100-80GB GPU，Tensor Core仅加速FP16/INT8运算，INT2/INT3无硬件原生支持。

- **论文方法是什么？如何对应解决Baseline的缺陷？**
  论文提出**EfficientQAT**两阶段QAT框架，通过Block-AP和E2E-QP分别解决PTQ和Q-PEFT各自的缺陷：

  **(1) Block-AP解决PTQ优化空间受限（对应PTQ缺陷）**：
  - PTQ（GPTQ/OmniQuant/AutoRound）仅训练rounding参数或clipping阈值，每次权重更新被限制在(-1,+1)区间内作为正则化防止过拟合，但大幅缩小解空间 → Block-AP是首个在block-wise reconstruction中直接训练所有权重和量化参数的方案（W, s, z全训练），无需额外引入rounding参数设计。
  - 实验证明（Table 5）：Block-AP (s,z,W全训练) PPL=8.53 vs 仅训练rounding PPL=15.50 vs 仅训练clipping PPL=11.28。且全训练内存(8.5GB)低于rounding训练(8.6GB)，因后者需额外保存rounding参数副本。
  - 训练数据和epoch数：Block-AP仅需4096样本、2 epoch即可收敛，验证损失与训练损失差距从1.07缩至0.06（Figure 3）。

  **(2) E2E-QP解决跨block交互缺失（对应PTQ缺陷）**：
  - Block-AP虽恢复了block内精度，但各block独立训练忽略跨block交互 → E2E-QP冻结Block-AP产出的量化权重W_q，仅端到端训练步长s（每个group的scale factor）。步长s参数占比约1.6%（g=64），使端到端训练内存极低（Llama-2-70B W2G64仅需34.2GB）。
  - E2E-QP中无需量化操作（Eq.1前向不执行），仅执行反量化W_hat = (W_q - z) * s → 梯度仅需计算∂W_hat/∂s = W_q - z，计算图简单高效。

  **(3) Block-AP+E2E-QP组合解决Q-PEFT精度不足（对应Q-PEFT缺陷）**：
  - Q-PEFT（PEQA等）使用RTN初始化量化权重后仅训练步长，无法恢复低比特(2/3-bit)的严重信息损失 → EfficientQAT先用Block-AP提供高质量初始化，再用E2E-QP跨block微调。
  - Table 4消融：仅RTN（即无Block-AP无E2E-QP）Avg PPL=453.49；Block-AP单独降至8.53；E2E-QP单独降至9.33；组合(BP+E2E-QP)降至7.68，Avg Acc从40.69→58.99→55.71→60.14。

  论文方法全栈执行例子（EfficientQAT, Llama-2-7B W2G64）：
  - 算法pipeline：加载FP16预训练Llama-2-7B → **Block-AP**：逐block将线性层权重W量化为W_int=clamp(round(W/s)+z,0,3)，反量化W_hat=(W_int-z)*s → 前向用W_hat计算block输出 → MSE损失对齐FP16 block输出 → STE反向传播同步更新W(梯度截断)、s(Eq.3梯度)、z(Eq.4梯度)，epoch=2，lr_W=2e-5，lr_s=1e-4 → 输出W_q(N-bit)、s(FP16)、z(N-bit) → **E2E-QP**：冻结W_q和z，仅训练s → 全模型前向（反量化W_hat=(W_q-z)*s，无量化前向）→ 语言模型cross-entropy损失 → 仅s反向更新，lr=2e-5，epoch=1，ctx=4096 → 最终模型W_q(N-bit)+s(FP16)+z(N-bit)存储，推理时反量化为FP16执行。W2G64 C4 PPL=8.50，5-task Avg Acc=60.14%（vs FP16=64.85%）。
  - 系统框架：PyTorch + HuggingFace Transformers。单A100-80GB GPU训练（70B 2-bit仅需34.2GB E2E-QP内存）。量化模型兼容MLC-LLM、AWQ、BitBLAS、Marlin、T-MAC等推理框架。
  - 编译框架：论文未明确说明（标准PyTorch eager mode，无自定义编译框架）。
  - kernel调度：使用BitBLAS在A100-80GB上评估INT2矩阵向量乘法加速（2.9x-4.4x vs FP16）。量化kernel原理：INT2权重packing存储 → kernel内SIMD解包 → 低精度整数MAC → 乘步长s反量化 → FP16累加输出。
  - 硬件架构：NVIDIA A100-80GB GPU。标准CUDA Core执行低精度整数运算，Tensor Core可通过BitBLAS等工具映射到INT8硬件通路。
