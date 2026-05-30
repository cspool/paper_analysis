## ARB-LLM Alternating Refined Binarizations for Large Language Models

- baseline方法是什么？
  Baseline 是 **BiLLM**（ICML 2024），SOTA 二进制 PTQ 方法。BiLLM 的流程：(1) 用 Hessian 敏感度选出 salient columns，剩余的为 non-salient columns；(2) 对 salient weights 使用二阶 binarization（Ŵ = α₁B₁ + α₂B₂ + μ），non-salient weights 按 magnitude 分为 sparse/concentrated 两组，分别用一阶 binarization（Ŵ = αB + μ）；(3) 所有 binarization 参数（μ, α, B）通过标准的闭式解一次性确定（μ = mean(W), α = mean(|W-μ|), B = sign(W-μ)），不做迭代精炼。BiLLM 的缺陷：(i) 二值化参数一次性计算后不做精炼，导致二值化权重与全精度权重存在分布偏移（均值不对齐，见图 2）；(ii) calibration data 仅用于 Hessian 敏感度评估和 block-wise error compensation，未参与 binarization 参数的更新；(iii) 仅使用 row-wise scaling（α, μ），无法处理 LLM 权重中显著的列间偏差（某些列值远大于其他列，见图 3）；(iv) 仅对 non-salient weights 做 sparse/concentrated 分组，salient columns 的 group bitmap 区域未被利用（见图 5 左侧）。

  Baseline 全栈执行例子（LLaMA-7B, BiLLM, ~1.09-bit）：
  - 算法pipeline：FP16 权重 W → 逐层计算 Hessian 选出 salient columns → salient weights: 二阶 binarization Ŵ=α₁B₁+α₂B₂+μ，α₁,α₂,μ,B₁,B₂ 一次性闭式解 → non-salient weights: 分 sparse/concentrated 两组，分别一阶 binarization Ŵ=αB+μ，α,μ,B 一次性闭式解 → block-wise OBC 补偿 → 压缩权重存储（bitmap 记录分区，B 存 ±1，α,μ 存 FP16）。WikiText2 ppl: 49.79。
  - 系统框架：PyTorch + HuggingFace，单卡 A800-80GB，耗时 45 min。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用标准 PyTorch 推理，W 存储为 packed 1-bit + FP16 scaling factors，推理时解包并反量化后 FP16 GEMM）。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ARB-LLM 通过四个递进式创新解决 BiLLM 的缺陷：

  **(1) ARB（Alternating Refined Binarization）**：引入迭代精炼机制，每轮交替更新 μ → α → B 以逐步缩小量化误差。解决了 BiLLM 一次性计算导致的分布偏移问题。理论保证（Theorem 1）：每轮迭代后 L₁^τ ≤ L₁⁰，量化误差单调不增。仅 ARB 基础版本（无 CGB）就将 WikiText2 ppl 从 49.79 降至 22.67。

  **(2) ARB-X**：将 calibration data X 引入 binarization 参数更新，用 L₂ = ||WX - ŴX||² 替代 L₁ = ||W - Ŵ||² 作为优化目标。通过预计算 S = Σ X_b^T X_b 将高维 calibration tensor 压缩为 S ∈ R^{m×m}，理论加速 389×（Theorem 2）。解决 BiLLM 中 calibration data 仅用于 Hessian 评估而未参与参数优化的问题。WikiText2 ppl: 21.81。

  **(3) ARB-RC**：引入 column-wise scaling factor α^c，与 row-wise α^r 形成双轴缩放（Ŵ = α^r·α^c·B），同时移除 μ 以节省存储。解决 BiLLM 仅用 row-wise scaling 无法保留列间偏差的问题（图 3 右验证 ARB-RC 有效保留列偏差）。ARB-RC 在性能和压缩上双赢：WikiText2 ppl 14.03（ARB-X 的 21.81），同时存储从 2.93GB 降至 2.63GB（LLaMA-7B raw bitmap）。

  **(4) CGB（Column-Group Bitmap）**：将 salient columns 也按 magnitude 分为 sparse/concentrated groups，使 group bitmap 的 salient 区域不再浪费（G_s = 1_n C_s^T ⊙ G, G_ns = 1_n C_ns^T ⊙ G）。解决 BiLLM 中 group bitmap 在 salient columns 区域未被利用的空间浪费。CGB 进一步提升性能（ARB-RC + CGB: 14.03 vs ARB-RC w/o CGB: 15.85）。

  论文方法全栈执行例子（LLaMA-7B, ARB-LLM_RC = ARB-RC + CGB, ~1.09-bit, #Iter=15）：
  - 算法pipeline：FP16 权重 W → 逐层：① Hessian 评估选出 salient columns C_s，C_ns = ¬C_s → ② CGB 分区（salient-sparse / salient-concentrated / non-salient-sparse / non-salient-concentrated）→ ③ salient zones: 二阶 ARB-RC，每轮交替更新 α^r→α^c→B₁,B₂（15 轮，式 13 更新 α^r,α^c，式 8 binary search 更新 B₁,B₂），无 μ → ④ non-salient zones: 一阶 ARB-RC，每轮交替更新 α^r→α^c→B（15 轮）→ ⑤ block-wise OBC 补偿 → 输出：Ŵ = α^r·α^c·B（±α^r_i·α^c_j 加权二值矩阵）+ bitmap。WikiText2 ppl: 14.03，LLaMA-7B QA 平均准确率首次在 binary PTQ 中超越同尺寸 FP16 模型。
  - 系统框架：PyTorch + HuggingFace → 单卡 A800-80GB → 128 calib samples from C4 → binarization 耗时 76 min（#Iter=15, CGB）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。推理时权重以 1-bit packed format 存储，bitmap + α^r/α^c scaling factors 用于反量化。ARB-RC 比 BiLLM 少存储 μ，存储效率更优（LLaMA-7B: 2.83GB raw → 2.09GB CSR vs BiLLM: 2.93GB raw → 2.19GB CSR）。
  - 硬件架构：论文未明确说明。
