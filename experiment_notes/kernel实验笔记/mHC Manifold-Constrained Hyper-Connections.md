## mHC Manifold-Constrained Hyper-Connections

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 mHC 设计了一套定制化融合 kernel 系统来消除 HC 引入的显存带宽瓶颈，使 n=4 时额外时间开销仅 6.7%。
    - **Kernel Fusion 策略**：
      1. **RMSNorm 重排序优化**：将 RMSNorm 的除以 norm 操作重排到矩阵乘法之后，减少高维 $\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$ 上的延迟。
      2. **融合核 1（Eq.14-15）**：将两次对 $\vec{x}_l$ 的扫描（一次用于 RMSNorm r 的计算，一次用于线性投影 $\vec{x}_l \varphi_l$）融合为单一 kernel，利用矩阵乘法单元最大化显存带宽利用率。前向单核完成；反向两个矩阵乘法同样融合为单核，避免重复加载 $\vec{x}_l$。前后向 kernel 均包含精细的流水线（load, cast, compute, store）处理混合精度。
      3. **融合核 2（Eq.16-18）**：将小系数上的轻量操作（RMSNorm 归一化 + gating factor 乘法 + bias 加法 + Sigmoid 激活）机会主义融合为单一 kernel，显著减少 kernel launch 开销。
      4. **融合核 3（Eq.19）**：Sinkhorn-Knopp 迭代（20 次交替行列归一化）实现在单一 kernel 内。反向 pass 实现自定义反向 kernel，在片上重新计算中间结果并遍历整个迭代过程。
      5. **映射应用融合核**：将 $\mathcal{H}_l^{\text{post}}$ 和 $\mathcal{H}_l^{\text{res}}$ 的应用与 residual merging 融合——将读取元素从 $(3n+1)C$ 降至 $(n+1)C$，写入元素从 $3nC$ 降至 $nC$。
    - **Recomputing 策略**：丢弃 mHC 所有中间激活的前向结果，在反向 pass 中重新执行 mHC kernel（不含沉重的 $\mathcal{F}$ 层计算）。对于 $L_r$ 连续层，仅需持久化首层输入 $\mathbf{x}_{l_0}$。最优块大小 $L_r^* \approx \sqrt{nL/(n+2)}$ 与 pipeline stage 中的层数对齐。
    - **DualPipe 通信重叠**：扩展 DualPipe schedule（DeepSeek-V3）来重叠 mHC 在 pipeline stage 边界的通信和计算。MLP 层的 $\mathcal{F}_{post,res}$ kernel 在专用高优先级 compute stream 上执行，避免阻塞通信流；attention 层避免 persistent kernel 防止长时间 stall。
  - 实验比较：
    - 系统开销度量：mHC (n=4) 仅引入 6.7% 的额外训练时间。
    - 对比 Baseline（标准残差连接）和 HC 的 I/O 开销分析（Tab. 2）：HC 将显存访问从 ~3C 增加到约 $(8n+2)C$，mHC 通过融合 kernel 缓解。

- 后端平台是什么，配置是什么。
  - 论文未明确说明 GPU 型号和规模。利用 TileLang 框架实现 kernel，涉及 bfloat16/tfloat32/float32 混合精度计算。
  - 训练通信涉及 NVLink（节点内）+ NIC（节点间），使用 DualPipe pipeline parallelism。

- 评估性能的软件/脚本是什么。修改了什么。
  - **TileLang** (Wang et al., 2025)：用于实现大部分 mHC kernel（除 Eq.14-15 的矩阵乘法融合核），TileLang 简化了复杂计算过程的 kernel 实现，以最小工程代价充分利用显存带宽。
  - **DualPipe** (DeepSeek-V3)：在 DualPipe schedule 基础上扩展，在 pipeline stage 边界增加通信-计算重叠。
  - **Kernel 修改详情**：
    - 新增 3 个专门的 mHC 计算 kernel（计算 $\mathcal{H}_l^{\text{pre}}, \mathcal{H}_l^{\text{post}}, \mathcal{H}_l^{\text{res}}$）
    - 新增 2 个映射应用 kernel（$\mathcal{F}_{pre}$ 和 $\mathcal{F}_{post,res}$）
    - Sinkhorn-Knopp 前向+自定义反向单 kernel 实现

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明 mHC kernel 代码是否开源。TileLang (https://github.com/anthropics/tilelang) 是开源框架。
  - **mHC kernel 的完整执行流程**（以单层前向为例）：
    1. **输入数据**：$\mathbf{x}_l \in \mathbb{R}^{n \times C}$ (bfloat16)，weights $\varphi_l \in \mathbb{R}^{nC \times (n^2+2n)}$ (tfloat32)，bias $\mathbf{b}_l \in \mathbb{R}^{1 \times (n^2+2n)}$ (float32)，scalars $\alpha$ (float32)
    2. **Kernel 1 — 融合线性投影+Norm**：
       - flatten $\mathbf{x}_l \to \vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$
       - 单 kernel 完成：(a) 计算 $r = \|\vec{\mathbf{x}}_l\|_2 / \sqrt{nC}$；(b) 计算 $\vec{\mathbf{x}}_l \varphi_l$ 得到 $[\tilde{H}^{\text{pre}}, \tilde{H}^{\text{post}}, \tilde{H}^{\text{res}}]$
       - 混合精度流水线：load (bfloat16) → cast (float32) → compute (tfloat32 MMA) → store (float32)
       - 评估原理：此 kernel 将两份独立的内存扫描合并为一份，消除重复从 HBM 读取 $\vec{\mathbf{x}}_l$ 的开销
    3. **Kernel 2 — 融合后处理**：
       - 输入：Kernel 1 的 float32 输出 + $\alpha$ scalars + bias
       - 计算：$1/r \cdot [\alpha^{\text{pre}}\tilde{H}^{\text{pre}}, \alpha^{\text{post}}\tilde{H}^{\text{post}}, \alpha^{\text{res}}\tilde{H}^{\text{res}}] + \mathbf{b}_l$
       - 随后：$\sigma(\cdot)$ 和 $2\sigma(\cdot)$ 应用于 pre/post 分支
       - 评估原理：全部小系数上的逐元素操作融合为单 kernel launch，避免数十次微 kernel 的 launch 开销
    4. **Kernel 3 — Sinkhorn-Knopp 迭代**：
       - 输入：$\tilde{H}^{\text{res}}$
       - 计算：$\mathbf{M}^{(0)} = \exp(\tilde{H}^{\text{res}})$，然后 20 次交替行列归一化
       - 评估原理：完整迭代在单 kernel 内执行，避免多次 kernel launch 和中间结果的 HBM 读写
    5. **Kernel 4 — Pre 映射应用**：
       - 计算：$\mathcal{F}_{pre} = H^{\text{pre}} \cdot \mathbf{x}_l \in \mathbb{R}^{1 \times C}$（n-stream → 1-stream 聚合）
    6. **层计算**：标准 attention/FFN $\mathcal{F}(\mathcal{F}_{pre}, W_l) \in \mathbb{R}^{1 \times C}$
    7. **Kernel 5 — Post+Res 融合映射应用**：
       - 计算：$\mathbf{x}_{l+1} = H^{\text{res}} \mathbf{x}_l + H^{\text{post}^\top} \cdot \mathcal{F}(\cdot)$
       - 融合 residual merging：读取量从 $(3n+1)C$ 优化为 $(n+1)C$，写入量从 $3nC$ 优化为 $nC$
    8. **反向 pass**：
       - 重新执行 Kernel 1-5（不含层函数 $\mathcal{F}$）计算梯度所需中间激活
       - Sinkhorn-Knopp 反向为定制 kernel，在片上重计算中间结果
  - **性能对比**：
    - Baseline（标准残差连接）I/O: 读 2C, 写 C
    - HC（无融合 kernel）I/O: 读 $(5n+1)C + n^2 + 2n$, 写 $(3n+1)C + n^2 + 2n$
    - mHC（融合 kernel 后）: 额外训练时间开销仅 6.7%（n=4）
