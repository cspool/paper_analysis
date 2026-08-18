## UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference

- baseline方法是什么？
  - baseline 是两类现有 LLM 量化加速器：(1) 固定精度加速器（如 BitMoD、AxCore、FIGNA、FIGLUT）——围绕定宽数据通路构建，只支持少数预设格式（如 AxCore 的 FPMA 定宽 W4A16），对单一精度效率高，但无法适配 LLM 多样化的量化格式需求（不同模型/不同使用场景需要不同位宽：Llama-2/OPT 可 4-bit、Llama-3/Qwen3 需 8-bit、同一模型还有低精度 draft 模式 vs 高精度 quality 模式）。(2) bit-composable 可组合乘法器加速器（Bit-Fusion 启发的 OliVe、Tender、M-ANT）——用小计算单元融合模拟更宽位宽，但乘法器成本 O(n²)（部分积生成与累加随位宽平方增长），吞吐随精度按 1/n² 塌缩：W4A4 高效，切到 W8A8/W16A16 时性能急剧退化（"dynamic tax"）。此外 FPMA 类设计（AxCore）虽用整数加法替代浮点乘法，但固定 datapath 不可融合/分解，且其 subnormal 重映射是近似转换引入额外噪声、低比特（FP4/FP3 中 25–50% 数值是 subnormal）下 FPMA 对数近似误差大导致精度崩坏。
  - baseline 全栈执行例子（AxCore 型 FPMA 加速器执行 Llama-3-8B W4A16 推理）：
    ```
    算法pipeline层：权重 W4A16 量化（权重 4-bit FP4、激活 16-bit），subnormal 用近似重映射处理；
               FPMA 乘法 r≈X+Y−B（整数加法），但 4-bit 权重 subnormal 重映射引入额外噪声
    系统框架层：论文未明确说明（AxCore/UNICORE baseline 是硬件加速器评估，不含 serving 调度）
    编译框架层：论文未明确说明（量化权重离线生成，无编译框架修改）
    kernel调度层：论文未明确说明（硬件模拟器中按固定 4/16 数据通路执行 GEMM，无多 kernel 运行时调度；
               激活保持 16-bit 不量化，无在线激活量化 kernel）
    硬件架构层：固定 W4A16 FPMA datapath——权重经近似 subnormal 重映射后驻留 PE，激活 16-bit 直接参与；
               乘法被整数加法替代但 datapath 定宽；换成 W4A4 或 W8A8 需要另一套硬件（复制 datapath），
               或切换到 bit-composable 乘法器设计（OliVe/Tender/M-ANT）——此时乘法器 O(n²) 部分积成本
               使 W8A8 吞吐相对 W4A4 按 1/4 塌缩、W16A16 按 1/16 塌缩
    ```
- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法 UNICORE 通过"加法型可组合原语 + 精度保持管线 + 分布自适应格式"的软硬件协同设计解决上述痛点：① S-FPMA——把 FPMA 分解为统一 4-bit 加法 slice，经进位链级联融合成 W8A8/W16A16，硬件成本随位宽严格线性 O(n)（每加一位只多一个 slice），直接消除乘法器型 bit-composable 的 O(n²) 动态税，同一 datapath 无需复制即可支持 W4A4/W4A8/W8A8/W16A16 及任意融合位宽；② 精度保持——(a) 位宽扩展：把低比特操作数无损转换到内部 E3M2 格式（尾数左移归一化 + 指数补偿），消除全部 subnormal，使 FPMA 对数近似假设成立（对比 AxCore 的近似重映射，是精确变换）；(b) 双路径补偿：细粒度 FG 补偿（LUT 按 (MA,MW) 预存残差 bit-pattern 拼接到结果 LSB 侧）+ 粗粒度 CG 补偿（1-bit 进位注入），解决 CG 单独在 2-bit 尾数粒度下无法表达修正的问题，使 FP4 下 FPMA 结果与全精度乘法逐位一致（Table III：无 FG 时 FP4 PPL 崩坏到 1.1E+4~4.9E+6，加 FG 后 11.15 与原始 FP4 相同）；③ DynFP——逐 group 可配置低比特浮点格式（自适应 E/M 分配 + 负零重映射 Z + 空位插入 I-flag），离线贪心搜索（96 候选 → k=16 palette，约 2 分钟/7B 模型）与在线 crest factor κ 映射（K/V，<0.2% QKᵀ FLOPs）把量化精度推到极致（UNICORE-Q 在 4/4/16 各模型 PPL 最低、接近 FP16，zero-shot 平均准确率多数配置最优）。对应解决 baseline 缺陷：乘法器 O(n²) 塌缩 → 加法 slice O(n) 线性扩展；定宽不可适配 → 单一可融合 datapath 覆盖全部量化格式；FPMA 近似误差/低比特精度损失 → 精确 subnormal 归一化 + FG/CG 双路径补偿恢复逐位精度；静态格式不适应分布 → DynFP 逐 group 自适应格式 + 自动化搜索。
  - 论文方法全栈执行例子（UNICORE 执行 Llama-3-8B W4A4KV4 推理，一个 4-bit 权重 × 4-bit 激活的 GEMM tile 数据路径）：
    ```
    算法pipeline层：权重经离线贪心搜索选 DynFP4 格式（E1M2+Z+I 等），存储为 E1M2 编码 + 8-bit group scale + 4-bit 格式索引
               （有效 4.375 bits/权重，仅比 MXFP4 高 2.9%）；激活与 K/V 在线量化，K/V 用 crest factor κ 选格式
    系统框架层：论文未明确说明（UNICORE 为硬件加速器论文，不含 serving 调度/框架修改）
    编译框架层：论文未明确说明（DynFP 格式索引离线确定后随权重加载；无编译框架实现）
    kernel调度层：在线激活/KV 量化 kernel（max-abs + RMS 归约算 κ → 阈值查表选 E/M 布局 → 量化），
               arithmetic intensity 0.63→0.87 仍 memory-bound，与 GEMM 大部分重叠（prefill 时延占比 7.1%–20.7%、
               decode 0.3%–1.6%、对 L≥2K 序列 κ 计算 <0.2% QKᵀ FLOPs）；量化后张量以原始低比特格式存储/传输
    硬件架构层：Weight Buffer 流出 DynFP4 权重 → Unified Format Converter 1 cycle 解码（LUT 按格式索引+数值映射到内部
               E3M2，负零 mux 选 Z 值）并按 Group 驻留 PE 列；激活经 PreAdd 指数偏置校正 T=A−B 后沿列广播；
               每 S-FPMA slice 对 (W,T) 做整数加法 T+W → FG 补偿 C_fg 拼接到 LSB、CG 1-bit 进位注入 → 乘积转
               sign-magnitude 与传入部分和经专用加法器累加 → 双链部分和 W4A4 模式求和、W8A8 模式拼接 → Rescale
               用 group scale 反量化 → 全局 Accumulator 合并输出；decode 阶段 K/V cache 同样 4-bit 存储
               大幅降低 DRAM 流量（对比 OliVe 16-bit K/V），DDR4/HBM2 下 decode 加速 2.96×/2.00×（vs OliVe）
    ```
