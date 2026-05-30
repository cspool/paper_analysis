## Attamba__Attending_To_Multi-Token_States

- baseline方法是什么？
  Baseline是标准Transformer（GPT-style，60M参数，8层8 heads，512 model dimension）。标准attention的全栈执行例子（生成一个token）：
  - 算法pipeline：输入token → embedding → 8层Transformer block（每层: RMSNorm → QKV线性投影 → causal self-attention: softmax(QK^T/√d)·V over所有历史token → residual → MLP → residual）→ LM head → logits。Attention需计算与所有历史token的QK内积，复杂度O(n²)，KV cache大小O(n)。
  - 系统框架：Meta Lingua（PyTorch LLM训练库）。单卡A6000 GPU训练。标准PyTorch autograd + Adam优化器。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用PyTorch标准attention kernel实现，如Flash Attention）。
  - 硬件架构：NVIDIA RTX A6000 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. Attention计算量O(n²)随序列长度平方增长，长序列下FLOPs和内存开销巨大
  2. KV-Cache随序列长度线性增长O(n)，autoregressive推理时内存瓶颈
  3. L² attention map的激活值占用大量显存（iso-activation条件下论文甚至无法找到等价的transformer设计，见公式11中(1-P)/P的负项）
  4. 现有KV-Cache压缩（如低秩分解Palu、ShadowKV）和稀疏attention（BigBird、LinFormer）等方法会牺牲attention表达能力

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Attamba，核心思想是用SSM block替换Transformer attention中的K/V投影矩阵，让SSM将连续P个token压缩为单个表示，然后attention仅在这些压缩表示上进行计算。Query投影保持不变以保证自回归训练的causality。配套设计包括cyclic chunk boundary（逐层偏移消除固定边界偏差）、leading tokens（保留对最近token的完整attention模拟sliding window）、pseudo-chunking（可选的仅替换投影不裁剪attention mask模式）。

  论文方法全栈执行例子（Attamba P=4, L=4推理时生成一个token）：
  - 算法pipeline：输入token → embedding → 8层Attamba block（每层: RMSNorm → Q = X·W_Q 标准Query投影 → 将KV序列分为P=4 token的chunks → SSM_K, SSM_V 在每个chunk上autoregressive扫描: h_t = A_t·h_{t-1} + B_t·x_t, k_t/v_t = C_t·h_t → 仅保存每个chunk的最后L=4个输出（即完整当前chunk的SSM输出）→ 用chunk attention mask: 仅attend已完成chunk的边界+当前chunk内causal → Softmax(Q·K_SSM^T/√d)·V_SSM → residual → MLP → residual）→ LM head → logits。Attention map从L×L缩减为L×(L/P+L)≈L²/4。KV-Cache仅保留chunk边界+L个leading token的K/V：n/4×e(每个chunk)+4×e(当前chunk L tokens) vs baseline n×e。每层不同chunk边界偏移量（cyclic: layer_idx%4），使SSM学习到位置鲁棒的压缩。
  - 系统框架：Meta Lingua + Mamba library（cu_seqlens处理变长chunk，无需padding）。开源：https://github.com/abdelfattah-lab/attamba（BSD-3-Clause）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。SSM扫描使用Mamba库的selective scan并行实现。
  - 硬件架构：NVIDIA RTX A6000 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（O(n²)计算）→ SSM压缩P个token为1个后，attention仅需对n/P个压缩表示计算attention，FLOPs从O(n²)降至O(n²/P)。同时论文提出Attamba-Linear方案：将序列分为固定数量P个chunk，无论序列多长，attention map大小恒定，实现O(n)复杂度（类似BigBird但保留可变chunk边界能力）。
  - 缺陷2（O(n) KV-Cache）→ 推理时仅缓存每个chunk的最终SSM输出K(p)[-1]和V(p)[-1]（而非全部P个token的KV），KV-Cache从2nE降至2(n/P+L)E（L为leading tokens数）。P=8时约8×压缩，P=128时128×压缩。
  - 缺陷3（L²激活值）→ Attamba的attention map大小为n×(n/P+L)，而非n×n。公式11推导显示(1-P)/P项为负，正是Attamba消除L² activation的结果，使得Transformer在同等激活值预算下无法匹敌Attamba。
  - 缺陷4（压缩损失attention表达能力）→ 与稀疏attention（BigBird）或低秩分解（LinFormer）不同，Attamba的SSM压缩是可学习的、data-dependent的。每个chunk内P个token通过SSM的selective mechanism（A_t, B_t, C_t依赖输入）被自适应压缩，而非简单丢弃或平均。实验表明：(a) Attamba在iso-KV+SWA条件下的困惑度远优于Transformer（图7）；(b) 随机chunk边界与均匀分块效果相当（图14），说明SSM压缩对chunk划分方式鲁棒；(c) cyclic chunking额外提升5%，不同层压缩不同的token组，增强模型对不同上下文模式的覆盖；(d) pseudo-chunking（仅替换投影，不裁剪attention）可略微优于标准Transformer（图16），说明SSM-based K/V投影本身比线性投影有更好的表示能力。
  - SSM state collapse（长序列信息丢失）→ 与纯SSM不同，Attamba的SSM仅需处理固定长度chunk（P个token），不会遇到state collapse问题。注意力在压缩chunk表示上进行，SSM不需要在任意长序列上维护state。
  - 固定chunk边界偏差 → cyclic chunking：第layer层从layer%P偏移开始分chunk，确保不同层处理不同的token分组模式，分布边界效应。实验表明cyclic比uniform/FAttn/FSSM均更优。
  - FFN不受益 → 论文诚实指出FFN层无优化（Query序列长度不变以保持自回归训练），但attention的FLOPs和KV-Cache已在主要开销上获得显著压缩。

  Baseline→Attamba的迁移路径（以P=4为例）：
  1. 保持所有非attention组件不变（embedding, MLP, LM head, RMSNorm）
  2. 将每个attention层中的W_K, W_V线性投影替换为SSM_K, SSM_V block（约4M参数，残差连接保留）
  3. 在训练时构造chunk attention mask（公式5），测试时仅缓存chunk边界（公式7）
  4. 逐层偏移chunk边界（cyclic: layer_idx % P）
  5. 可选：保留L个leading tokens的完整attention（模拟sliding window），增加可控的质量-效率trade-off
