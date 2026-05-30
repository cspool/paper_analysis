## Hardware-Efficient Attention for Fast Decoding

- baseline方法是什么？
  Baseline 是现有的硬件高效注意力变体（GQA、MQA、MLA），它们在解码阶段面临算术强度不足和并行扩展受限的问题。全栈执行例子如下：
  - **算法层**：MQA 缓存单一 KV head 供所有 query head 共享，将 KV cache 降至最低但模型质量显著下降，且 TP 时每设备需复制该单头，内存节省无效。GQA 将 query head 分组共享 distinct KV head，通过适中的 group 数（如 gq=4）保持质量，但当 TP 度低时（如 TP=2），每设备仍需存放大量 KV cache（例如 GQA-8 TP=2 时每 token 存 2dh 元素）。MLA（DeepSeek V2/V3）通过低秩压缩将 KV 投影为单头 latent c^{KV}（维度 4d_h），解码时吸收 up-projection 矩阵，算术强度达 ~2h_q——是 MQA 的 2 倍。但 MLA 的致命缺陷是**单头 latent 在 TP 时在所有设备上复制**，TP=2 时 KV cache 不减半，TP=4 时仍是每设备 4d_h，并行扩展性受限。MLA 只能通过混合 TP+DP 来缓解——将不同 batch 序列分配给不同 DP group，但这在序列长度不均匀时产生严重的 straggler 效应（一个 DP rank 处理长序列阻塞所有其他 rank）。
  - **系统框架层**：基于 SGLang / vLLM 等 serving 框架，使用 chunked prefill (Agrawal et al., 2023) 和 PagedAttention (Kwon et al., 2023)。MLA 在 TP-only 配置下 latent 被复制，在 TP+DP 混合下 attention 子模块跨 DP group 复制。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-3 / FlashMLA kernel（Li, 2025）——使用 warp specialization、TMA、软件流水线优化 MLA 解码。但 MLA 的算术强度 ~2h_q 使 kernel 在 L_q=1 时接近 compute bound（H100 上达 610 TFLOPS），在 L_q=2（推测解码）时超出 compute roof 变为 memory-bound。
  - **硬件架构层**：NVIDIA H100 80GB SXM5 GPU（989 TFLOPS BF16, 3350 GB/s HBM）。HBM 带宽与计算能力的差距持续扩大（FLOPs ~3×/2yr vs 带宽 ~1.6×/2yr），使 memory-bound decoding 问题日益严重。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过"以算术强度为设计视角"重新设计注意力，提出 GTA 和 GLA 两个变体，辅以低层次 kernel 优化，解决 basline 的三个核心缺陷：

  **缺陷 1: GQA 的 KV cache 在低 TP 度时仍然过大 → GTA 通过 KV tying 解决**
  GTA 将 GQA 中独立的 Key 和 Value 投影合并为单一的 *tied KV* 状态（m_kv=1 vs GQA 的 m_kv=2），基于以下洞察：(a) key 在加 RoPE 前位于极低秩子空间（Yu et al., 2024a），(b) 仅部分 head 维度需要 RoPE 用于位置区分（Black et al., 2022; Barbero et al., 2025）。GTA 将 tied KV 的前半维度用作未旋转的 key（K_NoPE），另加单头 RoPE 投影广播到所有 group 作为 key 的旋转部分。这使得每 token KV cache 从 hkv×2×d_h 降至 hkv×1.5×d_h（含广播的 RoPE 部分 0.5×d_h），算术强度从 gq 翻倍至 2gq。关键设计选择：对 tied 部分加 RoPE 再反旋转会损害质量，因此 tied 部分永不旋转。

  **缺陷 2: MLA 的单头 latent 无法分片到 TP rank → GLA 通过多 latent head 分组解决**
  GLA 将 MLA 的单头 latent c^{KV} (d_c=4d_h) 拆分为 h_c 个 latent head（每 head d_c=2d_h），每个 latent head 负责一组 query head。这一改动的关键后果是：(a) latent head 可以在 TP rank 间分片而无需复制——TP=2 时每设备仅存 2d_h（vs MLA 的 4d_h），TP=4 时每设备存 0.5×d_h/head；(b) 不牺牲算术强度——每 group 内的算术强度仍为 ~2gq（与 MLA 的 2h_q 可比）；(c) 保留模型质量——通过 per-group 的 up-projection 矩阵为每组 query head 学习专属的 K/V 特征。训练时每组有独立的 W^{UK}_i 和 W^{UV}_i；解码时这些矩阵被吸收进 Q/O 投影。

  **缺陷 3: 解码 kernel 未充分利用现代 GPU → 异步流水线 + 分布式偏移量计算 + Cooperative Softmax**
  论文将算法创新与低层次系统优化配对：(a) Warp Specialization 将 producer（内存加载）和 consumer（Tensor Core MMA）分配到不同 warp，利用 GPU warp scheduler 的异步性重叠执行；(b) Distributed Offset Calculation 解决 Paged KV 下 cp.async 指令的地址计算瓶颈——通过 warp 内多线程协作分摊 64-bit 整数地址计算，使 page size 1 的 kernel 速度匹配 page size 64（解锁 RadixAttention prefix caching）；(c) Cooperative Softmax 支持多 warp 并行下的正确 online softmax。

  全栈执行例子（GLA-2, TP=2, H100 serving）：
  - **算法层**：X → Q_{0,1} 投影 → c_{0,1}^{KV} latent heads（各 2d_h，分片到 rank 0 和 1）→ 吸收后 attention: O_i = softmax(Q_i @ c_i^{KV}^T) @ c_i^{KV} → partial output @ W^{VO}_i → AllReduce → 最终 O。每 token KV cache per device: 2d_h（vs MLA 的 4d_h）。
  - **系统框架层**：基于 SGLang live server 模式（含 HTTP 解析、动态队列、GPU kernel 调用的全程计时），使用 FlashAttention-3 kernel + chunked prefill (tile=8192)。GLA-8 (h_c=8, TP=8) 在 64 并发 8K/4K prefill/decode 下达到 1461 tok/s（vs MLA TP=8 的 859 tok/s，+70%）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：CUDA/PTX kernel：warp specialization（producer TMA/cp.async + consumer mma），distributed offset calculation（128 threads 分 8 组协作计算 paged KV 地址），cooperative softmax（sTMP cross-warp reduction）。GLA kernel L_q=2 时 2× faster than FlashMLA。
  - **硬件架构层**：NVIDIA H100 80GB SXM5。GLA kernel 达 93% peak memory bandwidth、70% peak TFLOPS。因 GLA 每 device KV cache 小，在 131K prefill + 不平衡负载下纯 TP=8 的 GLA-8 完成 2.7× MLA 混合 TP+DP 的吞吐。

  对比 baseline 的关键差异：
  - GTA vs GQA：每 token KV cache 减半、算术强度翻倍，同规模模型上 PPL 更低（XL: 10.129 vs 10.202）
  - GLA vs MLA：每 device KV cache 减半（TP≥2）、算术强度相当（~2gq vs ~2h_q），可纯 TP 部署（无需 DP）、对不平衡负载鲁棒（无 DP straggler），质量匹配或略优（XL downstream: 60.0% vs 59.1%）
  - GLA vs FlashMLA kernel：标准解码快 20%、推测解码快 2×，page size 1 无减速（解锁 prefix caching）
  - 设计哲学：算术强度最大化 = 多为每加载的字节做计算 = 更好利用现代 GPU 的算力过剩——通过减少 m_kv（tying KV→1）、增加 gq（分组）、引入 h_c（多 latent head 可分片）三个独立维度实现
