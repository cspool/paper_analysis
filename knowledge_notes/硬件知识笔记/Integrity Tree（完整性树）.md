## Integrity Tree（完整性树）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
完整性树是计数器模式内存加密（如 Intel SGX MEE）用于认证/防重放与防篡改的元数据结构：每个 cache line 维护自己的 IV 与 MAC，MAC 的哈希层层汇聚成树（如 Merkle/Bonsai 树），根哈希保存在片内可信位置，验证时沿树比对以检测外部篡改（重放、翻转攻击）。代价是元数据开销与验证延迟：Intel SGX 为保护完整性树预留约 25% 的 enclave 内存作为元数据。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
传统 TEE 内存加密流程：写数据时按 (IV,CL) 生成 OTP 加密 + 算 MAC → MAC 与 IV 存储片外并汇入完整性树（每层哈希）→ 读数据时先查树根验证路径哈希、比对 MAC，通过后才解密使用。树越深验证延迟越大。LÆGIS 的对比论证（Table I）：GPU-based CC（UVM）下完整性树"不必要"——(a) 数据到 GPU 后 HBM 假设可信，MAC 验证后即可丢弃、无需防篡改；(b) UVM 按页（4 KB/2 MB）迁移而非 cache line；(c) HBM 明文存储。因此 LÆGIS 用 8 MB HBM IV Bank（显式 IV，无树）替代完整性树，恢复乱序加密灵活性而避开 25% 元数据开销；防重放由 per-VABlock 显式 IV（ID+RV）与 CTR 追踪完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：MEE（Memory Encryption Engine）内嵌于内存控制器，树节点哈希用 AES 或 SHA 类原语，根在片内；代表性实现：Intel SGX（约 25% 元数据）、Bonsai Merkle Tree（面积高效）、Morphable Counters、VAULT（SGX 分页优化）。使用：需要防物理篡改/重放的 TEE 场景；在"内存可信"假设下（如 GPU HBM）可省去，换取更低开销与更大灵活性——LÆGIS 的核心取舍点。

涉及论文标题：
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
