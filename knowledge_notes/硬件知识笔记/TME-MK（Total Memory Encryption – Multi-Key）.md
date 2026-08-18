## TME-MK（Total Memory Encryption – Multi-Key）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TME-MK 是 Intel 的内存加密技术（TME 的多密钥版本）：内存控制器内置的硬件引擎对所有访问 TDX 私密内存的数据做 AES-XTS 加解密，支持多把硬件生成的临时密钥（每个 TD/密钥域一把，密钥仅存在于硬件内），未授权访问私密内存返回全零。属于 counter-less encryption（CLE）的商用形态；对 CPU 而言，TME-MK 在内存接口处透明加解密，核心读到的已是明文。
MC-ORAM 中把 TME（Total Memory Encryption）作为通用机制：TME 引擎提供无界加密 DRAM（C = AES_XTS(addr_128, D_128)，tweak=物理地址、无 nonce 元数据），与 ORAM"服务器内存本需加密"的需求巧合匹配——ORAM 树/暂存直接放 TEE 加密内存，省去客户端额外重加密（冗余加密消除）。但 TME 的确定性产生密文侧信道，MC-ORAM 在硬件加密前把块处理为 (data XOR mask)||ctr 使密文每次访问变化，带宽仅 baseline 的 1.125×。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
LÆGIS 平台中 TME-MK 的运转：TDX 信任域私密内存 → GPU CC 的 UVM 页迁移时，driver 读 TD 页（触发 TME-MK AES-XTS 自动解密，硬件实现仅约 40 cycle 内存访问延迟）→ 明文页再被 CC 要求的 AES-GCM 软件加密 → 经 PCIe 传 GPU。LÆGIS 的 X-LÆGIS 变体（硬件加速）发现 TME-MK 引擎在 GPU kernel 执行期间大多空闲，且其 AES-XTS 与 AES-GCM 运行在同一个 Galois 域，经重配置（不新增硬件）即可作为 CPU 侧 AES-GCM 引擎：40-cycle fill 延迟、16 B/cycle 吞吐——X-pIFN-LÆGIS 据此达平均 3.57×（最大 6.82×）加速。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Intel 处理器内存控制器内嵌（TME/TME-MT/TDX 需 BIOS 使能），AES-XTS256、per-key 加密（多密钥版本支持 64+ 密钥域）；Web 证据：Intel 白皮书确认 TME-MK 基于 AES-XTS256、硬件生成临时密钥（https://cdrdv2.intel.com/v1/dl/getContent/917058）。使用：作为 CPU TEE 私密内存的默认保护层，对软件透明；研究界可探索复用其空闲算力做 CC 数据面加密（LÆGIS 的 X 变体思路）。

涉及论文标题：
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
