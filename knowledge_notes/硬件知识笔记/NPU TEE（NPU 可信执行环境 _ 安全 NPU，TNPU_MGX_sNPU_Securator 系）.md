## NPU TEE（NPU 可信执行环境 / 安全 NPU，TNPU/MGX/sNPU/Securator 系）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NPU TEE 是把 CPU 式可信执行环境移植到 NPU 的研究方向：在片外内存接口部署加密/认证引擎，数据片外加密、片内解密成明文参与计算，依靠 enclave 隔离假设与 scratchpad 清除（tile/layer 级）防止残留明文泄漏。代表工作：TNPU（HPCA'22，无树完整性）、MGX（ISCA'22，近零开销内存保护，tile 版本号）、sNPU（ISCA'24，片上 TEE + 多租户）、Securator（HPCA'23）、Azure 的 Confidential AI 加速器（ATC'23）。IroKnight 的对比基线即采用 MGX 式设计（片外接口加密认证引擎 + 片上 tile 版本号）。核心局限：明文驻留片上存储（register/cache/buffer/SRAM），不保有加密所有权；且 NPU 是纯软件管理，驱动/运行时漏洞（Samsung/Qualcomm/Huawei/AMD NPU 已有大量 CVE）可在执行中窃取片上明文，scratchpad 清除无法防御执行期泄露，layer 级清除还违反 SLO。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - NPU TEE 的运转流程：主机/模型提供方经 attestation 与 NPU 建立信任 → 模型参数/用户数据在片外加密存储 → 数据读入 NPU 时经片外接口的加解密引擎解密为明文，载入 scratchpad/寄存器 → 明文在片上执行全部计算（GEMM、softmax 等）→ 结果写回片外时再加密。为防残留明文被下一个请求或攻击者读到，需 scratchpad 清除：tile 级（每 tile 用完清除，运行时开销 19.6%）或 layer 级（每层清除，5.1% 但违反 SLO）；不清除则 2.8% 但明文全程驻留。因为计算在明文上执行、片上存储处处是明文，微架构侧信道（缓存/寄存器/SRAM 攻击）与 NPU 软件漏洞都能直接取走数据——TEE 无法提供加密所有权。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：工业界已部署 CPU/GPU 版（Intel SGX、AMD SEV、NVIDIA H100/H200 Confidential Computing）；NPU 版停留在研究（TNPU/MGX/sNPU/Securator），sNPU 有开源实现（ipads 组）。使用场景：云端 LLM 推理的机密计算，保护用户 query 不被云运维读取。IroKnight 的对比结论：相同资源与加密方案下，NPU TEE（tile 清除）19.6% 运行时/5.4% 能量、无清除 2.8%/4.8%，IroKnight 3.3%/15%——IroKnight 略高能量但始终不暴露明文并防篡改，确立"存储全加密"相对"清除式"的新设计点。

涉及论文标题：
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving
