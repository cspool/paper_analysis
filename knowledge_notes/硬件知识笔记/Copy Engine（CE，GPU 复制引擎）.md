## Copy Engine（CE，GPU 复制引擎）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Copy Engine（CE）是 GPU 上的架构级 DMA 引擎（architectural engine），负责发起 CPU-GPU 间的 DMA 数据搬运：驱动把 copy 命令写入 push buffer（pushbuffer），CE 按命令执行 PCIe DMA 传输，传输与 SM 计算异步流水。在 CC-capable GPU 中，CE 增强集成硬件 AES 引擎，可对收到的密文做流水解密（异步、可与后续传输重叠）。GPU 上同类的安全相关引擎还有 GSP（GPU System Processor）与 SEC2（Secure Processor），均为集成硬件 AES 的 RISC-V 微控制器（NVIDIA 专利披露片上 fuse 存安全密钥）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
LÆGIS 中 CE 的运转（UVM 页迁移）：CPU driver 加密 4 KB 页后，注入 CE copy 命令到 push channel（gpu->parent->ce->decrypt(push, gpu_dst, va_bb, PAGE_SIZE, va_tag)）→ CE 经 PCIe DMA 把密文页 + 128-bit（MAC||ID）传到 GPU → CE 用硬件 AES 引擎解密（构造 128-bit 输入 = 96-bit IV + 17-bit 块索引 + padding 生成 OTP 异或）→ 明文写 HBM。解密可异步流水、GPU 侧解密延迟可被隐藏（baseline 已如此）；LÆGIS 的 IV 访问路径：CE 先经片上 IV cache/共享 crossbar 从 HBM IV Bank 取 IBE，再进 AES 引擎。PCIe 大突发传输（预加密页批量提交 bounce buffer）提升 CE DMA 链路利用率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GPU 硬件 DMA 引擎（NVIDIA CE/CE3 等，公开研究参考：压缩 DMA 引擎 CDMA、模块化 DMA 架构等）；CC 版本集成 AES-GCM 引擎。使用：cudaMemcpy/UVM 迁移的物理搬运；LÆGIS 在 GPGPU-Sim 中把 CE/GMMU 路径上的 AES 引擎建模为流水 AES（解密延迟暴露为 GPU 侧组件，性能分解中单独列项）。论文未明确说明 CE 内部微架构细节。

涉及论文标题：
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
