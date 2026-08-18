## 细粒度交错流水线（Fine-grained Interleaved Pipeline）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
细粒度交错流水线是本论文为缓解 MLP 推理 memory-bound 问题设计的流水调度优化：把 16×16 tile 细分为 subtile，将"当前 subtile 的光栅化"与"下一 subtile 的深度装载 + MLP 推理（算 F(d)）"在时间上重叠，从而隐藏 MLP 推理的深度访存/装载延迟。动机（roofline 分析）：光栅化每投影 GS 9 参数做 256×6 MAC（算术强度高、compute-bound），MLP 推理 1 深度参数仅 6 MAC（算术强度低约 30 倍、memory-bound）；若先对全部 GS 做 MLP 推理再逐 tile 光栅化（naive 流水），MLP 阶段 PE 利用率极低。细粒度是必须的：不同 tile 的 GS 数差异大（80~10000+）且 depth buffer 容量有限，整 tile 粒度无法稳定重叠。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件流水（本论文 V-B 章 + Fig.10 右下）：
```
对每个 subtile 按序：
  [装载] DRAM 传 GS 深度 d 到 depth buffer（4KB，capacity-limited 故需细粒度）
  [MLP]  depth buffer → PE 阵列 MLP 模式 → F(d) 写回 depth buffer
  [光栅化] PE 阵列 rasterization 模式用 F(d) 光栅化当前 subtile
           （同时 depth buffer 装载下一 subtile 的 d —— 重叠点）
```
效果：除第一个 subtile 外，深度访问延迟被完全隐藏；相对 naive 流水（先全量 MLP 后光栅化）显著提高 PE 利用率。整体流水 BS+AR+OIT+IP 吞吐 2.27×（几何均值，vs BS 1×/BS+AR 1.37×/BS+AR+OIT 2.16×）；对 edge GPU 排序的加速从 naive MLP-OIT 的 21~119× 提升到 300×+。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PE 阵列 + depth buffer（存深度与 F(d)）之间的双缓冲/流水控制；depth buffer 4KB、GS-feature 88KB、output 4KB 的片上存储布局（表 II）。一般意义：这是"计算-访存重叠（compute-memory overlap / pipeline overlap）"在专用加速器中的实现——与 GPU kernel 的 double buffering、软件流水线同思想，但由硬件流水控制器静态执行，粒度（subtile）由 depth buffer 容量与 tile 负载波动决定。论文未开源该流水控制 RTL（论文未明确说明细节，如 subtile 具体尺寸）。

涉及论文标题：
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance
