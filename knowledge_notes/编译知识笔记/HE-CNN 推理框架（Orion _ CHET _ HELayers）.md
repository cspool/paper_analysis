## HE-CNN 推理框架（Orion / CHET / HELayers）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- HE-CNN 推理框架是把预训练 CNN 模型自动编译/转换为 FHE（主要是 CKKS）密文电路的软件栈，负责打包布局选择、旋转方案（BSGS）、噪声/level 管理、bootstrapping 放置与运行期执行。本论文涉及三个代表性框架：
  - CHET（Dathathri 等，PLDI'19）：面向 FHE NN 推理的优化编译器，用 Toeplitz 变换 + row-major 单通道打包做卷积，自动选旋转策略。
  - HELayers（IBM，PETs'23）：tile tensors 框架，用 block tiling + 多图（multi-image）打包填槽，支持 Spatial Packing 等布局，面向大网络。
  - Orion（Ebel, Garimella, Reagen，ASPLOS'25 最佳论文）：当前 SOTA，PyTorch 集成（torch.nn.Module），单次多路复用打包（single-shot multiplexed packing，任意 stride、level 消耗 2→1）、自动 bootstrap 放置（最短路求解）、自动参数选择；首个跑通 ImageNet ResNet-50（~8.98h、351 次自举）与高分辨率目标检测 YOLO-v1；基于 CKKS，ReLU 用高次 Chebyshev 多项式近似（约 14 level）。代码开源。
- 本论文把这三个框架（以及 Hyena+/Batchwise+/Fhelipe 等）作为 baseline：指出它们的打包布局是静态/启发式、跨层碎片化、只优化单侧旋转，并证明 FEnc² 布局包含它们为特例（S=1 等价 row-major/Orion，S=M 等价 pixel-wise）且在旋转复杂度（Table III）与实测（Fig.6、Table V）上全面占优。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Orion 式框架的编译-运行流程（本论文作为 baseline 的执行例子）：
```
1) 加载预训练模型 (torch.nn.Module) → 构建计算图
2) 布局决策：单次多路复用打包（密文打包布局 + stride 处理）
3) 图优化：BSGS 拆卷积循环、选择旋转策略、最短路求 bootstrap 放置
4) 参数选择：加密参数（logN/logQ/scale）自动选取
5) 客户端编码加密 → 服务端执行 HE 电路（旋转/keyswitch/NTT 主导，~70% 延迟）
6) 解密取结果
```
- Annotations：FEnc² 与之的差异在"布局决策"一步——Orion 用静态手工/多路复用布局，跨层密度不保持（ImageNet 大 feature map 下单通道即饱和密文容量、退化为 CHET 布局）；FEnc² 用解析凸模型选块大小 + AAC 跨层保密度，无需运行期 profiling，且不改模型/加密参数/硬件。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：C++/Python 库 + 图编译器（CHET 是专用编译器；HELayers 是 tile tensor 运行时；Orion 基于 PyTorch + SEAL/自定义底层）。使用：加载开源预训练模型、指定输入形状与 batch，框架自动生成 FHE 程序与参数。使用场景：隐私推理服务（MLaaS）、加密视觉任务；是 FEnc² 评估中"比较目标"的直接载体（表 VIII CPU 端 Orion vs FEnc²：LeNet 0.18s vs 40.87s、MobileNet 328s vs 3094s）。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
