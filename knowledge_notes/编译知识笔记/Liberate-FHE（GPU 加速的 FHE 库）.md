## Liberate-FHE（GPU 加速的 FHE 库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Liberate.FHE（Desilo 团队，https://github.com/Desilo/liberate-fhe）是一个开源 FHE 库，口号"bridging the gap between theory and practice"，聚焦性能与精度：纯 Python + CUDA 实现、原生支持多 GPU、RNS-CKKS 方案、BSD-3-Clause-Clear 许可，设计目标是让非密码学背景的开发者也能用（最小依赖、易与 AI 框架集成）。现已弃用，继任者为 DESILO FHE 库（https://fhe.desilo.dev/，含 bootstrap 等更多功能）。
- 本论文角色：FEnc² 的 GPU 后端 HE 执行框架（Evaluation Methodology）：所有 HE 原语（编码、加密、Add/PMult/CMult、旋转、keyswitch、NTT）跑在 Liberate-FHE 上，FEnc² 只在其上做布局/打包层面的编排；CPU 端实验用 CPU 实现（Liberate-FHE 也支持纯 Python 运行）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 使用流程（README quick start + 本论文用法）：配置参数（presets 按安全等级选 logN/logQ，本论文固定 scale Δ=2^40、λ≥128 bits）→ 建 engine（ckks_engine）→ 生成 sk/pk/evk（旋转/重线性化密钥）→ 客户端编码加密输入 → 服务端按 FEnc² 生成的密文布局调用 Add/PMult/Rot/Rescale 原语序列执行 CNN 电路 → 解密。本论文在框架层之上额外统计了各原语调用次数（Table V：Rot/keyswitch/NTT/iNTT/ct count/Mult）与系统级指标（Table VI：kernel 调用、GPU 内存传输大小与次数），证明布局优化压缩了暴露给库层/硬件的原语数量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CUDA kernel 实现多项式算术（NTT/iNTT、pointwise 乘、旋转 shuffle、keyswitch），多 GPU 用 cuBLAS/cuFFT 类后端做并行；Python API 屏蔽细节。使用场景：GPU 上的 CKKS 加密推理/科学计算原型；本论文用它支撑 LeNet/VGG5/SqueezeNet/ResNet18/MobileNet 的 GPU 端到端评测与 Hecate 式算子统计。注意：仓库已标记 deprecated，新项目建议用 DESILO FHE（fhe.desilo.dev）。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
