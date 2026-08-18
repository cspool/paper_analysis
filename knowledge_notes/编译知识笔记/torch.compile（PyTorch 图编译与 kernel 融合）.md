## torch.compile（PyTorch 图编译与 kernel 融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- torch.compile 是 PyTorch 2 引入的图编译栈（PyTorch 2 paper [6]，ASPLOS'24）：把 eager 模式的 Python 算子序列通过动态字节码变换（bytecode transformation，解析 Python 帧与 FX 图捕获）编译成优化后的计算图，再经后端（默认 Triton 代码生成，也可用 Inductor 的 CUDA/C++ 代码生成）自动生成/融合 kernel，实现算子融合、layout 优化、自动并行等。对用户是单行 API：model = torch.compile(model)。
- 在 LoKA 中的角色：评估与部署的基础设施——LoKA Mods 消融实验（Fig.13，batch 1024）在 torch.compile 开启下测量；LoKA Dispatch 的 wrapper 需与 torch.compile 协作集成新低精度 kernel 以获得最佳性能（论文 limitation 明确：引入新低精度 kernel 时可能需要人工干预以保证与 torch.compile 正确集成）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译流程：Python 源码 → Dynamo 字节码变换（捕获 FX graph，遇到不支持算子回退 eager，guard 缓存）→ FX graph → 后端选择（Inductor 默认）→ Triton 代码生成/图调度（算子融合 pass、layout 传播）→ 编译后 kernel 缓存 → 运行时按 guard 命中复用编译产物。
- LoKA 使用例子：含 LoKA Mods（No Bias/BlockNorm/Hard Swish）的 Wukong 线性层经 torch.compile 后，BlockNorm+Hard Swish+量化反量化被融合进 GEMM epilogue kernel（与手写融合 kernel 目标一致，但由编译框架自动完成）；LoKA Dispatch 的 custom autograd 包装层在编译图内作为可调用的 kernel 选择点，前向/反向分别编译。
- 与 Triton 关系：LoKA 的 epilogue 融合与 Triton 的 tl.dot + 尾处理融合同源（论文引用 Triton [65] 的 epilogue fusion 概念）；torch.compile 默认后端即 Triton 代码生成，故论文在 torch.compile 下测量 BlockNorm 融合收益。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch 2 内置（Dynamo + Inductor），默认 Triton backend；支持自定义 backend（如 torch.compile(model, backend="inductor")）。使用场景：训练与推理的图优化；LoKA 用它加速含归一化/激活融合的模型，配合 Dispatch 的包装层达最佳性能。局限：LoKA 指出新低精度 kernel 集成进 torch.compile 可能需要人工干预；训练中 Dispatch 的动态精度切换会触发重编译（故仅作分布大漂移安全阀）。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
