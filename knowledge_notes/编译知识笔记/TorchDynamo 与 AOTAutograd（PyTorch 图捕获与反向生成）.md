## TorchDynamo 与 AOTAutograd（PyTorch 图捕获与反向生成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TorchDynamo 是 PyTorch 的图捕获前端：在 Python 字节码（frame）层面用 FX 追踪（PEP 523 frame evaluation API）拦截执行，把 eager 执行的算子序列捕获为 FX Graph（支持控制流/不支持的 op 落回 eager，即 guards 驱动的图拼接），是 torch.compile 的默认前端。AOTAutograd 在 Dynamo 捕获的前向图基础上自动生成反向图（autograd.Function 图 + 可微分版本），并把前/反向编译为一个可整体优化的图（配合编译后端），同时处理激活保存（saved tensors）与梯度累积。MTIA 300（ISCA'26）把两者作为其自定义 PyTorch backend 的前端：TorchDynamo 追踪前向图、AOTAutograd 生成反向图，再经 MTIA 优化算子分解与 TorchInductor 融合/代码生成。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
MTIA 300 编译流程（DLRM 训练一次迭代）：
```python
# 1. TorchDynamo: 捕获用户模型前向（TorchRec DLRM）为 FX Graph（frame 级追踪）
fwd_graph = torchdynamo.capture(model.forward, batch)
# 2. AOTAutograd: 从 fwd_graph 生成反向图（saved tensors 由图记录）
fwd_g, bwd_g = aotautograd.forward_and_backward(fwd_graph)
# 3. MTIA backend: 应用 MTIA 优化算子分解（算子 → MTIA 原生 op）
# 4. TorchInductor: 手写 pattern-based fusion + 编译器驱动 fusion + Triton 代码生成
# 5. 图调度器（ILP 启发式）降低峰值内存 + activation rematerialization
# 6. compute 与 collective（HCCL）合入单一大图（见"单图编译"条目）
```
与标准 torch.compile 差异：collectives 也进入同一张图（compute+collective 单图）、图调度器做训练内存优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TorchDynamo/AOTAutograd 为 PyTorch 开源组件（torch._dynamo / torch._functorch.aot_autograd）；MTIA 以自定义 backend（torch.compile(backend="mtia")）接入。使用场景：PyTorch 原生训练/推理图编译（FSDP2/DTensor/TorchRec/XFormers 均可）；MTIA 支持 eager + graph 双模式（eager 模式有 host 开销问题，见论文 VI 节）。信息缺口：论文未给出 MTIA backend 在 Dynamo/AOTAutograd 上的具体接入点与 guards 处理。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
