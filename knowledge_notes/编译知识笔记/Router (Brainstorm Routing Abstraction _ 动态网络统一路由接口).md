## Router (Brainstorm Routing Abstraction / 动态网络统一路由接口)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Router 是 Brainstorm 框架提出的统一路由抽象，解耦动态网络中的控制流（routing decision）与数据流（branch execution）。它包含两个核心部分：(1) `router_fn`——由模型开发者定义，接收 Cell-annotated tensor 并返回 Routes tensor（每个 Cell 的目标 branch ID，-1 表示丢弃）；(2) `Router` 框架执行引擎——负责根据 Routes 高效完成 Cell 的分发（单 GPU 用 custom rearrangement kernel，多 GPU 用 sparse point-to-point 通信），并将 Routes 决策异步写入 JIT Profiler 的 profile 文件供后续动态优化使用。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Router 是 Brainstorm 编译框架与数据流图交互的关键接口。与传统 DL 框架的 IR 将 control-flow 嵌入 dataflow graph 不同，Brainstorm 将 Router 抽象为一个"数据分发算子"——编译器不关心 router_fn 内部的控制流逻辑（如 Top-K、KNN、if-else），只需关心输入 tensor、Router 配置、输出 tensor 的 structure。这使得：
1. **图分析简化**：编译器可以提取静态 sub-network（每个 branch 内部），用现有静态优化（TVM auto-tuning、vertical fusion）处理
2. **Profile 收集自动化**：每个 Router 被调用时的 Routes 决策自动记录，无需开发者手动插桩
3. **优化 Pass 解耦**：torch.fx 的优化 Pass 通过识别 Router 节点来触发图变换（替换 Router 后的串行 branch 为 fused kernel、插入 preload 算子、重排算子的执行顺序实现 speculative execution）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
```python
class Router:
    def __init__(self, router_fn: Func):
        self.router_fn = router_fn  # 开发者定义的 routing logic
    def forward(self, x: Tensor, **kwargs) -> Tuple[Tensor], Routes:
        routes = self.router_fn(x, **kwargs)  # 计算每个 Cell 的目标 branch
        outputs = dispatch_cells(x, routes)   # Brainstorm 的 GPU kernel 执行分发
        if profiling_enabled:
            jit_profiler.record(routes)       # 异步写入 profile
        return outputs, routes
```
开发者仅需将现有 routing 逻辑包装为 router_fn（如 MoE 的 softmax + top-k），框架自动处理 Cell 分发和 profile 收集。端口 SwitchTransformer 仅需改 12 行代码。

涉及论文标题：
- Optimizing Dynamic Neural Networks with Brainstorm
