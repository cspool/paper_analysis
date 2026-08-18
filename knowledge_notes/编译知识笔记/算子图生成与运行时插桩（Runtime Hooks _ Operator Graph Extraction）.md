## 算子图生成与运行时插桩（Runtime Hooks / Operator Graph Extraction）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 运行时插桩是在不修改原算法代码的前提下，通过轻量钩子拦截函数调用、记录执行轨迹的技术；算子图生成是把轨迹处理成有向无环算子图（DAG）供后续分析。NeRArch-Sim 的模块化软件工作流用运行时钩子插桩现有算法框架（Nerfstudio），捕获函数标识符（按 software.json 指定）、调用次数、张量形状、张量值并保持 caller-callee 关系，处理成算子图（含函数依赖、I/O 数据量、调用频率），输出 execution_dag.pkl；还会重放提取的算子图并与原 pipeline 输出对比验证无质量退化（保证不漏算子）。另可在 GPU 上对算子做 roofline 分析作开发者参考。与本库既有条目（Pin DBI、kprobes 动态插桩）的区别：NeRArch-Sim 的钩子针对算法框架函数（taxonomized interface）而非二进制/内核指令。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：checkout Nerfstudio 指定 commit → 拷贝 Instrumentation/nerfstudio_vendor 的 tracing.py/trace_config.json/eval.py/train.py 等 → 训练模型（vanilla-nerf/instant-ngp/splatfacto）→ `ns-eval --load-config <config> --enable-trace --trace-config-path .../trace_config.json --eval-image-indices 0`（DISABLE_TRACE_PLOT=1）→ 得到 $OUT/execution_dag.pkl → `python Instrumentation/plot_transformed_operators.py` 可视化 / `./nerarch_sim analyze ...` 直接消费。算子图是后续 map/schedule/report 的输入。验证：`./nerarch_sim validate` 检查 DAG 无环、四阶段覆盖/顺序（SAMPLING→ENCODING→FIELD_COMPUTATION→BLENDING）、重放捕获 field 输出渲染成图并与 Nerfstudio 渲染/ground truth 三方对比（强制重算 PSNR 门限 35dB、阶段消融 ≥10dB、GT 一致性 ~1dB），输出 PASS/FAIL。latency（表 XI）：instrumentation 22.8~45.2s、graph construction 0.4~0.8s；硬件-only DSE 时算子图静态、instrumentation 只做一次复用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Python 运行时钩子（tracing.py）注入 Nerfstudio 的模型 forward 路径，trace_config.json 配置要捕获的算子集合；Operators/ 目录提供算子分类学、绘制与集成。使用：作为"算法→硬件"的桥——由于 NeRArch-Sim 分类学与算法框架（Nerfstudio、GauStudio、Kaolin Wisp 等）相似，可无缝接入其 workflow 支持算法-硬件协同设计；扩展新 pipeline（SRender/Lumina/4DGS）时只需实现新增算子（<300 行 C++），算子图结构不变则改参数无需重新插桩（可秒级评估算法变体）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
