## Nerfstudio（神经辐射场开发框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Nerfstudio（Tancik et al., SIGGRAPH 2023，nerfstudio-project/nerfstudio）是模块化的 NeRF/神经渲染开发框架，统一了数据加载、模型（vanilla-nerf、instant-ngp、splatfacto 等）、训练、渲染与评估流程，提供 `ns-train`/`ns-eval` 等 CLI。NeRArch-Sim 论文把 NerfStudio 选为其算法基础设施框架（Fig. 5 的 algorithm infrastructure framework），利用其与 NeRArch-Sim 统一分类学的相似性，通过运行时钩子插桩（注入 tracing.py/trace_config.json/eval.py 等）在其上采集算子级执行 trace（execution_dag.pkl），从而把"用户定义的渲染 pipeline 与数据集"变成可调度算子图。框架层面的算法框架与 serving 框架（vLLM/SGLang 类）不同：它是离线训练/渲染研究框架，而非多请求服务系统。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（NeRArch-Sim 插桩版）：`git clone nerfstudio && git checkout 50e0e3c70c775e89333256213363badbf074f29d && pip install -e .` → 拷贝 Instrumentation/nerfstudio_vendor 的 tracing.py/trace_config.json/eval.py/train.py/eval_utils.py/train_utils.py → 训练模型（vanilla-nerf/instant-ngp/splatfacto）→ `ns-eval --load-config <config> --render-output-path $OUT --enable-trace --trace-config-path .../trace_config.json --eval-image-indices 0`（DISABLE_TRACE_PLOT=1）渲染指定图像并写 $OUT/execution_dag.pkl → `python Instrumentation/plot_transformed_operators.py` 可视化 / `./nerarch_sim analyze ...` 消费。支撑的模型/数据集：vanilla-nerf（MLP-based，NeRF-Synthetic）、instant-ngp（grid-based，哈希编码）、splatfacto（primitive-based，3DGS）；NeRArch-Sim 评估用 NeRF-Synthetic（小场景）与 Unbounded360（大场景）数据集。兼容性：Nerfstudio、GauStudio、Kaolin Wisp 等框架的分类学与 NeRArch-Sim 相似，可直接接入其软件工作流。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Python/PyTorch，pip 安装，`ns-train <method>` 训练、`ns-eval` 评估；NeRArch-Sim 的插桩通过 vendor 文件注入实现（不修改原算法），tracing.py 用轻量运行时钩子拦截函数调用记录算子标识/张量形状/值。使用场景：作为"算法→硬件"co-design 的前端——用户定义任意渲染 pipeline 与数据集，NeRArch-Sim 据此生成算子图并映射到模块化硬件库（SystemC/Catapult HLS 20+ 模块）做 PPA 评估与 DSE。替代/相关框架：GauStudio（3DGS 模块化框架）、Kaolin Wisp、nerfacc（加速工具箱）、tiny-cuda-nn（Instant-NGP 哈希编码实现，NeRArch-Sim 环境依赖）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
