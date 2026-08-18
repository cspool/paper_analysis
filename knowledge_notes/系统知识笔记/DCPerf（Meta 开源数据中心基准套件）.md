## DCPerf（Meta 开源数据中心基准套件）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DCPerf 是 Meta 开源的基准套件（https://github.com/facebookresearch/DCPerf），复现 Meta 生产服务的代表性行为，用于数据中心服务器评估。PhaseWeave 使用其中 5 个应用：Django（动态 Web 应用）、Mediawiki（PHP wiki 引擎）、FeedSim（社交信息流排序/聚合）、AdSim（GEMM 型广告排序 ML 推理）、TaoBench（look-through Memcached 数据缓存）；并用套件内的 WDLBench 微基准（https://github.com/facebookresearch/DCPerf/tree/main/packages/wdl_bench）离线训练 phase 预测器。这些应用以微服务架构运行（如 Mediawiki = Nginx+HHVM+Memcached+MySQL 共置一机），覆盖 web 服务、对象排序、数据缓存、CPU 型 ML 推理四类数据中心负载。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在评估流程中的作用：作为请求级负载注入器——论文以 Poisson 分布到达间隔生成请求流，按 baseline 服务器 CPU 利用率 25%/50%/75% 分低/中/高三级负载；端到端延迟=客户端提交请求到收到响应的时长；吞吐=在 SLO（100ms @P99）下提升请求率得到的最大 QPS。DCPerf 提供"生产真实 + 可复现"的工作负载面：其微服务共置/调用链（含 datacenter tax）正是 phase 异质性的载体；WDLBench 提供有标签的 phase 微基准用于离线训练+灵敏度打标（每个微基准 phase 在各 chiplet 上跑一遍确定最优硬件并打标）。泛化验证用另一套未参与训练的 DeathStarBench（https://github.com/delimitrou/DeathStarBench）在预测器上得 91% 准确率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用方式：git clone 后按套件文档部署各 workload（Django/Mediawiki 等为真实开源服务栈），配置请求发生器；论文在仿真（QEMU+SST 内运行 Ubuntu 22.04+Linux 6.8.0-85）与真实 EMR 服务器（28 核 3GHz、128GB DDR5、双 NIC）两种环境运行。价值：给数据中心系统研究提供 Meta 生产形态的开源工作负载基准（区别于传统 SPEC/CloudSuite），使 phase 检测、异构调度等研究的可复现性更强；局限性：论文未说明 DCPerf 的具体部署配置细节（并发数、数据集规模等），复现实验需自行确定，部分细节论文未明确说明。

涉及论文标题：
- PhaseWeave Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters
