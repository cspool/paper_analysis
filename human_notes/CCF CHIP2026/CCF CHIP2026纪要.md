# CCF CHIP2026 纪要

## 微纳核芯：三维存算一体 3D-CIM

- RV 扩展存算一体架构的指令集。
- 微纳核芯：芯片设计，22 nm 平替 7 nm 工艺。

**三维存算一体 3D-CIM 创新架构**

![三维存算一体 3D-CIM 创新架构][img-01]

**PCIe-CIM 与 LP-CIM 核心产品**

![PCIe-CIM 与 LP-CIM 核心产品][img-02]

## 多模态 BEV 感知与 Slim-UniTR 加速

- BEV 多模态 Slim 加速。

**自动驾驶全时段、全场景感知需求与 nuScenes/NDS 评估基准**

![自动驾驶感知需求与评估基准][img-03]

**多模态感知算法 UniTR**

![多模态感知算法 UniTR][img-04]

**单模态稀疏点云加速器与多模态融合处理器**

![感知硬件加速器进展][img-05]

**多模态 BEV Transformer 的挑战与 Slim-UniTR 芯片方案**

![多模态 BEV Transformer 的挑战与芯片方案][img-06]

**显著性引导的自适应计算策略**

![显著性引导的自适应计算策略][img-07]

**Transformer 权重共享的多模态融合数据流架构**

![Transformer 权重共享多模态融合架构][img-08]

**高能效 SoC 顶层架构设计**

![高能效 SoC 顶层架构设计][img-09]

**28 nm 芯片实现与测试结果**

![28 nm 芯片实现与测试结果][img-10]

## 沐曦：先进封装与高速互连

**算力需求与 GPU 服务器互连背景**

![算力时代的应用需求][img-11]

**3D Fabric 等先进封装技术**

![先进封装技术][img-12]

**互连演进的三个世代与生态系统的四个分层**

![互连演进与生态系统分层][img-13]

**MetaXLink 高速互连：Scale Up 与 Scale Out**

![MetaXLink 高速互连][img-14]

## TileAI、TileLang 与 TileRT

- TileAI 和编译生成算子：以 M+N 作为 CUDA C 之上的中间层。
- 不同于 Triton，Gluon 的方式：Triton 易用，Gluon 性能。
- TileRT：多卡流水。
- 访存方面 TileLang 较为极致，Tensor Core 还没有极致。
- TileLang 相比 Triton 的编程方式提供更多硬件层次。
- TileLang 定义硬件模型，以适配更多芯片。
- TileLang 能否用于 DCU？
- Warp serialization？B GPU 特性？
- TileRT 不同于 megakernel：Kimi 500 token 场景中，低延迟下获得高吞吐的成本较高，需要重新设计硬件。
- 低延迟方向：面向特定模型的特定软硬件框架。

**统一软件生态用于降低模型与芯片组合的 M×N 适配复杂度**

![统一软件生态与 M×N 适配复杂度][img-15]

**TileAI 基础软件生态**

![TileAI 基础软件生态][img-16]

**TileAI 系统组件**

![TileAI 系统组件][img-17]

**TileLang：Tile Virtual Machine 与控制 API**

![TileLang 虚拟机与控制 API][img-18]

**TileRT：Tile Runtime**

![TileRT 运行时][img-19]

### TileRT 技术补充（官方仓库 v0.1.5）

> 资料基线：[TileRT 官方仓库](https://github.com/tile-ai/TileRT)的 `main` 分支与 v0.1.5 文档，查阅于 2026-07-22。以下将项目的设计目标与当前公开实现分开描述。

#### 定位与核心机制

- **优化目标是低时延而非大批量吞吐。** TileRT 面向对单请求响应时间敏感的场景，核心指标是每输出 token 时间（time per output token，TPOT）；其目标是在不缩小模型或牺牲模型质量的前提下，将数千亿参数模型的 TPOT 压到毫秒量级。
- **调度粒度下沉到 tile。** 编译器先把 LLM 算子拆成细粒度的 tile 级任务，运行时再跨设备动态重排这些任务，而不是把一个算子或一次完整迭代视为不可拆分的调度单元。
- **统一覆盖计算、I/O 与通信。** 运行时把三者放进同一调度视野，通过多设备间的高度重叠减少流水气泡和设备空闲。可将其主路径概括为：`模型算子 → tile 任务 → 跨设备动态调度 → 计算/访存/通信重叠 → 降低 TPOT`。
- **与 megakernel 的侧重点不同。** 官方仓库没有直接给出二者对照；根据其公开设计描述可推断，megakernel 主要依靠扩大融合范围、减少 kernel launch 和中间同步，而 TileRT 保留 tile 级运行时调度，重点是跨算子、I/O、通信和多设备的动态重叠。这是调度抽象上的差异，不等价于简单的“kernel 更大或更小”。

#### MTP：缩短自回归串行深度

- TileRT 支持 Multi-Token Prediction（MTP）式推测解码，可通过 `--with-mtp` 开启：一次前向提出并验证多个 token，从而减少逐 token 解码的串行步数。
- 当前 DeepSeek-V3.2 生成器在 MTP 模式下设置长度为 4 的预测窗口，并统计每一步实际接受的 token 数；因此性能不仅取决于单次前向时间，也取决于平均接受长度。仓库 README 的 v0.1.5 图表分别报告无 MTP、平均接受长度 3.2 和理想接受长度 4.0 的结果。

#### 多卡执行、权重布局与模型后端

- 当前发布配置以 **8× NVIDIA B200** 为目标。`weight_converter` 将 Hugging Face checkpoint 转为 8 个按设备划分的 shard，键名带有 `*_dev_0` 至 `*_dev_7` 后缀，运行时直接加载该布局。
- DeepSeek-V3.2 与 GLM-5/5.1 分别使用 `libtilert_dsv32.so` 和 `libtilert_glm5.so`。Python 层通过 `tilert.load_backend(model_type)` 延迟加载对应原生后端。
- 两个后端都会注册同一个 `torch.ops.tilert` 命名空间，所以同一 Python 进程只能加载一个模型后端；不同模型需要运行在独立进程中。

#### Prefill–Decode 分离

当前版本可以让 vLLM 负责 prefill、TileRT 专注低时延 decode，链路为：

`OpenAI 兼容请求 → 路由器 → vLLM prefill → 传输注意力/KV 状态 → TileRT decode → 流式返回`

- 集成使用 vLLM V1 的标准 `KVConnector` 插件接口，不要求维护 vLLM fork；`TileRTConnector` 负责请求认领、分块 prefill 跟踪、暂存区管理和后台发送，模型相关的缓存提取、布局与 RDMA 规划则委托给所选 profile。
- KV 数据面支持 **NIXL** 或 **Mooncake**。使用 NIXL 时可通过 RDMA NIC 传输；prefill 与 decode 两端的 KV-cache dtype 必须匹配，否则握手会拒绝连接。
- 解码端依次执行 `receive → convert → inject → decode`：先接收并转换外部 prefill 产生的缓存，再注入 TileRT 引擎并继续自回归解码。当前公开的 PD decode server 是 `batch size = 1` 的独占槽位，繁忙时返回 HTTP 429，由上层路由器做门控分发。
- `MultiConnector` 可以让同一 vLLM prefill 池同时连接 TileRT decode 池与原生 vLLM decode 池：延迟敏感请求进入 TileRT，其余请求保留在通用 vLLM 路径，共享同一个 OpenAI 兼容入口。

#### 当前公开实现的边界

- v0.1.5 仍被标为 preview。官方 wheel 绑定精确 ABI：Linux x86-64、Python 3.12、`torch==2.11.0+cu130`，并要求可支持 CUDA 13.2 runtime 的驱动；这不是可随意放宽的最低版本声明。
- 公开仓库主要提供 Python 生成接口、模型封装、基准测试和 PD/vLLM 集成；核心 TileRT 后端随 wheel 以预编译 `.so` 提供。仓库的 `pyproject.toml` 明确不提供本地构建 wheel 所需的 build-system，底层编译技术仍在逐步并入并公开到 TileLang/TileScale。
- README 中的速度数据均绑定特定版本、模型、序列长度、MTP 接受率和 8× B200 环境，不宜直接外推到其他 GPU、batch size 或服务负载。

相关源码入口：[`tilert/__init__.py`](https://github.com/tile-ai/TileRT/blob/main/tilert/__init__.py)、[`tilert/generate.py`](https://github.com/tile-ai/TileRT/blob/main/tilert/generate.py)、[`tilert/pd_vllm`](https://github.com/tile-ai/TileRT/tree/main/tilert/pd_vllm)、[`pyproject.toml`](https://github.com/tile-ai/TileRT/blob/main/pyproject.toml)。

**TileLang 的异构芯片适配与代码生成**

![TileLang 的异构芯片适配][img-20]

**TileLang 对 AMD GPU 的官方适配**

![TileLang 对 AMD GPU 的适配][img-21]

## 模型量化、动态计算与推理优化

- 低秩分解：擅长手机场景；受硬件限制与精度计算种类影响。
- 刘方鑫。
- 动态 skip 语言模型。

**从稀疏化、量化到硬件映射的核心挑战**

![模型量化与硬件映射的核心挑战][img-22]

**智能模型、算法、系统/编译、架构与异构硬件的研究视野**

![模型与系统协同优化研究视野][img-23]

### ASTER：Transformer 自适应动态层跳过

![ASTER 现有方案与核心问题][img-24]

![ASTER 自学习与执行流程][img-25]

![ASTER 的核心观察][img-26]

![ASTER 方案总览][img-27]

### MILLION：基于 FPQ 的 KV 缓存压缩与推理加速

![MILLION 基于 FPQ 的 KV 缓存压缩与推理加速][img-28]

### CSD：推测式解码框架设计

![CSD 推测式解码框架设计][img-29]

### EARTH：基于信息熵表征的 MoE 编码加速

![EARTH 基于信息熵表征的 MoE 编码加速][img-30]

### STEP：基于时空预取的高效 MoE 推理优化

![STEP 基于时空预取的高效 MoE 推理优化][img-31]

### BEEMS：通过计算图优化实现内存“削峰填谷”

![BEEMS 计算图优化与内存削峰填谷][img-32]

[img-01]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=1073056524733036964&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-02]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=817167660672122046&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-03]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=6033895274712179415&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-04]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=1918353605020269486&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-05]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=5947363884491431773&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-06]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=1897114481558919550&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-07]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=7113658207520686704&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-08]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=777735658410387085&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-09]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=1944285844335513367&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-10]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=6087046548308056556&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-11]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=6920369600014091380&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-12]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=4342443095425047695&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-13]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=9128670197875006368&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-14]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=4356906795582301908&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-15]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=8750773257428481508&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-16]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=2401960118923555436&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-17]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=5986944373451856120&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-18]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=8028348179555281052&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-19]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=6843077480281932308&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-20]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=7201352098357517194&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-21]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=5492797159185319875&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-22]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=4690253111055066291&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-23]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=7621189830830984238&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-24]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=5953613715492679462&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-25]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=4441478901545846945&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-26]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=8742450183335683847&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-27]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=1642569138390618447&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-28]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=181145510837521056&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-29]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=3438597452069399898&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-30]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=8649053117428896258&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-31]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=5690769330677148119&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
[img-32]: <_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=8190134147301125154&skey=@crypt_c64525af_9af421491ada415095bb811e9dcc9880&mmweb_appid=wx_webfilehelper.jpeg>
