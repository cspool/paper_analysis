## SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution（近似层次匹配：本层取其 MoE 专家级 offloading/缓存/预取/CPU-GPU 调度；论文是自研 Python 推理运行时，而非修改开源 Serving 框架）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是面向边缘 GPU 显存受限场景的 online MoE expert offloading 调度器 SMoE：所有专家常驻 CPU 内存，按需经 PCIe 加载到 GPU 或直接在 CPU 计算，完全在线运行、无需 offline 准备。调度策略四部分：①expert-cache router 按 gate score 把激活专家分成 top-score/low-score，low-score 替换为 GPU 已缓存的近似 score 专家（提升 GPU 命中率、减少 PCIe 传输与 CPU 计算）；②score-aware cache eviction（按近 n 次平均 activation score 淘汰最低分专家、替代 LRU，带 protection shield）；③top-score prefetching（用 GPU 常驻共享专家+缓存专家预测下一层 top-score 专家，与当前层计算重叠）；④CPU-assisted task load scheduling（Algorithm 2，双指针按 C_load/C_CPU 成本最小化 max(T_load, T_CPU)）。目标：低 batch（1–3，边缘单设备场景）下最小化 TPOT 与 TTFT。
  - 实验比较：与 MoE-infinity（预取+历史 router 数据）、llama.cpp（纯 CPU 推理）、DeepSpeed（layer-wise 加载、静态缓存）、HybriMoE（CPU-GPU 调度+缓存管理）比 TPOT/TTFT/GPU cache ratio/精度，batch=1/3；S3 下 TPOT 平均降 48%（batch=3）/34%（batch=1），GPU cache ratio 相对提升 ≥65%、命中率 >60%；多 batch 时额外用同 batch 其他 token 的 top-score 专家替换 low-score 专家（top-score 反正要算，免去额外加载）。
  - 硬件平台是什么，配置是什么。
  - 单卡边缘/低端 GPU：S1 RTX 3080 Ti 12GB（PCIe 3.0 + Intel E5-2683 v3）；S2 RTX 4060 Ti 16GB（PCIe 3.0）；S3 A6000 48GB（PCIe 4.0 + Intel Xeon Gold 6444Y）。S3 复现需约 150GB CPU 内存（Qwen2-57B 权重 107GB）。模型：deepseek-moe-16b-base / Qwen2-57B-A14B-Instruct / XVERSE-MoE-A4.2B-Chat。
  - 开源Serving框架是什么。修改了什么。
  - 论文未修改开源 Serving 框架：SMoE 是自研 Python 推理运行时（HuggingFace Transformers 加载模型 + 自实现专家缓存/预取/调度），非 vLLM/TGI 等 Serving 框架改造。调度决策（expert-cache router 轻量搜索、cache eviction、CPU-assisted 调度）全在运行时内完成，与 GPU 计算/PCIe 加载流水线重叠。baseline 中 HybriMoE 基于 ktransformer 架构（自带量化），论文称已移除量化效果做公平对比；DeepSpeed 与 llama.cpp 仅作 baseline 对照，未修改。
  - 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：https://github.com/goingshr/SMoE（figshare: https://doi.org/10.6084/m9.figshare.31982136）。环境：Python 3.13 free-threading（no-GIL）、Ubuntu 专属依赖脚本（dependency.sh：apt/snap/sudo + Rust 工具链 + tokenizers 源码编译）；run.sh 以大写环境变量传参，main.py 为直接入口；论文文档中的 --alpha 控制替换阈值（S3 默认 0.25），仓库实际 config 字段为 replaceScoreRatio/window_size（null=LRU）/if_prefetch/if_usecpu/if_replace。artifact 用 shell 脚本自动跑 Gaokao/triviaqa/WiC/Race-mid/gsm8k 五个数据集并自动解析日志提取 TPOT 与 GPU cache hit ratio（对应 Fig.12/13）。
  - 使用例子（一次 token 解码跨相邻两层，框架输入到硬件执行全过程）：
    ```
    # 输入：batch token 序列（如 gaokao_math_ii 数据集，input_num=20, output_len=60, batch=3）
    # 层 i 执行：
    # 1) GPU 计算 attention + gate（common params 常驻 GPU）→ 该层全部 expert 的 gate score
    # 2) CPU 上 expert-cache router（与 GPU 计算/PCIe 重叠）：按 α 阈值分 top/low-score，
    #    low-score 替换为 GPU 已缓存的近似 score 未激活专家；cache eviction 更新 +
    #    当层保护 shield（防用前被淘汰）
    # 3) CPU-assisted task load scheduling（Algorithm 2）：剩余未命中专家按双指针分配——
    #    高 score 的走 PCIe 加载到 GPU（load 侧），低 score 的留 CPU 计算（CPU 侧），
    #    最小化 max(T_load=n_load×C_load, T_CPU=n_CPU×C_CPU)，C_load 为主成本、C_CPU 视为常数
    # 4) PCIe 操作：预取层 i+1 的 top-score 专家（GPU 上用共享专家+缓存专家预测），
    #    同时加载层 i 需要的新专家；CPU 计算层 i 的部分专家
    # 5) GPU：先算已驻留专家（直接计算），PCIe 完成后继续算新加载专家；CPU 结果传回 GPU 合并
    # 6) 进入层 i+1（层间串行，bubble 被 PCIe 延迟掩盖；层 i+1 的预取已在第 4 步开始）
    # 输出：逐 token TPOT 与 GPU cache hit ratio 日志；hit rate 提升（>60%）→ TPOT 下降
    ```
    作用：以专家替换+命中率优化把 TPOT 中占比 42% 的 low-score 专家加载时间转为 GPU 命中计算，降低边缘设备 MoE 解码延迟且精度无损。
