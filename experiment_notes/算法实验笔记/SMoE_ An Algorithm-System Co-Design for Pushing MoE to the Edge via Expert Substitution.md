## SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 expert substitution（专家替换）算法：针对 fine-grained MoE（DeepSeekMoE、Qwen2-57B-A14B、XVERSE-MoE），利用 router gate score 将激活专家分为 top-score 专家（score > (1+α)S_{k+1}）与 low-score 专家（S_{k+1} ≤ score < (1+α)S_{k+1}），把 low-score 专家替换为 GPU 显存中已缓存、gate score 落在 [(1−α)S_{k+1}, S_{k+1}] 的未激活专家，在几乎不损失精度的前提下减少 CPU→GPU 的 PCIe 专家加载与 CPU 专家计算。算法组件：①expert-cache router（Algorithm 1）：按 α 阈值分类并选替换候选，使 |E_l \ G| 与 |E_s| 匹配（替换不足的 low-score 专家回退为 PCIe 加载或 CPU 计算）；②score-aware cache eviction：按过去 n 次迭代的平均 activation score 淘汰最低分专家（Eq.3，替代 LRU），并对当层被选中专家加 protection shield 防止用前被淘汰；③top-score prefetching：用 GPU 常驻共享专家+缓存专家计算 hidden state 预测下一层 top-score 专家并预取（命中率约 82%，95% 概率为 active）；④CPU-assisted task load scheduling（Algorithm 2）：双指针按 C_load/C_CPU 成本最小化 max(T_load, T_CPU)。优化目标为逐层最大化 max |G∩E_a| + min(|E_l\G|, |E_s|)（Eq.1/2）；超参 α 通过 min_α A(α) s.t. T(α) ≤ R（TPOT 预算）多项式拟合+一维搜索选取。
  - 实验比较：baseline 为 4 个支持显存不足时本地运行 MoE 的系统——MoE-infinity（activation-aware 预取，依赖历史 router 数据）、llama.cpp（纯 CPU 推理）、DeepSpeed（layer-wise 加载、静态缓存）、HybriMoE（CPU-GPU 调度+缓存管理，基于 ktransformer 架构自带量化，论文移除了量化效果做公平对比）；另对比 expert-skipping（直接丢弃将被替换的专家，激活数动态变化）与 GPTQ INT8 量化（配置为与 SMoE 相同 GPU cache hit rate）。指标：TPOT（解码，batch=1/3）、TTFT（prefill，batch=1）、GPU expert cache ratio、Accuracy/GPT-4 Score/pass@1。主要结果：相对最优 baseline，TPOT 平均降 24%（batch=1）/35%（batch=3），S3 下 48%（batch=3）/34%（batch=1）；GPU cache ratio 相对提升 ≥65%、命中率 >60%（Qwen2-57B 达 71%）；TTFT 平均降 11%；α<0.35 时精度几乎无损（部分数据集反而提升，归因于抑制 low-score 专家的 noisy activation）。Ablation（batch=1，逐步增量）：baseline=LRU offloading（无 router/prefetch/cache 策略）；+CE（cache eviction）TPOT −8%、cache ratio +11%；+CR（expert-cache router）再 −20%、+60%；+Pre（prefetching）再 −14%、+12%（PCIe 时间略增但被重叠掩盖）；+BA（CPU-assisted scheduling）再 −34%、−3%。
  - 硬件平台是什么，配置是什么。
  - 单卡边缘/低端 GPU（无量化、lossless 推理）：S1 用 NVIDIA RTX 3080 Ti 12GB 跑 deepseek-moe-16b（31GB）；S2 用 NVIDIA RTX 4060 Ti 16GB 跑 XVERSE-MoE-A4.2B（49GB）；S3 用 NVIDIA A6000 48GB 跑 Qwen2-57B-A14B（107GB）。Edge/Legacy 平台为 PCIe 3.0 + Intel E5-2683 v3；High-end 平台为 PCIe 4.0 + Intel Xeon Gold 6444Y；S3 复现需约 150GB CPU 内存承载模型权重加载/卸载过程。
  - 模型是什么。数据集和bench分别是什么。
  - 模型：deepseek-moe-16b-base（S1）、Qwen2-57B-A14B-Instruct（S3）、XVERSE-MoE-A4.2B-Chat（S2），均为 fine-grained MoE（含 shared experts）。数据集/bench：Gaokao（Math_I/Math_II/History/Biology）、MMLU（College_computer_science/Management/International_law/Logical_fallacies）、TriviaQA、RACE-mid、WiC、GSM8K、MT-Bench（GPT-4 Score 1–10）、HumanEval（pass@1）；每数据集采样约 1000 条、prompt 取自 OpenCompass；精度评估用 OpenCompass 框架（Gaokao/triviaQA/WiC/Race-mid/gsm8k/MMLU）。
  - 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：已开源 https://github.com/goingshr/SMoE（figshare 备份 https://doi.org/10.6084/m9.figshare.31982136），artifact 聚焦复现 S3（Qwen2-57B-A14B）的 TPOT 与 GPU cache hit ratio（对应 Fig.12/13）。环境：Python 3.13 free-threading（no-GIL）conda 环境、Ubuntu 专属 dependency.sh（apt/snap/sudo、Rust 工具链、tokenizers 源码编译）；运行脚本 run.sh 以大写环境变量传参（MODEL_NAME/MODEL_PATH/CONFIG_PATH/DATASET_PATH/INPUT_NUM/BATCH_SIZE/OUTPUT_LEN/GPU_MEM/CPU_CORES/LOG_DIR 等），直接入口 main.py；--alpha 为论文文档中的替换阈值参数（S3 默认 0.25；仓库实际 config 用 replaceScoreRatio 等字段：replaceScoreRatio/window_size（null=LRU）/if_prefetch/if_usecpu/if_replace）。安装与运行例子：
    ```
    conda create -n SMoE python=3.13 python-freethreading -c conda-forge
    conda activate SMoE
    bash dependency.sh   # sudo；装 Rust 工具链并源码编译 tokenizers（no-GIL 无预编译 wheel）
    pip install -r requirements.txt
    MODEL_NAME=qwenmoe MODEL_PATH=parameters/qwenmoe GPU_MEM=43 \
      CONFIG_PATH=configs/qwen2moe_config.json bash run.sh
    # 或直接：
    python main.py --model_name qwenmoe --model_path parameters/qwenmoe \
      --config_path configs/qwen2moe_config.json --dataset_path gaokao_math_ii \
      --input_num 20 --output_len 60 --cpu_cores 4 --GPU_mem 43
    ```
    算法 pipeline 执行例子（Expert-Cache Router，Algorithm 1 伪代码级，单 token t）：
    ```
    # 1) gate 输出该层全部 expert 的 score 向量 S_t，降序排序；S_{k+1} = (k+1)-th score
    # 2) 分档：T=(1+α)S_{k+1}（top-score 线）、L=S_{k+1}、R=(1−α)S_{k+1}
    #    - score > T            → top-score 专家，直接保留进输出 O[t]（C 集）
    #    - L ≤ score < T        → low-score 专家（B_t 集 = E_l）
    #    - R ≤ score < L 且已在 GPU 或属于 C 集 → 可替换候选（A_t 集 = E_s）
    # 3) 替换决策：若 |A_t| ≥ |B_t|，取 A_t 中最高分的 |B_t| 个替换 low-score 进 O[t]；
    #    否则全部 A_t 替换，剩余 |B_t|−|A_t| 个 low-score 专家留给 PCIe 加载/CPU 计算
    # 4) score-aware eviction（Eq.3）：对 expert i 维护近 n 次迭代平均 activation score，
    #    淘汰 argmin_i (Σ_{k=max(1,j-n)}^{j} S_{i,k})/(窗口长度)；当层被选中的 expert
    #    加 protection shield（层计算完成后自动解除）避免用前被淘汰
    # 5) 输出：GPU 直接计算替换后 expert 集合；top-score 专家由 prefetch 提前加载
    ```
    （论文未涉及编译框架、硬件架构、芯片设计层次。）
