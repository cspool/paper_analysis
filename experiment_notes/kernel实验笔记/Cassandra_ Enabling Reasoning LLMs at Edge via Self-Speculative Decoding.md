## Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - （近似归类，反馈：本论文无 GPU 性能 kernel 创新，性能由硬件模拟器（见实验_硬件架构条目）评估；此处记录其评估软件中的 kernel 层实现。）实现为功能建模软件模拟器的自定义 CUDA kernels：PyTorch + custom CUDA kernels 精确模拟格式变换中的 Wanda 激活感知剪枝、KV per-token 剪枝、mantissa 截断、unary 编码与 MX 格式转换、bitmap 稀疏化/去稀疏化；作者明确说明该模拟器不带来 GPU 性能增益，仅用于精度与接受率评估。运行时数据管理由 superblock-based data management 承担（decoder buffer 拼接、memory controller 按 block 跳过与地址记账，见硬件架构条目）。实验比较：zero-shot 精度 vs BF16/SmoothQuant/QoQ/SqueezeLLM/DuQuant/Wanda；接受率与性能 vs Draft&Verify/MagicDec/Lookahead Decoding/EAGLE-3。
- 后端平台是什么，配置是什么。
  - 软件模拟器运行于 NVIDIA GPU（超参 grid search 在 A100，8B 模型约 5 分钟）；目标后端为 Nvidia RTX 4090、Jetson AGX Orin 与自研 64 TFLOPS NPU（其性能由 Accel-Sim/Scale-Sim 扩展模拟器评估，见硬件架构条目）。
- 评估性能的软件/脚本是什么。修改了什么。
  - PyTorch + 自研 custom CUDA kernels（论文未给出具体 kernel 名/接口）；vLLM 官方 INT8 量化实现仅作 SmoothQuant W8A8 量化 baseline，FP8 动态量化在 RTX 4090 实测作对照。CUDA kernel 实现点：importance = |W|⊙‖X_calib‖₂ 与 top-k 选择、unary 码字切分与并行 0 计数、MX 共享指数广播、bitmap 稀疏化/去稀疏化、zero-padding 重建。
- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - Cassandra 模拟器代码：论文未提供开源链接，未检索到公开仓库（无法确认）。评估原理示例：输入 BF16 权重张量 W 与校准激活 X → kernel 1 计算 |W|⊙‖X‖₂ 得 importance → kernel 2 top-k 生成 bitmap 并分组 speculation/verification → kernel 3 对 speculation 组指数做 unary 编码（按频率 codebook）或 MX 共享指数、并截断 mantissa 低 4 bit → kernel 4 按 bitmap 重建草稿张量（补 0）→ 标准 FP GEMM 前向得 draft logits；KV cache 每 token 在线执行同样变换。输出：接受率、压缩率与 zero-shot 精度（如 Deepseek-R1-Distillated-Llama3-8B Cassandra-1：GPQA 49.0 / Math-500 87.0 / AIME 26.7，接受率 γ=5 时 0.74–0.88）。
