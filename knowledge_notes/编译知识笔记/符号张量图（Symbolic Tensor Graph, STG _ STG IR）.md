## 符号张量图（Symbolic Tensor Graph, STG / STG IR）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- STG 是 STAGE（Symbolic Tensor grAph GEnerator，Georgia Tech + NVIDIA，ISCA'26，https://github.com/astra-sim/stage）提出的中间表示（IR）：把 LLM 的 tensor 形状、算子和分布策略符号化，使共享相同图结构、仅维度不同的 workload 可复用同一张图。Tensor 表示为 Shape 含符号维（Batch B、Sequence S、Hidden H）与分布符号（dp/tp/sp/fsdp/ep 等），可选 Hidden(H) 标记 partial sum；算子用 `output = op[op_attr](input1,...)` 表达，如 `y = einsum[bm,mn->bn](x,w)`。模块模板（MHA、GQA、MLA、Up-down/Gate-up-down FFN、RMSNorm、MoE、SSM 等）以 STG IR 集成，按层数重复连接组装整模型 STG。vault 笔记证据：paper_secs 中本论文 IV.-STAGE-SYMBOLIC-TENSOR-GRAPH-GENERATOR.md（omnisearch score 4451）、主文（3447）、A.-Large-Language-Models（1209）；RoCC 论文参考文献中也引用 "Symbolic Tensor Graph (STG) Generator"（473）。Web 证据：GitHub astra-sim/stage 描述为 "a generator for Chakra Execution Trace (ET) files" 的合成 transformer workload 生成器，论文 arXiv:2511.10480。
- 从编译框架角度拆解：STG 是"workload 编译器"的前端 IR——①模块模板→②按层组装→③Workload Distributor（张量级分布：按并行维度分片并匹配集合通信；图级分布：PP 切图插 send/recv）→④Graph Instantiation（符号→数值：把 batch/seq/hidden 替换为具体数值并自动传播，可注入 PyTorch/Kineto 真实值）→⑤输出 Chakra DAG。用户只需给模型名/模板 + 并行度两个输入即可编译出任意规模（128K GPU）的执行图。例：x[B/dp,H] 与 w[H,4H] 的线性层按 TP 分片后符号化为 x[B,H/tp@1]、w[H/tp,4H@1]、y[B,4H@1/tp]，实例化时替换 B=128、H=8192 等数值。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 Python 框架（依赖 numpy/sympy/python-graphviz/protobuf/pandas，可 Docker 运行）；命令行 `python main.py --output_dir generated/ --output_name workload.%d.et --comm_group_file comm_group.json --dp 2 --tp 2 --pp 2 --weight_sharded 0`，模型形状用 --dmodel/--dff/--batch/--seq/--head/--num_stacks/--experts/--kexperts 等指定，--model_type 可选 llama/gpt/moe/debug。输出每 rank 一个 .et 文件 + comm_group.json。它使 workload 生成与具体模拟器/系统解耦：同一 STG 可接入 ASTRA-Sim、SimAI、ScaleSim、Genie 等（各 <100 LoC 翻译层）。

涉及论文标题：
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
