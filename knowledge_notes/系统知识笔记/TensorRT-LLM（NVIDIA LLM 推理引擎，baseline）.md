## TensorRT-LLM（NVIDIA LLM 推理引擎，baseline）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TensorRT-LLM 是 NVIDIA 的 LLM 推理引擎/服务框架：基于 TensorRT 与 CUDA，对 Transformer 各组件提供高度优化的预编译 kernel（融合 attention、KV cache 管理、in-flight batching、paged KV cache、量化支持 FP8/INT4 等），以 Python/C++ API 部署大模型，是业界代表性手工优化推理引擎。QiMeng-Tensify（ISCA'26）把它作为网络级（end-to-end）评估的"vendor-provided inference framework"baseline（v1.1.0）：相对 TensorRT-LLM，QiMeng-Tensify 在 A100 上平均快 1.22×、H100 上快 1.29×（网络级 4 个 LLM：Chameleon-7B/LLaMA3-8B/GPT-3-7B-LoRA/nGPT-1B，batch 1/8、seq 4096）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
作为 baseline 的系统运转流程（端到端 LLM 推理）：输入请求 → 引擎做图优化（算子融合、kernel 选择、KV cache 分配/分页、in-flight batching 调度）→ 逐层执行预编译的 TensorRT 优化 kernel（各算子均为专家手写/模板优化，如 fused attention、GEMM）→ 输出 token。局限（论文视角）：所有算子/子图优化是手工设计、固定模板，只能覆盖预定义算子模式——对 LoRA 层（3 个 GEMM 的依赖链）、GatedMLP 全融合、NSA 等新算子无法自动发现跨算子融合；且不同模型需重新手工调优。QiMeng-Tensify 用 LLM-guided MCTS 自动搜索的 kernel 在整网推理延迟上反超（相对 1.08×~2.42× 延迟更低）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NVIDIA 闭源/部分开源（github.com/NVIDIA/TensorRT-LLM），基于 TensorRT 的 engine 化 + 自定义 CUDA 插件；支持 llama 等主流架构，通过 engine build（含量化、kernel 选择）→ runtime（KV cache、batching）两阶段使用。使用方式：作为论文端到端 baseline（与 PyTorch、Mirage 一起，Table VI 列出 v1.1.0 版本）；其"预定义算子模式"限制正是论文"手工优化难以随模型创新扩展"论点的例证；论文未修改 TensorRT-LLM（纯对比）。层次说明：TensorRT-LLM 作为推理引擎兼具编译（engine build）与 serving（运行时调度）功能，此处归入系统架构（网络级推理运行时），其 engine 构建部分与编译框架层相关。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
