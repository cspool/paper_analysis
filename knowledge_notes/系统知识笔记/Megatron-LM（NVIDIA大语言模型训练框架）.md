## Megatron-LM（NVIDIA大语言模型训练框架）

术语是什么？
Megatron-LM 是 NVIDIA 开发的大语言模型分布式训练框架，核心贡献是提出并实现了高效的 Tensor Parallelism (TP) 和 Pipeline Parallelism (PP) 技术 (Shoeybi et al., 2019; Narayanan et al., 2021)。支持 Transformer-based 模型（GPT、BERT、T5 等）在数百至数千 GPU 上的 3D 并行训练（TP+PP+DP）。PPMoE 基于 Megatron-LM v2.6 实现；DPMoE baseline 基于 Megatron-LM v2.5 + DeepSpeed v0.5.10。

从系统架构角度拆解术语：
Megatron-LM 在 MoE 训练中的架构：
1. TP：将每个 Transformer layer 内的 GEMM 沿列/行切分到单节点 GPU 上，通过 NVLink all-reduce 同步。
2. PP：将模型按 layer 切分为多个 stage 分布到不同节点，通过 p2p send/recv 传递 hidden states。
3. DP：复制完整模型到不同 DP group，通过 all-reduce 同步梯度。
4. EP (v2.5+)：支持 MoE expert parallel，experts 分布在 EP group 内，通过 all-to-all 通信 dispatch/gather。
PPMoE 的核心修改在 model definition 层：将 EP 与 TP（而非 DP）绑定，用 index_select + all-reduce 替代 all-to-all。

术语一般如何实现？如何使用？
GitHub: https://github.com/NVIDIA/Megatron-LM。通过 `--tensor-model-parallel-size`、`--pipeline-model-parallel-size`、`--num-experts` 等参数配置。典型 MoE 训练命令包含 TP、PP、DP、EP 四个维度的并行度配置。PPMoE 实验中 TP=8, PP=4/16, DP=1, EP=64。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
- ReXMoE Reusing Experts with Minimal Overhead in Mixture-of-Experts

REXMoE 在 Megatron-LM 上修改了 MoE Block 和 TopK Router 实现以支持跨层 expert reuse：每层 forward 时从相邻 r 层收集 expert 参数引用组成扩展候选池，实现了 PSR (Progressive Scaling Routing) 的 curriculum learning 训练。训练使用 4 nodes × 32 Hopper GPUs，Expert Parallelism (EP)=8（当 routed experts >8 时）。
