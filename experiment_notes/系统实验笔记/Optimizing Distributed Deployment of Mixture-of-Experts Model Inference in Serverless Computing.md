## Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：在 AWS Lambda（CPU serverless 平台）上构建完整的 MoE 模型分布式推理服务系统，包含：(1) 基于贝叶斯决策的专家选择预测器（使用 token ID、position ID、attention ID 三维特征）；(2) 三种 serverless 通信方法（pipelined indirect transfer、non-pipelined indirect transfer、direct transfer）的 scatter-gather 通信设计；(3) 基于 MIQCP 求解的最优部署选择（ODS）算法，联合决策通信方法、内存配置、专家函数副本数；(4) 多维度 ε-greedy search 的贝叶斯优化框架进行全局优化。
  - 实验比较：(1) 不同 scatter-gather 通信方法在不同 token 数下的 billed cost 和吞吐量；(2) ODS vs. MIQCP 直接求解 vs. 随机选择，在不同吞吐量目标下的 billed cost；(3) BO 框架使用不同 acquisition function（multi-dim ε-GS vs. single ε-greedy vs. TPE vs. random vs. no BO）的 billed cost ratio 和专家预测差异；(4) 最终整体性能：Serverless+BO vs. Serverless+real distribution vs. Serverless no BO vs. LambdaML vs. CPU cluster vs. CPU betterTransformer 的 billed cost 和吞吐量。

- 硬件平台是什么，配置是什么。
  - AWS Lambda（CPU-based serverless 平台），14 档内存配置 [128, 768, 960, 1152, 1344, 1536, 1728, 1920, 2112, 2304, 2496, 2688, 2880, 3072] MB。
  - CPU cluster baseline：2×64-core AMD EPYC CPUs，512GB DRAM。
  - 外部存储：2 个 S3 bucket，各 512MB。
  - 最大专家副本数：G=8。

- 开源Serving框架是什么。修改了什么。
  - 论文未基于开源 serving 框架，而是直接在 AWS Lambda 上构建自定义 serverless MoE 推理系统。
  - 使用 PyTorch + transformers 构建 MoE 模型，Optuna 实现 BO 算法，Gurobi 求解 MIQCP 问题。
  - 修改/设计内容：
    1. 设计了三种 scatter-gather 通信方法（pipelined indirect via S3、non-pipelined indirect via S3、direct function invocation），支持按 MoE 层混合选择。
    2. 实现了基于 Bayes 定理的专家选择预测器，使用 token ID、position ID、attention ID 三个特征计算后验概率。
    3. 实现了 ODS 算法：将 MIQCP 分解为三种通信方法各求解一次，然后逐层选择最低 cost 方法，若不满足延迟约束则迭代替换最高延迟层。
    4. 实现了带 feedback processor 的 BO 框架，根据实际 billed cost 反馈调整 key-value dataset table，优化专家预测准确性。
  - 未修改：MoE 模型权重、gating network 路由决策本身。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未提供独立开源仓库（无公开代码链接）。代码实现基于 PyTorch、Optuna、Gurobi 在 AWS Lambda 上构建。
  - 全过程示例（GPT2 MoE，10240 tokens，Enwik8 dataset，text generation task）：
    1. **离线 Profiling**：在 ≥100 个样本上运行 MoE 推理，记录每个 (token_id, position_id, attention_id) → expert_index 映射的出现频次，构建 key-value dataset table Ω。
    2. **BO 训练循环（~20 iterations）**：
       a. **Expert Selection Predictor**：对每个新 token，提取 token ID f1（已知）、position ID f2（均匀分布）、attention ID f3（用最高 attention score 对应的 token ID 近似），通过两重积分计算后验概率 P(N_{e,i}|f1') = ∫∫ P*(N_{e,i}|f1',f2,f3) · P*(f1',f2,f3)·P'(f3)/P*(f1',f2) · P*(f1',f2)·P'(f2)/P*(f1') df3 df2；取 argmax 得到预测专家。
       b. **Policy Maker**：将三组固定通信方法的 MIQCP 送入 Gurobi 求解（≤60s）；ODS 算法逐层选择最低 cost 通信方法，若不满足 end-to-end latency limit T_limit 则迭代替换最高延迟层。
       c. **部署执行**：各 expert 函数按配置的内存大小（如热门 expert 3008MB、冷门 128MB）和副本数部署到 AWS Lambda；模型参数从 S3 加载到各函数；gating 网络函数按所选通信方法（如 a^e=1 pipelined indirect）将 token minibatch 写入 S3，expert 函数从 S3 下载并计算，结果写回 S3，下一非 MoE 层从 S3 下载聚合。
       d. **Feedback**：收集 J 个 batch 的实际 billed cost c_τ = (1/J)Σc_{τ,j}，更新 BO 历史集 B_τ；多维度 ε-GS 调整 key-value table：低性能 key-value pairs 在 limited range L 内加大探索（ε 衰减更慢），其余在正常 range P 内探索。
    3. **收敛后部署**：BO 收敛后的最优配置部署 MoE 模型，开始服务真实推理请求。
