## MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：将现有的 Speculative Decoding（SD）算法（standalone draft model 和 Eagle speculation head）应用于稀疏 MoE 模型推理，并通过修改 MoE 模型的 `num_experts_per_token`（K）参数研究 MoE 稀疏度 ρ 对 SD 加速效果的影响。核心创新在于（1）理论分析：推导全激活专家数 N(t) 和每专家平均 token 数 Texp(t;ρ)，证明中等 batch size 下所有专家已激活时 SD 验证不会带来额外参数加载开销，且更稀疏的 MoE 延迟 memory-bound→compute-bound 转变；（2）新指标 target efficiency = T_T(B,1)/T_T(B,γ)，用于解耦系统瓶颈与算法优化；（3）基于 roofline model 的 SD speedup 性能建模（Algorithm 1），通过参数拟合预测任意 workload 下的 SD 加速比。
  - 实验比较：（1）不同 batch size 下 MoE SD speedup 趋势（先升后降，验证理论预测）；（2）不同 sparsity ρ（K=1,2,4,8,16）对 SD speedup 的影响；（3）MoE vs dense model 的 target efficiency 和 end-to-end speedup 对比；（4）不同 γ、temperature、dataset 下的 speedup；（5）性能模型拟合 vs GPU 实测的对比。

- 硬件平台是什么，配置是什么。
  - 2xGPU-A, 2xGPU-B, 4xGPU-A, 4xGPU-C（论文对 GPU 型号做了匿名化处理，GPU-A/GPU-B/GPU-C 为不同 ridge point 的 GPU 平台）。

- 模型是什么。数据集和bench分别是什么。
  - Target 模型：Qwen2-57B-A14B-Instruct（sparsity ρ=8/14），Mixtral-8x7B-Instruct-v0.1（ρ=2/8）。
  - Draft 模型：Qwen2-0.5B-Instruct（standalone small model），Eagle speculation head（trained head integrated in target model）。
  - 稀疏度实验：通过修改 Qwen2-57B-A14B-Instruct 的 config.json 中 `num_experts_per_token` 为 K=1,2,4,8,16 来模拟不同 ρ。
  - Dense 对比模型：Opt-30b（target）+ Opt-350m（draft）。
  - 数据集：HumanEval（code generation）和 MT-bench（conversation）。Tokenized prompt 长度：HumanEval 38-391 tokens，MT-bench 5-356 tokens。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供独立开源代码仓库。实验基于开源框架 vLLM（支持 batched SD、cudagraph optimization）。
  - 算法 pipeline 伪代码：
    1. 输入：B 个 requests 的 prompt tokens，MoE target model M_T，draft model M_D
    2. for each decoding round r = 1, 2, ..., R:
       a. Draft: M_D 自回归生成 γ 个 draft tokens，耗时 γ × T_D(B, 1)
       b. Verify: M_T 并行处理 B × γ 个 (prompt + draft) tokens
          - MoE Gate 路由每个 token 到 K 个 expert
          - N(Bγ) = E × (1 - ((E-K)/E)^{Bγ}) 个专家被激活
          - 若 Bγ 足够大使得 N(Bγ) ≈ E（全激活），验证时间 T_T(B, γ) ≈ T_T(B, 1)
          - 否则 T_T(B, γ) > T_T(B, 1)（额外 expert 参数加载开销）
       c. Rejection Sampling: 基于 target/draft logits 比较丢弃错误预测 token
       d. 本轮接受 token 数 S/R = σ × (γ+1)，σ = (1-α^{γ+1})/((1-α)(γ+1))
    3. Speedup = (S/R) / (γ × T_D(B,1)/T_T(B,1) + T_T(B,γ)/T_T(B,1) + T_reject/T_T(B,1))
    4. Target Efficiency = T_T(B,1) / T_T(B,γ) 作为系统瓶颈度量
    5. 稀疏度调整：修改 config.json 中 `num_experts_per_token` → 影响 ρ 和 N(t) 曲线
