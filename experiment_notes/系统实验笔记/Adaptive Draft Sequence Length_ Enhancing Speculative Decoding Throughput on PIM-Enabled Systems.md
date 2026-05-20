## Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

- 属于Serving调度的实现是什么？实验比较什么？
  提出SADDLE，面向PIM+GPU异构系统的adaptive draft sequence length speculative decoding系统，核心Serving调度实现包含：(1) 运行时自适应draft length调整：Controller在DLM每生成一个draft token时读取采样概率p_t，维护累计接受概率H_t=∏p_i，当H_t低于离线校准的阈值τ时停止该请求drafting，避免生成低置信度token被TLM拒绝后浪费计算；(2) Shared Pool跨micro-batch聚合draft tokens：各micro-batch的Draft Generator将draft tokens存入Shared Pool (1KB CAM)，当token数达GPU verification capacity C或GPU空闲时触发TLM并行验证，避免micro-batch内因最长draft请求同步等待；(3) Eager Pool乐观执行：TLM正在验证Shared Pool中token时，DLM基于"当前token将被接受"的乐观假设继续生成后续draft tokens暂存到Eager Pool (1KB，按micro-batch划分)，验证通过后迁入Shared Pool否则丢弃并以TLM修正token重新开始；(4) prediction-verification解耦异步pipeline：DLM prediction与TLM verification并行执行，通过Shared Pool/Eager Pool token migration消除同步气泡。实验比较throughput、energy efficiency、latency breakdown、communication cost、GPU/PIM utilization、area overhead，对比GPU-AD (autoregressive on GPU)、GPU-SD (speculative on GPU)、PIM-AD (PIM attention+GPU FC autoregressive)、PIM-SD (SpecPIM类PIM-enabled speculative decoding)。在OPT-66B+OPT-1.3B、Llama3.1-70B+Llama3.2-1B、OPT-175B+OPT-6.7B三组TLM/DLM组合上，SADDLE相比GPU-AD/GPU-SD/PIM-AD/PIM-SD平均吞吐分别提升3.36×/2.88×/1.94×/1.71×。消融实验：(Ssaddle-d)仅自适应draft length反而比PIM-SD低1.22×；(Ssaddle-p)+Shared Pool比Ssaddle-d高1.52×；(Ssaddle-s)+Eager Pool+动态operator mapping进一步提升1.24×和1.13×。异步pipeline实现端到端延迟降低1.73×（与PIM-SD对比），monitoring/decision-making仅占0.83% latency。

- 硬件平台是什么，配置是什么。
  SADDLE系统含8个SADDLE PIM devices，每device配1个NVIDIA A100 GPU (centralized processor) + 5个HBM3 stacks (每stack 16GB, 5.2Gbps/pin)，总GPU显存640GB、总HBM 640GB，A100 DGX聚合带宽16TB/s，PIM内部带宽144TB/s (9× DGX)。GPU baselines在8×A100 DGX上评估(DeepSpeed Inference)。PIM baselines用相同数量GPU和40 HBM stacks (各16GB)。SADDLE Manager配1KB Shared Pool (CAM)、1KB Eager Pool (每micro-batch最多512 tokens)、1KB SRAM (存logits和累计接受概率)。Controller集成softmax unit、multipliers和comparators。HBM3 PIM chip每bank附1个PE (16 FP16 multipliers + 16 FP16 adders)，buffer die上集成SFU(softmax/layer norm/activation)。互联：NVLink或CXL。

- 开源Serving框架是什么。修改了什么。
  GPU baselines基于DeepSpeed Inference实现。PIM baselines基于AttAcc风格HBM-PIM架构，PIM-SD采用SpecPIM的离线静态operator mapping（基于初始batch size和max sequence length的design-space exploration）。SADDLE自身通过cycle-accurate simulator (修改Ramulator2 + ATTACC) 评估，非基于开源Serving框架修改。SADDLE核心新增：(1) Draft Generator per micro-batch (Controller + Eager Pool)；(2) Shared Pool跨micro-batch聚合验证；(3) Controller的H_t累积概率计算和阈值比较hardware module；(4) Scheduler的arithmetic intensity估算和动态remapping逻辑。模型权重和KV cache按pipeline parallelism分配到S组PIM devices，batch切成> S个micro-batches占满pipeline。KV cache mapping：每attention head分配一个HBM stack，K^T column-wise partitioning across BGs + row-wise across banks，V row-wise across BGs + column-wise across banks。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：论文未提供SADDLE代码或模拟器开源链接（HPCA 2026）。引用Dolly dataset开源仓库 https://github.com/databrickslabs/dolly 仅用作评测数据集。SADDLE speculative decoding scheduling使用流程：
  1. 离线阶段：用验证集校准阈值τ——对每个请求运行完整prediction-verification pipeline，记录每draft step j的H_j和验证结果，估算H_j上的条件成功率曲线，选20%区间内平均draft length最高且≥90%验证成功率的τ
  2. 推理启动：batch切成micro-batches，每个micro-batch分配Draft Generator。DLM逐token生成draft→Controller读取p_t更新H_t→若H_t<τ则停止该请求drafting→draft token存入Shared Pool
  3. 当Shared Pool token数≥GPU capacity C(=512)或GPU空闲时→TLM并行验证所有Shared Pool tokens（跨micro-batch聚合）。同时DLM对H_t仍高于τ的请求继续生成新token→暂存Eager Pool
  4. TLM验证返回：若请求所有旧draft tokens被接受→Eager Pool中该请求新tokens迁入Shared Pool→进入下一轮verification。若有token被拒绝→该请求Eager Pool tokens全部丢弃→用TLM修正token重开drafting
  5. Scheduler在prediction后根据活跃请求数估算DLM FC算术强度→决定GPU或PIM执行；在verification前根据Shared Pool每请求token数估算TLM attention算术强度→决定GPU或PIM执行
  6. 以OPT-66B+OPT-1.3B, Dolly, BS=64为例：SADDLE自适应draft length+异步pipeline相比PIM-SD固定d=8时吞吐提升1.71×，有效避免了固定draft length在batch增大时因大量draft token被拒绝导致的吞吐下降

