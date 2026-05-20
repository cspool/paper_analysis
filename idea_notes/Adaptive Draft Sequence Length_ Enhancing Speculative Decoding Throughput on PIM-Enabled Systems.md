## Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

- baseline方法是什么？
  Baseline是SpecPIM类PIM-enabled heterogeneous speculative decoding系统（PIM-SD）。PIM-SD在HBM-PIM+GPU异构系统上运行speculative decoding：
- (1) DLM (如OPT-1.3B) 在PIM/GPU上自回归生成固定长度d=8的draft tokens；
- (2) TLM (如OPT-66B) 并行验证所有draft tokens；
- (3) operator mapping通过离线design-space exploration基于初始batch size和fixed draft length确定，推理中不改变；
==batch中不同请求的draft可能提前结束，不同请求的实际负载动态。==
- (4) DLM prediction和TLM verification严格串行执行。
![[Pasted image 20260519215246.png]]

  全栈执行例子（以OPT-66B+OPT-1.3B, Dolly dataset, batch_size=64, d=8为例）：
  - 算法层：standard speculative decoding with fixed draft length d=8。DLM (OPT-1.3B) autoregressive生成8个draft tokens→TLM (OPT-66B) parallel verification with rejection sampling。
	  - **当d=8时acceptance rate从d=4的~0.6降至~0.4(Fig.4a)，大量draft token被拒绝后丢弃，浪费DLM生成和TLM验证的计算**。
	==d不能盲目拉大==
  - 系统框架层：PIM-SD采用**静态operator mapping**，DLM prediction→TLM verification串行执行。
	  - 每轮speculative iteration：所有请求先等DLM生成d=8 tokens（**micro-batch同步屏障**），再统一TLM验证。
	  - batch内请求间draft长度相同无bubble，但**固定长度**导致整体吞吐在BS=64时反而低于autoregressive baseline (Fig.3a)。
	==忽略不同请求的动态负载，统一固定draft-length。==
  - 编译框架层：论文未明确说明。
  - kernel调度层：PIM-SD离线分析后固定映射：DLM attention→PIM, TLM FC→GPU。当effective batch size和draft length变化时映射不变。
	  - 例如当batch内部分请求提前完成drafting后effective batch size降低，DLM FC算术强度降低（从GPU带宽限制转为PIM计算限制，Fig.6），但operator仍固定映射在GPU上执行，导致suboptimal utilization。
	==draft动态，TLM的实际batch填不满GPU。==
  - 硬件架构层：AttAcc风格HBM-PIM架构：每bank 1 PE，bank-level并行。PIM-SD的Manager==无runtime adaptive control==，仅执行**离线确定的mapping schedule**。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出SADDLE，针对PIM-SD的三大缺陷分别设计解决方案：

  **缺陷1：固定draft length → 生成大量被TM拒绝的无效token，浪费计算和带宽**
  → **SADDLE方案**：运行时自适应draft length。Controller每生成draft token时读取采样概率p_t→维护累计接受概率H_t=∏p_i→当H_t<阈值τ（离线用验证集校准，选20%区间内平均draft length最高且≥90%验证成功率的τ）时停止该请求drafting。简单请求（高p_t token）自动获得更长draft，复杂请求（低p_t token）更早停止。运行时可根据系统负载动态调节τ：轻负载降低τ允许更长draft→提升并行度。

  **缺陷2：DLM-TLM串行执行 + 自适应draft长度引入的同步bubble → 请求间等待加剧延迟**
  → **SADDLE方案**：prediction-verification解耦异步pipeline + Shared Pool + Eager Pool。(a) Shared Pool跨micro-batch聚合draft tokens：各micro-batch不再单独等待所有请求完成DLM prediction，draft tokens逐生成即存入Shared Pool，当token数达GPU capacity C(=512)或GPU空闲时触发TLM并行验证；(b) Eager Pool乐观执行：TLM验证Shared Pool时，DLM基于"当前token将被接受"假设继续生成后续tokens暂存Eager Pool，验证通过后迁入Shared Pool，被拒绝则丢弃；(c) 异步重叠prediction和verification，消除串行pipeline的idle time。

  **缺陷3：算术强度动态变化 → 静态operator mapping suboptimal**
  → **SADDLE方案**：arithmetic intensity-aware operator scheduler。(a) predication后根据仍活跃请求数估算DLM FC有效micro-batch size→计算arithmetic intensity→与预标定PIM/GPU阈值比较动态remap；(b) verification前根据Shared Pool每请求token数估算TLM attention arithmetic intensity→同理动态remap。初始固定映射：DLM attention→PIM (低强度)、TLM FC→GPU (高强度的GEMM)。动态remap使SADDLE中14.89% ops在PIM、85.11%在GPU执行（vs 无scheduling时的9.51%/90.49%），吞吐提升1.21×。

  论文方法全栈执行例子（以OPT-66B+OPT-1.3B, Dolly, BS=64为例）：
  - 算法层（核心创新）：**自适应draft length**。每请求dynamically调整draft长度：DLM生成token x_t时获取p_t=DLM(x_t|x_{<t})→更新H_t=H_{t-1}·p_t→若H_t<τ=θ则停止该请求drafting。H_t基于DLM自身采样概率，无需额外训练或分类器。
	  - 相比baseline固定d=8时acceptance rate ~0.4，SADDLE自适应停止在H_t低于阈值时，每个请求的draft length在[1, optimal]区间动态变化，减少无效token生成。
	==按照采样概率自适应停止draft，设置在硬件controller。==
  - 系统框架层（核心创新）：异步pipeline。batch切成micro-batches→每micro-batch有独立Draft Generator。
	  - 请求#0 (simple task) H_t始终>τ→持续draft→tokens入Shared Pool。
	  - 请求#1 (complex task) H_t在第3 token后<τ→停止drafting。
	  - 不等待请求#1继续：Shared Pool累计token数达C→TLM并行验证所有已存tokens。
	  - 同时请求#0在TLM验证期间继续生成新tokens→Eager Pool暂存→验证通过后migrate到Shared Pool。
	==设置shared-pool跨micro-batch进行draft-token验证，eager-pool管理验证时生成的draft-token。==
  - 编译框架层：论文未明确说明。
  - kernel调度层（核心创新）：==动态operator mapping==。prediction后Scheduler统计仍活跃请求数→估算DLM FC effective batch size→与roofline阈值比较→决定FC在PIM或GPU执行。verification前Scheduler统计Shared Pool每请求token数→估算TLM attention arithmetic intensity→同样动态remap。例如当请求#1停止drafting后effective batch size降低→DLM FC arithmetic intensity降到PIM compute-bound区→Scheduler将DLM FC从GPU remap到PIM执行。Operator mapping随每speculative iteration动态调整。
==DLM-Attn固定PIM，TLM-FC合并跨请求token计算而固定GPU，DLM-FC（token和共享权重）根据活跃请求的总token动态调度，TLM-Attn（请求内token之间）根据单个请求内的token动态调度。==
```
def schedule_after_prediction(requests):
    active = count(req.H >= tau for req in requests)
    CI_dlm_fc = estimate_dlm_fc_CI(active)

    if CI_dlm_fc in PIM_preferred_region:
        map("DLM_FC", "PIM")
    else:
        map("DLM_FC", "GPU")


def schedule_before_verification(shared_pool):
    tokens_per_req = shared_pool.tokens_per_request()
    CI_tlm_attn = estimate_tlm_attention_CI(tokens_per_req)

    if CI_tlm_attn in GPU_preferred_region:
        map("TLM_Attention", "GPU")
    else:
        map("TLM_Attention", "PIM")

```
  - 硬件架构层（核心创新）：==SADDLE Manager硬件==。Controller以专用硬件（softmax unit + multipliers + comparators）低延迟计算H_t并比较τ（仅占end-to-end latency 0.83%）。Shared Pool (1KB CAM)和Eager Pool (1KB) 的token migration为lightweight on-chip memory operation，每verification iteration后刷新无容量压力。SFU在buffer die加速softmax/layer norm等非矩阵运算。PE沿用HBM-PIM design (16 FP16 MACs/bank)，面积overhead仅13.4% DRAM die。
![[Pasted image 20260519224724.png]]