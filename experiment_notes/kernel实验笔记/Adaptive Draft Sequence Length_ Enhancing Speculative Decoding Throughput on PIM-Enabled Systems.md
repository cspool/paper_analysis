## Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出arithmetic intensity-aware operator scheduler，在PIM+GPU异构系统上动态调度speculative decoding的DLM FC和TLM attention operator。核心实现：(1) 初始映射：DLM attention每次迭代每请求仅生成1 token，算术强度低→固定映射到PIM；TLM FC因Shared Pool聚合token后变为compute-bound→固定映射到GPU/xPU。(2) DLM FC动态调度：每次prediction后Scheduler识别仍可继续drafting的请求数→估算effective micro-batch size→近似DLM FC算术强度→与预标定的PIM compute-bound和GPU memory-bound阈值比较→决定FC operator在PIM还是GPU执行。当effective batch size从12降至4时，DLM FC算术强度从GPU带宽限制区降到PIM计算限制区，optimal target从GPU变为PIM。(3) TLM attention动态调度：verification前Scheduler统计Shared Pool中每请求draft token数→估算TLM attention算术强度→同理与阈值比较决定PIM或GPU执行。draft length从1增至8时，TLM attention即使memory-bound在GPU上也优于PIM。实验：SADDLE变体Ssaddle-s（含动态operator mapping）相比Ssaddle-p（仅Shared Pool无动态mapping）再提升1.13×吞吐。无operator scheduling时SADDLE 9.51% ops在PIM、90.49%在GPU执行；启用后变为14.89%和85.11%，吞吐提升1.21×。对比baselines GPU-AD/GPU-SD/PIM-AD/PIM-SD，SADDLE平均吞吐提升3.36×/2.88×/1.94×/1.71×。

- 后端平台是什么，配置是什么。
  8个SADDLE PIM devices，每device含1×NVIDIA A100 GPU (80GB HBM2e, peak bandwidth 1555 GB/s) + 5×HBM3 stacks (各16GB, 5.2Gbps/pin)。HBM3 PIM chip每bank附1 PE (16 FP16 multipliers + 16 FP16 adders, 256-bit operands/cycle)，pCH内所有PE跨bank并行。buffer die集成SFU（softmax/layer norm/activation非矩阵运算）。PIM内部带宽144 TB/s (9× DGX system 16 TB/s)。GPU baselines在8×A100 DGX系统(DGX A100)上评估。

- 评估性能的软件/脚本是什么。修改了什么。
  构建cycle-accurate simulator，修改Ramulator2（DRAM simulator）和ATTACC（PIM accelerator simulator）来模拟GPU systems和SADDLE。输入系统配置和模型规格，输出execution time和energy consumption。PE面积/能耗通过Synopsys Design Compiler 28nm 1GHz综合并缩放到DRAM process评估。HBM energy参考prior work的activation/read energy值。GPU baselines (GPU-AD/GPU-SD)使用DeepSpeed Inference在A100 DGX上评估，PIM baselines (PIM-AD/PIM-SD) 基于AttAcc HBM-PIM架构。benchmark：FP16精度，TLM/DLM组合：Llama3.1-70B+Llama3.2-1B、OPT-66B+OPT-1.3B、OPT-175B+OPT-6.7B。数据集：Dolly instruction-following dataset。BS=16-128，max sequence length=1024 (OPT-175B=512)。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：论文未提供SADDLE simulator开源链接（HPCA 2026）。Ramulator2为开源项目：https://github.com/CMU-SAFARI/ramulator2。ATTACC论文发表于ASPLOS'24。SADDLE operator scheduling使用流程：
  1. 预标定阶段：离线测量每个hardware device的peak compute performance和memory bandwidth→标定PIM compute-bound vs GPU memory-bound阈值线（roofline模型，Fig.6）
  2. Scheduler初始化：DLM attention固定→PIM（每iteration 1 token/request，低arithmetic intensity）；TLM FC固定→GPU（token pooling后compute-bound）；DLM FC和TLM attention标记为dynamic
  3. 每次prediction后：Scheduler统计仍活跃请求数→计算effective micro-batch size→估算DLM FC arithmetic intensity→与预标定阈值比较→若低于PIM compute-bound→remap到PIM；若高于GPU memory-bound→保留GPU
  4. 每次verification前：Scheduler统计Shared Pool每请求token数→估算TLM attention arithmetic intensity→同理remap决定PIM或GPU
  5. 例如：一个micro-batch初始12请求，8个短draft请求先完成→effective batch size从12降到4→DLM FC arithmetic intensity降低→Scheduler将DLM FC从GPU remap到PIM

