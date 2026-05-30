## BLASST Dynamic BLocked Attention Sparsity via Softmax Thresholding

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现两套专用CUDA kernel：prefill kernel（compute-bound优化）和decode kernel（memory-bound优化），均基于FlashAttention-3/4的tiled online softmax pipeline修改。Prefill kernel跳过softmax指数计算和attention-value MMA操作（tensor core），decode kernel跳过Value matrix的HBM加载（memory bandwidth节省）。两个kernel的skip decision通过复用online softmax已有的running maximum和local maximum统计量，仅增加少量指令（warp-level VOTE + ATOMIC to shared memory），零额外开销。
  实验比较：(i) BLASST prefill kernel vs FlashAttention-3 BF16 baseline在B200和H200上的speedup，sparsity从0%到~94%；(ii) BLASST decode kernel vs FlashAttention-3 BF16 baseline在B200和H200上的speedup；(iii) 0% sparsity时的overhead验证（kernel overhead被pipeline隐藏）。

- 后端平台是什么，配置是什么。
  - NVIDIA Blackwell B200 GPU：prefill batch=1, 64K seq len; decode batch=148, 32K seq len
  - NVIDIA Hopper H200 GPU：prefill batch=1, 64K seq len; decode batch=128, 16K seq len
  - 容器化环境：Docker（NVIDIA Container Toolkit）或Singularity，基于nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6 容器镜像
  - 编译：CUDA nvcc，target sm90a（Hopper）和 sm100（Blackwell）

- 评估性能的软件/脚本是什么。修改了什么。
  - 框架：TensorRT-LLM和FlashInfer中集成的BLASST CUDA kernel
  - 评估脚本：自动sweep不同threshold scale factor，测量sparsity、执行时间、memory bandwidth、speedup vs dense baseline
  - 修改：(i) prefill kernel pipeline（Figure 3）：在softmax warpgroup中增加skip check逻辑（predicate + VOTE + ATOMIC），当block被跳过时消除BMM2（MMA）和exp计算，压缩pipeline时间线；(ii) decode kernel pipeline（Figure 4）：改为batched load scheduling——连续发射多个K^TQ计算（back-to-back BMM1），再根据skip check结果批量发射仅需要的V_j加载，消除pipeline bubble；(iii) 两类kernel均支持MHA/GQA/MQA/MLA attention变体。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源集成到TensorRT-LLM（https://github.com/NVIDIA/TensorRT-LLM）和FlashInfer。Artifact repo: https://github.com/cameronshinn/blasst-ae-mlsys26.git（Apache 2.0）。

  评估原理和流程（以Hopper prefill kernel benchmark为例）：
  1. 容器启动：`./start_docker.sh` → 自动pull TensorRT-LLM release容器并挂载repo到/workspace
  2. 进入目录：`cd /workspace/hopper_prefill/` → 阅读README.md中的具体编译和运行指令
  3. 编译kernel：nvcc将BLASST CUDA kernel模板编译为sm90a target的二进制
  4. 自动benchmark脚本：对一系列threshold scale factor进行sweep，对每个factor：
     a. 创建随机初始化tensor（Q/K/V，模拟64K sequence length, batch=1）
     b. 执行BLASST kernel → 收集execution time和memory bandwidth
     c. 执行FlashAttention-3 baseline → 收集baseline执行时间
     d. 执行closed sm100 binary → 测量实际达到的exact sparsity percentage
     e. 计算speedup = baseline_time / BLASST_time
  5. 结果输出到stdout：sparsity vs speedup表（对应论文Table 5）

  Prefill kernel pipeline详解（Figure 3）：
  - FlashAttention: 每轮mainloop迭代包含BMM1(QK^T) → softmax(EX2) → BMM2(PV)，顺序执行，18个time units完成4轮
  - BLASST prefill: 同样执行全部BMM1，但在skip check通过的block中跳过softmax和BMM2。例如loop 1和loop 3被跳过时，pipeline被压缩到14个time units。Schedule中不同tile row（T0/T1）的运算用不同色调标注，skip check、rowsum和softmax scaling被隐藏在BMM1之后。

  Decode kernel pipeline详解（Figure 4）：
  - FlashAttention decode: V load → BMM1(QK^T) → BMM2(PV)，顺序流水线，38个time units完成所有V loads
  - BLASST decode (batched load scheduling): 连续K1^TQ, K2^TQ, ..., K_B^TQ背靠背（BMM1s）→ 批量skip check → 仅发射通过检查的V_j加载。需维护B个S_j的shared memory buffer（因query len=1，开销小）。消除pipeline bubble，31个time units完成。Arrows指示scoreboard dependency：skip check完成后才能决定加载哪些V blocks。
