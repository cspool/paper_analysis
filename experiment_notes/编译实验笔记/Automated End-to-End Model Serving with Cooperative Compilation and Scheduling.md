## Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

- 属于编译框架的实现是什么？实验比较什么？
  提出Infera compiler，基于TVM 0.16.0实现tile-based zero-tuning DNN编译框架。核心编译实现：(1) Tile-based Graph/Operator Partition：将大型operator切分为micro operators（tiles），tile size与kernel tile size联合决定；将小型operators合并为shepherd operator避免过细调度开销；(2) Multi-version Micro Kernel Generation：为每个micro operator生成多种ILP/TLP/intensity trade-off配置的kernel candidates——register使用上限64/96/128、shared memory上限48/80/112/144 KiB、global-level grid size固定64、reduction axis tile size参数化为kernel argument、pipeline stage设为2/3/4；(3) Warp Specialization：4个mainloop warps+4个shared→register data copy warps固定分配，GPU scheduler将每组4连续warp分发到同一SM的4个SMSP，shared memory异步copy配合pipeline同步；(4) Cut-and-Patch Instruction Scheduling：关闭nvcc优化生成CUDA binary→反汇编SASS→切出mainloop computation segment→list scheduling最小化stall cycles→每64条指令插入yield flag平衡warp进度→插回优化片段。实验比较operator-level speedup和compilation time，对比Ansor、MetaSchedule、Roller、cuDNN。

- 硬件平台是什么，配置是什么。
  Intel Xeon Gold 6330 CPU, 512 GB RAM, NVIDIA A100-PCIE-40GB GPU, Linux 6.1.0, CUDA 12.0。

- 开源编译框架是什么。修改了什么。
  基于TVM 0.16.0。修改包括：(a) tile-based TVM compiler——将ONNX model→TVM Relay→Tile-based TensorIR，实现operator partition和micro operator生成；(b) multi-level code optimizer——在CUDA C++→PTX→SASS多级别做instruction reconstruction、warp specialization、cut-and-patch instruction scheduling；(c) code generator——对global→shared使用asynchronous copy+warp specialization，对shared↔register使用padding消除bank conflict，对register→global使用wide data types (STG.128)和__threadfence()保证一致性；(d) static library consolidation——将所有生成code和数据打包为CUDA binary static library。

- 开源情况。编译框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  未找到明确开源仓库（EuroSys 2026, DOI: 10.1145/3767295.3769392，通讯作者Haipeng Dai主页标注TBD）。Infera compiler使用流程：
  1. 导入模型：`raw_model = inf.import_model(onnx_model)` 将ONNX格式模型导入
  2. 指定目标设备：`target = inf.device.gpu(gpu_id)`
  3. Zero-tuning编译：`rt_model = inf.compile(raw_model, target)` ——编译器自动完成tile partition + micro kernel generation + instruction scheduling，不同kernel编译完全并行化
  4. 注册模型：`model_tpl_id = infrt.register_model(rt_model)` 和 `model_id = infrt.register_param(model_tpl_id, params)` 将编译结果和权重注册到inference server
  5. 例如编译BERT模型：ONNX输入→TVM Relay computation graph→tile-tailored TVM compiler生成TensorIR→TVM code generator生成CUDA C++→multi-level optimization (instruction reconstruction + warp specialization + cut-and-patch)→CUDA binary static library。编译过程中对Gemm/Conv2D/Transpose/AveragePool等算子均生成多版本kernel，Infera compiler kernel性能平均比Ansor/MetaSchedule/Roller/cuDNN至少高5%，编译时间比Ansor/MetaSchedule低2-3个数量级，比Roller节省66%-86% CPU时间
