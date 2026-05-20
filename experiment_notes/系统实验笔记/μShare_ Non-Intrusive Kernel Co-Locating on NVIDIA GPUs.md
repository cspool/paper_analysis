## μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

- 属于Serving调度的实现是什么？实验比较什么？
  提出μShare，一个非侵入式GPU kernel co-location inference serving系统，通过在Linux userspace拦截kernel launch并动态调整blocksize，实现不同kernel在同一SM内的scattered co-location。核心Serving调度实现包含：(1) Kernel Interceptor：通过LD_PRELOAD劫持CUDA kernel launch函数（cudaLaunchKernel、cuBLAS/cuDNN封装函数），使用dlopen+dlsym获取原始函数地址和参数，在传递参数前修改blocksize；(2) Block Shaper：对modifiable kernel设置half-plus blocksize（A40: 800=768+32, 即1536/2+32；A800: 704=683+32, 即2048/3+32），使同kernel的block不能堆叠在同一SM内，剩余线程供其他kernel block使用；对unmodifiable kernel（cuDNN/cuBLAS/tiling kernel）使用time-shifted launching——根据profiled resource utilization延迟启动kernel使其与SM上互补资源需求的kernel co-locate；(3) Batch Manager：feedback-based自适应batch size，每个time window结束后计算SLO slack s→j = Σ 2^(1-i) * (tSLO - ti)，positive slack线性增加batch（bj+1 = bj + k×s→j），negative slack指数减少batch（bj+1 = max{bj - e^(λ×s→j), 1}）；(4) Profiler：离线分析每个kernel的9-tuple资源特征{rFP32, rFP64, rINT32, rLDST, rSFU, rTensor, rmem, rreg, tLaunch}。实验在co-located multi-model serving场景比较μShare vs INFless (ASPLOS'22) 和Orion (EuroSys'24)。指标：system throughput (QPS)、normalized throughput (QPS/unit batch)、SLO violation rate（box plot, 20次重复实验）、end-to-end latency (CDF)、6种low-level hardware unit utilization timeline (Nsight Compute)、A800 GPU可移植性。额外对比Tacker (kernel fusion intra-SM) 和CUDA Graph优化。消融实验：μShare shape 1024（固定blocksize）、μShare w/o shape（无blocksize调整）、μShare w/o batch（无batch size反馈）。不同unmodifiable kernel比例（100%/89.67%/79.35%/69.02%/58.70%/48.37%）下的throughput伸缩。Co-locate scientific computing workload (Parboil benchmark) + inference models。

- 硬件平台是什么，配置是什么。
  8台服务器，每台配备Intel Xeon Gold 6338 CPU（128逻辑核，2.00GHz base/3.20GHz max, 251GB memory）+ (A) NVIDIA A40 GPU（84 SMs, 44.784GB memory, 每SM 1536 threads/102,400B shared memory/65,536 registers, CUDA 11.8）或 (B) NVIDIA A800 GPU（108 SMs, 80GB memory, 每SM 2048 threads/167,936B shared memory/65,536 registers, CUDA 12.1）。Inference framework: PyTorch 2.2.0。

- 开源Serving框架是什么。修改了什么。
  基于PyTorch 2.2.0作为GPU推理框架，μShare组件编译为.so shared libraries通过LD_PRELOAD加载到PyTorch进程。修改内容：(1) Kernel Interceptor：使用libdl的dlopen()打开CUDA动态链接库（libcudart.so/libcublas.so/libcudnn.so）→dlsym()获取原始kernel launch函数地址→创建同名函数通过LD_PRELOAD先加载→拦截kernel参数（blocksize/gridsize/sharedMem/stream）；(2) Block Shaper：使用shm_open()创建共享内存区域→通过mmap()映射修改共享内存中的kernel参数→kernel_process()接口上传modified blocksize→返还参数到dlsym()获取的原始函数地址恢复执行；(3) 设置PyTorch的C10_LAUNCH_BOUNDS(blocksize)宏与CUDA limit一致（1024）；(4) Batch Manager：基于exponential decay algorithm监控时间窗口内的SLO slack动态调整batch size。基线系统INFless (ASPLOS'22) 使用MPS+memory control实现SM和memory资源的不均匀分配实现inter-SM spatial sharing；Orion (EuroSys'24) 通过控制kernel launch time实现compute-intensive和memory-intensive kernel的耦合共置。两者均采用stacked co-location，不修改kernel blocksize。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确提供开源代码仓库链接（HPCA 2026）。以10模型co-located serving on NVIDIA A40为例说明μShare使用流程：
  1. Offline profiling：部署每个模型（Llama2-7b/GPT-2/Bert/ResNet50-v1.5/MobileNet v2/Swin Transformer/ViT/Yolostiny/Resnet101/EfficientNet B7）→确定满足200ms SLO的最大batch size→Nsight Compute记录6种low-level hardware utilization→Nsight Systems记录kernel launch time→输出每kernel的9-tuple资源profile
  2. 部署：编译μShare的.so文件→设置LD_PRELOAD环境变量→启动PyTorch推理服务（每模型4 replica, 共40 replica分布于8 GPU）
  3. 请求到达：使用Azure INFless production trace→Batch Manager batch多个请求→发送batch到PyTorch→PyTorch sequential launch kernels
  4. Kernel intercept：μShare Kernel Interceptor通过LD_PRELOAD劫持cudaLaunchKernel等函数→读取kernel blocksize/stream参数→计算kernel launch slack（sk = tLaunch - tIntercept）
  5. Half-plus shaping：对前x个slack最小的kernel（|X| = x, x为最小满足前x个kernel的总block数超过SM数的值）→设置blocksize = 800 (A40, half+32)→large block占超过半SM threads→阻止同kernel blocks stacked→剩余threads供small block kernel使用
  6. Time-shifted launch：对kernel set Y，检查6种hardware资源combined utilization ≤ 100%且shared memory/registers充足→满足则直接launch→不满足则delay β=10μs后重检→更新slack并重排→若进入top-x则升级为half-plus shaping
  7. SLO反馈：每个time window结束后计算SLO slack→positive slack linear增加batch size→negative slack exponential减少batch size→保守增加、激进减少
  8. 结果：μShare system throughput 3046 QPS（vs INFless 1722 QPS/Orion 1192 QPS），提升26.90%–54.09%；average low-level hardware utilization 15.10%（vs INFless 10.90%/Orion 9.37%），提升38.53%–61.15%；SLO violation rate 3.35%（vs INFless 2.05%/Orion 1.12%），μShare v7 (k=0.05, λ=-0.2) SLO violation 0.84%同时throughput仍提升19.28%-44.83%

