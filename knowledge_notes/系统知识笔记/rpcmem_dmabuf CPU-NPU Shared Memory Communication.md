## rpcmem/dmabuf CPU-NPU Shared Memory Communication

术语是什么？通过联网搜索让回答具体和精准。

rpcmem是Qualcomm Snapdragon SoC上CPU与Hexagon NPU/DSP之间共享物理内存的wrapper机制，底层基于Linux kernel dmabuf (DMA Buffer)。rpcmem允许CPU和NPU通过同一物理内存区域交换数据而无需显式拷贝（zero-copy），分配/释放/映射接口由Android vendor library libcdsprpc.so提供。FastRPC是Hexagon SDK的跨处理器RPC设施，用于启动和管理remote NPU session。论文组合使用FastRPC（session管理）+ rpcmem/dmabuf（数据共享），相比默认RPC通信有更低延迟。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
// 初始化
rpc_session = fastrpc_session_create(HEXAGON_DSP)
shm_req = rpcmem_alloc(SYSTEM_HEAP, REQ_SIZE)   // 请求区域
shm_data = rpcmem_alloc(SYSTEM_HEAP, MODEL_SIZE) // 模型数据
npu_thread = create(np_poll_loop)                 // NPU侧轮询线程

// CPU → NPU 请求提交
request.op = OP_GEMM; request.act = npu_va(shm_data_act)
request.wgt = npu_va(shm_data_wgt); request.dims = {M,K,N}
cache_flush(shm_req); cache_flush(shm_data_act)  // 手动flush
request.ready = 1; cache_flush(shm_req)

// NPU侧轮询
while running:
    cache_invalidate(shm_req)
    if shm_req->ready:
        dispatch(request)      // HVX dequant + HMX tile MM
        request->done = 1
        cache_flush(shm_req)

// CPU检查完成
while !request.done: cache_invalidate(shm_req)
```

Snapdragon SoC上CPU↔NPU仅单向cache coherence：CPU写入后NPU不自动invalidate，需手动cache maintenance。论文使用shared memory polling而非RPC降低通信延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

依赖Android vendor libraries (libcdsprpc.so)，通过dlopen/dlsym动态加载避免link-time依赖。使用注意：(1) rpcmem buffer需从特定heap分配；(2) 需要Android系统权限访问vendor RPC service；(3) CPU-NPU coherence需手动cache flush/invalidate；(4) Hexagon NPU 32-bit虚拟地址空间限制buffer总大小(V73上≥3B模型无法运行)。论文在llama.cpp新增hexagon NPU backend，使用rpcmem作为underlying buffer type，复用llama.cpp的memory management。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones
