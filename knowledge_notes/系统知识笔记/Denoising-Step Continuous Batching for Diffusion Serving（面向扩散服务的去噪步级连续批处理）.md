## Denoising-Step Continuous Batching for Diffusion Serving（面向扩散服务的去噪步级连续批处理）

术语是什么？通过联网搜索让回答具体和精准。

Denoising-Step Continuous Batching是FlashPS提出的将LLM serving中的continuous batching思想适配到扩散模型图像编辑serving的调度技术。与传统diffusion serving的static batching（请求整批一起开始、一起完成）不同，FlashPS在每个denoising step边界允许已完成请求动态退出running batch，新请求在下一个step边界加入。关键区别于LLM continuous batching：LLM的continuous batching在token级别混合prefill/decode，而扩散模型的continuous batching在denoising step级别（50步去噪过程）进行。FlashPS还进一步将CPU密集的image preprocessing（VAE encode）和postprocessing（VAE decode）拆分到独立进程，避免阻塞GPU denoising主进程。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FlashPS denoising-step continuous batching的系统运转流程：

```
Worker主进程（GPU denoising loop）:
  running_batch = []  // 当前正在执行去噪的请求
  denoising_step = T  // 从T=1000 (纯噪声) 开始
  
  while True:
      // Step 1: 退出已完成请求
      for req in running_batch:
          if req.current_step == 0:  // 完成全部去噪步
              running_batch.remove(req)
              postprocess_queue.push(req)  // → 独立postprocess进程处理
      
      // Step 2: 加入新请求（仅当batch有空间）
      while len(running_batch) < max_batch_size:
          if request_queue.empty():
              break
          new_req = request_queue.pop()
          new_req.current_step = T  // 初始化从首步开始
          running_batch.append(new_req)
      
      // Step 3: 执行一个denoising step（所有batch内请求同步执行）
      for req in running_batch:
          latent_noisy = req.latent
          for each transformer_block:
              if use_cache[block_id]:  // 由Bubble-free DP决定
                  Y = masked_compute + cached_unmasked_Y
              else:
                  Y = full_compute(all_tokens)
              req.latent = Y
          req.latent = scheduler_step(req.latent, noise_pred, req.current_step)
          req.current_step -= 1
      
      // Step 4: 循环到下一步

Preprocessing进程（独立，CPU执行）:
  while True:
      req = incoming_request_queue.pop()
      req.image_latent = VAE.encode(req.image)     // CPU-bound
      req.mask_latent = process_mask(req.mask)       // CPU-bound
      request_queue.push(req)  // → 交给GPU主进程

Postprocessing进程（独立，CPU执行）:
  while True:
      req = postprocess_queue.pop()
      req.output_image = VAE.decode(req.latent)    // CPU-bound
      return_response(req.output_image)             // → 返回客户端
```

与LLM continuous batching的关键差异：
1. **同步执行**：扩散模型在同一个denoising step内，batch中所有请求执行相同层数的transformer blocks。LLM CB则是prefill跨tokens并行、decode逐token串行。
2. **加入/退出粒度**：扩散模型在step边界（而非token边界）进行batch变更。一个新请求需等待当前step完成才能加入，一个完成请求在step完成后立即退出。
3. **Disaggregated pre/post**：LLM serving的prefill是GPU计算（attention over prompt），FlashPS的preprocessing（VAE encode）是CPU操作。拆到独立进程避免了CPU burst对GPU denoising的干扰。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashPS的实现要点：
1. **asyncio + ZeroMQ通信**：scheduler和worker间通过ZeroMQ传递请求信息和状态上报。Worker内部使用asyncio event loop管理request queues和batch状态。
2. **FastAPI前端**：接收HTTP请求（image template、mask、editing conditions），将请求转发给scheduler。
3. **进程模型**：每个GPU上运行一个worker进程（GPU denoising），额外spawn preprocessing和postprocessing子进程。进程间通过multiprocessing.Queue传递数据。
4. **与TetriServe Selective CB的区别**：TetriServe的CB仅对同分辨率请求在step级别batching，且不拆分pre/post process。FlashPS的CB：(a) 不限分辨率（batch内可有不同分辨率/mask ratio请求）；(b) 拆分CPU pre/post为独立进程；(c) batch入口/出口在step边界而非请求边界。
5. **效果**：FlashPS disaggregated continuous batching相比static batching降低P95 latency约35%，相比LLM-style naive continuous batching（不拆CPU进程）降低P95 latency约40%。

涉及论文标题：
- FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling
