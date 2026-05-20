## FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

- 属于Serving调度的实现是什么？实验比较什么？
  属于Serving调度的实现包括三部分：(1) 将continuous batching迁移到diffusion model的每个denoising step级别——已完成请求在denoising step后退出，新请求在下一个step边界加入，避免等待整个batch完成；(2) 将CPU密集的image preprocessing/postprocessing拆到独立进程（disaggregated preprocessing），避免打断GPU denoising主进程；(3) mask-aware load balancing——scheduler根据请求mask ratio用离线拟合的线性模型估算computation latency和cache loading latency，选择预计pipeline latency最低的worker路由请求。实验比较baseline：static batching（Diffusers默认）、naive continuous batching（LLM-style，不拆CPU进程）、request-count负载均衡、token-count负载均衡。指标包括端到端延迟、P95尾延迟、throughput、queue time。

- 硬件平台是什么，配置是什么。
  SD2.1: NVIDIA A10 GPU。SDXL和Flux: NVIDIA H800 GPU。在线服务评测使用单台8-GPU机器（每个worker独占1张GPU），SD2.1最大batch size=4，SDXL/Flux最大batch size=8。请求到达按Poisson process生成，不同RPS测试。每条请求的mask ratio按生产trace分布采样。

- 开源Serving框架是什么。修改了什么。
  基于HuggingFace Diffusers / PyTorch，保留FlashAttn等已有优化。前端FastAPI接收请求（image template、mask、输入条件）。Scheduler与workers间使用ZeroMQ通信，continuous batching的request queues和load balance scheduler用asyncio实现。主要修改：(1) worker内实现denoising-step级continuous batching逻辑——在每步去噪后检查已完成请求并退出，从队列拉新请求加入running batch；(2) 拆分出独立preprocessing/postprocessing进程——编码/解码latent不与GPU denoising竞争，通过进程间通信传递数据；(3) scheduler扩展为mask-aware——接收worker状态上报，用离线拟合的线性模型估算各worker的computation latency和cache loading latency，选择pipeline expected latency最低的worker路由。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/Sylvia-16/FlashPS；Zenodo: https://zenodo.org/records/17176576。提供Docker镜像jiangxiaoxiao/flashps，包含Diffusers定制包、scheduler、baseline、scripts和mask_ratio_distribution目录。
  FlashPS Serving调度使用流：
  1. 部署：在每个GPU上启动一个FlashPS worker进程，包含model executor、cache engine和continuous batching逻辑。集群侧启动scheduler进程，通过ZeroMQ与所有worker通信。
  2. 请求到达：FastAPI前端接收请求→scheduler调用mask-aware load balance→选取预计延迟最低且有batch slack的worker→将请求路由到该worker的队列。
  3. Preprocessing阶段：独立进程将输入image和mask编码为latent（VAE encode），结果传递给GPU主进程。
  4. Denoising loop：GPU主进程在每个denoising step组成running batch——对每个transformer block执行mask-aware计算（DP决定哪些block用cache）。已完成所有steps的请求立即退出batch，交postprocessing进程解码为输出图像；新请求在step边界加入running batch。
  5. Postprocessing阶段：独立进程执行VAE decode等操作，生成最终编辑图像返回客户端。
  6. 效果：disaggregated continuous batching相比static batching和naive continuous batching将P95延迟分别降低约35%和40%；mask-aware load balance在高请求流量下相比request-granularity/token-granularity baselines将tail latency降低最多26%。

