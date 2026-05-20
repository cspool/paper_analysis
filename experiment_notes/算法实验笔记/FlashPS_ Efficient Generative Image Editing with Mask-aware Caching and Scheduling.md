## FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

- 属于算法pipeline的实现是什么？实验比较什么？
  属于算法pipeline的实现是FlashPS的mask-aware cached activation计算pipeline：将扩散模型transformer block内token按mask划分为masked/unmasked两类，对attention和feed-forward等token-wise计算只对masked tokens执行，unmasked tokens的输出直接复用预先缓存的block输出activation Y（非K/V cache）。同时用DP在block粒度动态决定哪些transformer block使用cache加载+masked-only计算、哪些block直接全量计算以消除pipeline bubble。实验比较baseline：Diffusers（标准全图生成）、FISEdit（mask sparsity，仅SD2.1）、TeaCache（通用activation reuse）。指标包括端到端延迟、图像质量（CLIP/FID/SSIM/用户研究）、P95尾延迟。

- 硬件平台是什么，配置是什么。
  SD2.1: NVIDIA A10 GPU。SDXL和Flux: NVIDIA H800 GPU。在线服务评测使用单台8-GPU机器，每个worker分配1张GPU。SD2.1最大batch size=4，SDXL/Flux最大batch size=8。

- 模型是什么。数据集和bench分别是什么。
  模型：Stable Diffusion 2.1 (SD2.1)、Stable Diffusion XL (SDXL)、Flux。数据集与benchmark：InstructPix2Pix（图像编辑）、VITON-HD（虚拟试穿）、PIE-Bench（图像编辑benchmark）。生产trace：2025年1月14天trace，覆盖20k GPU、3400万张生成图像、970个templates（平均各复用约35k次），平均mask ratio为0.11。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源地址：https://github.com/Sylvia-16/FlashPS；Zenodo: https://zenodo.org/records/17176576。提供Docker镜像jiangxiaoxiao/flashps。
  FlashPS mask-aware算法pipeline：
  1. 图像和mask编码为latent，将latent reshape后的空间位置视为transformer tokens，按mask划分masked/unmasked tokens。
  2. 对于linear projection、feed-forward、LayerNorm、activation等token-wise计算：unmasked token的输出直接从cached activation（存储在host memory/disk/distributed storage）读取，masked tokens执行完整计算。
  3. 对于attention：观察到unmasked token在输出Y上跨请求高度相似，且masked/unmasked token间cross-attention较弱，因此缓存transformer block输出Y中unmasked tokens的activation（非K/V cache），减少cache footprint。
  4. Bubble-free DP：对每个transformer block比较"加载cache后仅计算masked tokens的完成时间"与"全量计算完成时间"，用O(N) DP为每个block决定是否使用cache——避免在cache loading比直接计算慢的block上产生pipeline bubble。
  5. CUDA stream异步加载：cache load stream从host memory异步加载cached activations到GPU HBM，computation stream并行对masked tokens执行attention/feed-forward；block输出时将masked token计算结果与cached unmasked activations合并为完整Y送入下一block。
  6. 理论speedup约为1/m（m为mask ratio），实际mask ratio=0.2时SD2.1/SDXL/Flux加速比分别为1.3x/2.2x/1.9x。SSIM最高达0.99。
