## Expert Pinning（专家钉选/固定）

术语是什么？
Expert Pinning 是 ES-MoE 的 adaptive offloading 优化：将一小部分 hot experts 永久保留在 GPU 显存（不参与 offload），其余 experts 继续动态 offloading。动机：expert 增多时 per-expert compute time 下降，若所有 expert 每次上传，upload time 占比上升导致 pipeline overlap 效果减弱。Pinning hot experts 减少总 upload volume，延长可 overlap 的计算窗口。

从kernel调度角度拆解术语：
每个 iteration：读取上一 iteration 的 per-expert token counts → 排序 → 选择 top 25% → 固定到 GPU → 其余 experts 参与动态 offloading + dynamic placement。MoE-M 32 experts + 4 GPUs: 25% pinning → 22.8% 吞吐量提升（vs no pinning）。

术语一般如何实现？如何使用？
实现为 adaptive offloading 控制器子模块。利用相邻 iteration 间 expert load 的 temporal locality 选择 pinning target。适用于 expert 数量较多场景（>32 experts）。

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
