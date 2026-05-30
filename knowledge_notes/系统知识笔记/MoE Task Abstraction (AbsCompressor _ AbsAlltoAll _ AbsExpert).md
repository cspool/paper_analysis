## MoE Task Abstraction (AbsCompressor / AbsAlltoAll / AbsExpert)

术语是什么？
MoE Task Abstraction 是 ScheMoE 提出的一套模块化任务抽象接口，将 MoE layer 训练中的三类关键操作封装为独立的可替换模块：**AbsCompressor**（提供 compress/decompress 接口，支持 ZFP/FP16/INT8 等压缩算法）、**AbsAlltoAll**（提供 all_to_all 接口，支持 NCCL-A2A/1DH-A2A/2DH-A2A/Pipe-A2A 等通信算法）、**AbsExpert**（封装 expert fflayer 计算，主要用于 Profiler 测量和 Scheduler 调度）。

核心价值：解决了现有 MoE 系统（Tutel、Faster-MoE）的扩展性问题——这些系统中调度算法与 A2A 实现紧耦合，新增一种 A2A 算法或压缩方法就需要重新设计调度逻辑。ScheMoE 通过统一任务队列 + Profiler + Scheduler 的解耦设计，使新算法只需实现抽象接口即可自动享受 OptSche 最优调度。

从系统架构角度拆解术语：

```
// C++ 抽象基类 (示例)
class AbsCompressor {  // abstract_compressor.h
public:
    virtual Tensor compress(const Tensor&) = 0;
    virtual Tensor decompress(const Tensor&) = 0;
};
class AbsAlltoAll {  // abstract_comm.h
public:
    virtual void all_to_all(const Tensor&, Tensor&, size_t) = 0;
};

// 用户自定义实现:
class ZFPCompressor: public AbsCompressor {
    Tensor compress(const Tensor& t) override { return zfp_compress(t, 8); }
    Tensor decompress(const Tensor& t) override { return zfp_decompress(t); }
};
class PipeAlltoAll: public AbsAlltoAll {
    void all_to_all(const Tensor& in, Tensor& out, size_t sz) override {
        launch_intra_stream(in, out, sz);
        launch_inter_stream(in, out, sz);
        sync_both();
    }
};

// Python 使用:
moe = ScheMoE.MOELayer(compressor=ZFPCompressor, alltoall=PipeAlltoAll, **config)
// moe 可作为普通 nn.Module 使用
```

术语一般如何实现？如何使用？
使用 C++ 虚基类（~1200 行核心代码），用户通过继承并实现纯虚函数接入。已内置：AbsCompressor → FP16/INT8/ZFP；AbsAlltoAll → NCCL-A2A/1DH-A2A/2DH-A2A/Pipe-A2A。第三方库依赖：ZFP（github.com/LLNL/zfp）、NCCL、Hetu、Tutel。

涉及论文标题：
- ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling
