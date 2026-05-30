## SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts

- 属于Serving调度的实现是什么？实验比较什么？
  实现：Samba-CoE运行时系统（CoE Runtime），用于在SN40L上部署和管理含150个experts、超1T参数的Composition of Experts系统。核心机制：(1) 三级存储管理 — Router始终驻留HBM，所有expert权重存储于高容量DDR，按需将当前活跃expert从DDR拷贝到HBM执行（DDR→HBM聚合带宽>1 TB/s）；(2) 动态内存管理器 — 类似传统动态链接器/加载器，每个expert模型独立编译，二进制文件中预先声明HBM和DDR空间需求，运行时由CoE Runtime在DDR中动态分配、按需激活（拷贝HBM段）、执行后回收；(3) LRU淘汰策略 — 尽量保持HBM中同时驻留尽可能多的活跃expert，超过HBM容量时淘汰最久未使用的expert（跳过read-only weight的回写）；(4) 硬件orchestrated kernel launch — AGCUs实现硬件级别的kernel调度，对decode阶段的短kernel消除host软件调度开销；(5) 静态垃圾回收 — 利用SN40L无动态内存分配/无指针别名的特性，编译器进行符号生命周期分析，将非重叠生命周期的符号分配到相同设备虚拟地址。
  实验比较：SN40L Node（8 socket）vs DGX A100（8×A100 80GB）vs DGX H100（8×H100 80GB），在Samba-CoE推理场景下，测量BS=1/BS=8、20/200 output tokens场景的端到端延迟和模型切换时间；以及随expert数量增加（1到150+）系统占用（machine footprint）的变化。DGX上模型切换需经过host DRAM（A100: 32 GB/s, H100: 64 GB/s），SN40L直接DDR→HBM（聚合>1 TB/s）。

- 硬件平台是什么，配置是什么。
  SN40L Node：8个SN40L RDU socket + 1个host x86 CPU。每socket：638 BF16 TFLOPS，64 GiB HBM（1.8 TB/s），最高1.5 TiB DDR（200 GB/s，8 socket聚合>1 TB/s）。模型以tensor-parallel (TP8) 方式映射到8个socket。DGX A100：8×A100 80GB PCIe（HBM ~2 TB/s aggregate），32 GB/s host-to-GPU PCIe带宽。DGX H100：8×H100 80GB（HBM ~3.35 TB/s aggregate），64 GB/s host-to-GPU带宽。DGX上假设全部HBM和host memory可用于权重和KV cache存储。

- 开源Serving框架是什么。修改了什么。
  论文未基于开源Serving框架。SambaNova自研CoE Runtime构建在低层设备驱动之上，是专有软件栈。论文未开源。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未开源。Serving执行全过程：(1) 应用层发出batch请求（如8个prompt）→ CoE Runtime接收；(2) Router在HBM中执行（BS=8），为每个prompt选择对应expert；(3) CoE Runtime检查被选中的experts是否已在HBM中（LRU cache hit）— 若未命中，从DDR按需拷贝expert的HBM段到HBM（跳过已在HBM中的部分）；若HBM空间不足，LRU淘汰旧expert；(4) 每个(prompt, expert)对依次在HBM中的expert上执行自回归解码（硬件orchestrated kernel launch模式，AGCUs编排kernel序列）；(5) 对多token生成（如200 tokens），expert权重在decode循环中被重复读取，充分利用HBM的时域局部性；(6) 完成后返回控制权给CoE Runtime，等待下一请求。
