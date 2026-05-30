## CoE Runtime / Dynamic Expert Model Switching（CoE 运行时 / 动态专家模型切换）

术语是什么？
CoE Runtime 是部署 Composition of Experts 系统的专用运行时层，负责管理多个独立编译的 expert 模型在异构内存（HBM + DDR）之间的动态切换。类似操作系统的动态链接器/加载器，CoE Runtime 在接收到推理请求时：(1) 检查目标 expert 是否已在 HBM 中（cache hit/miss）；(2) 若 miss，从 DDR 拷贝 expert 的 HBM 段到 HBM（带宽 >1 TB/s on SN40L 8-socket）；(3) 若 HBM 空间不足，按 LRU 策略淘汰最久未使用的 expert（跳过 read-only weight 的 DDR 回写以节省带宽）；(4) 激活 expert 并执行推理；(5) 完成后返回控制权等待下一请求。每个 expert 模型独立编译，二进制文件预先声明 HBM/DDR 空间需求，运行时由 CoE Runtime 动态分配 DDR 块并管理生命周期。与 GPU 上通过 PCIe 走 host DRAM（32-64 GB/s）的传统模型切换相比，加速器直连 DDR 的切换延迟降低 15×-31×。

从系统架构角度拆解：
CoE Runtime 调度流程：
```
请求到达 → Router(HBM)推理 → CoE Runtime检查HBM cache
  ├─ hit → 直接执行expert(HBM)
  └─ miss → HBM空间检查
       ├─ 空间充足 → DDR→HBM拷贝expert → 执行expert
       └─ 空间不足 → LRU淘汰旧expert(skip read-only回写)
            → DDR→HBM拷贝新expert → 执行expert
```
关键设计：(1) 编译器标注 read-only 符号（如模型权重），运行时跳过这些符号的 DDR 回写；(2) 静态符号生命周期分析实现内存复用（无指针别名 → 编译器可确定符号生命周期 → 重叠符号共享地址）；(3) 每个 expert 独立管理生命周期：开发→编译→训练→微调→量化→部署→服务→共享均可独立进行。

术语一般如何实现？如何使用？
实现需要：(1) 编译器支持符号级 read-only 标注和生命周期分析；(2) 底层驱动提供 DDR/HBM 的动态内存分配接口；(3) 运行时维护 HBM 中驻留 expert 的 LRU 链表；(4) 每个 compiled model binary 携带 HBM/DDR 需求元数据。使用时，应用层通过 CoE Runtime API 提交 prompt batch，Runtime 透明处理 expert 调度和切换，应用开发者无需感知 DDR/HBM 差异。

涉及论文标题：
- SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts
