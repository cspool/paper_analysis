## NVML（NVIDIA Management Library，NVIDIA 管理库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NVML 是 NVIDIA 的 GPU 管理 API（C 库，CLI 前端 nvidia-smi，Python 绑定 pynvml）：查询 GPU 状态（温度、利用率、功率、显存、时钟频率）并设置管理参数（锁频、功率上限、性能模式等），是数据中心运维与电源管理软件的标准接口。PowerWeave 用它下发频率变更："Frequency changes are issued through the NVIDIA Management Library (NVML), which exposes per-GPU clock-setting interfaces"（当前 GPU 只暴露 per-GPU 时钟接口，即设备级——空间 DVFS 在真实硬件上用多 GPU 仿真）。
- 相关工具链：DCGM（NVIDIA Data Center GPU Manager，4.2.2）提供批量遥测与能量测量；PowerWeave 用 DCGM 测能量，并与 DCGM 功率×时长交叉验证（因 [60] 报告 NVIDIA 功率传感器可能不一致，两者接近才保留测量，否则重跑）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 系统架构中的角色：NVML 是"软件控制平面 → GPU 硬件电源管理"的执行通道。运转流程：DVFS Controller 算出每域目标频率 → 经 NVML 时钟接口写入 → GPU 固件执行 V/f 迁移（Blackwell ≈10–100µs）→ 后续 kernel 在新频率下执行。能量侧：DCGM 采样功率/能量 → 系统据此评估节能（PowerWeave 把总能量=多 GPU 能量和−按未分配 TPC 比例扣除的 idle ≈140W）。监控侧：Power Sloshing（ISCA'26）等系统也用 NVML 读 GPU 利用率 u_G 作为负载代理信号（f_G·u_G≈QPS）。
- 局限：NVML 只提供设备级频率接口（无 per-SM/per-TPC 频率域），这是 PowerWeave 必须用多 GPU 仿真空间 DVFS 的根因；也解释了为什么论文专门做硬件面积模型论证"原生多域 DVFS 值得加"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：libnvidia-ml 随 driver 安装，nvidia-smi 命令行可查/设（如 -lgc 锁频、-pl 功率上限）；编程用 NVML API（https://developer.nvidia.com/nvidia-management-library-nvml）或 pynvml（https://pypi.org/project/pynvml/）；数据中心批量遥测用 DCGM（https://docs.nvidia.com/datacenter/dcgm/）。论文场景：PowerWeave 在 interposer 后台线程经 NVML 设频率、DCGM 4.2.2 测能量；类似电源管理论文（Power Sloshing、RPU、Untangling GPU Power）都用 NVML 读功率/利用率。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
