论文标题：A Deadlock-Free Bridge Module for Inter-Chiplet Cache-Coherent Communication in an Open Chiplet Ecosystem

解决chiplet之间的cache一致性协议中的死锁。

开源仓库确认：
    - 状态：未找到明确开源仓库
    - 链接：N/A
    - 说明：本地 PDF、HPCA 2026 官方议程页面、DBLP / DOI 记录均能确认论文与作者信息；论文正文说明实现了 gem5/Garnet 多 chiplet 模拟器和 Verilog 原型，但未在正文、官方 HPCA 页面或面向该标题/DFBM/CVN-DB 的 GitHub 检索中发现明确官方仓库或 artifact 链接。因此当前只能确认论文存在实现与原型，不确认其开源。

1、论文工作：
    - 论文要解决的核心问题：
      这篇论文解决的是开放 chiplet 生态中的 inter-chiplet deadlock 问题。单个 chiplet 内部的 2D NoC 即使已经是 deadlock-free，把多个 chiplet 通过 active interposer 互连后，chiplet NoC 与 interposer NoC 之间仍可能形成跨芯粒的 cyclic channel dependency，导致包在垂直通道、边界路由器和片内 NoC 之间互相等待。问题的关键不只是传统 NoC 的 latency / throughput / area / power，而是开放生态要求：不同供应商的 chiplet 应该能在不了解彼此内部 NoC 实现细节的情况下直接互连。
    - 论文的主要贡献：
      第一，论文把 cache coherence protocol transaction flow 与 NoC packet routing dependency 建立映射，用一致性事务之间的因果关系预测 inter-chiplet packet 后续会触发多少响应包或转发包。第二，提出 Deadlock-Free Bridge Module（DFBM），作为 chiplet NoC 与 interposer NoC 之间的独立桥接模块，通过 packet injection control 隔离 intra-chiplet traffic 和 inter-chiplet traffic，从而避免跨 chiplet CDG 形成环。第三，提出 Cross Virtual Network Deadlock Buffer（CVN-DB），让多个 virtual network 共享 deadlock buffer，并按 VN 依赖优先级控制入队顺序，降低专用 per-VN deadlock buffer 的面积开销。第四，论文实现了 gem5/Garnet 多 chiplet 模拟器和 Verilog 原型，用 synthetic workloads、PARSEC full-system workloads、OpenSMART 生成的 RTL 与 EDA 工具评估性能、面积和功耗。
    - 论文所处背景：
      UCIe 等 die-to-die 标准推动了开放 chiplet 生态，但接口标准本身不能保证互连网络在拓扑组合后仍然 deadlock-free。已有 inter-chiplet deadlock 处理方法通常把假设放在 chiplet 内部 NoC 上，例如需要特定转向限制、VC 划分、权限网络、逃逸路径或 bubble 恢复逻辑。论文认为这种“需要内部 NoC 知识或修改”的方式破坏了开放生态的模块化目标：集成方不能任意复用供应商 chiplet，供应商也可能需要为每种互连 floorplan 修改 NoC。

2、相对 Baseline 解决的问题与设计方法：
    - Baseline 的具体问题：
      MTR 使用 boundary router turn restriction 来打破跨 chiplet CDG 环，成本低，但会限制垂直通道选择，造成 vertical channel load imbalance，并且依赖 TSV / interposer wiring layout，portable 性差。DeFT 通过 upward / downward traffic 的 VC isolation 保证 deadlock freedom，但要求每个 virtual network 至少有 2 个 VC，增加路由器资源和集成成本。RC 使用片内 permission network 做 packet injection control，能隔离 inter- 与 intra-chiplet packet，但需要在 chiplet 内部 NoC 加专用控制网络，增加验证复杂度，并产生 vendor lock-in。UPP、Steered Bubble 等 recovery 方法允许 deadlock 先发生再恢复，需要检测逻辑、escape channel 或 directional bubble routing，也会引入架构修改和验证负担。共同缺陷是：它们把 deadlock 处理与 chiplet 内部 NoC 实现绑定，不能满足开放生态中“plug-and-play chiplet”的目标。
    - 论文的设计方法：
      DFBM 被放在 chiplet NoC 与 interposer NoC 的边界，作为独立 bridge module。它包含两个核心模块：Credit Management（CM）和 Cross-VN Deadlock Buffer（CVN-DB）。CM 根据一致性事务类型查询 Expected Credit Table，预测一个 incoming request 或 forward request 可能触发的 passive response 数量，并在允许包进入 chiplet 前预留足够 buffer credit。CVN-DB 则在输出端拥塞时接收可能造成堵塞的包，并按照 VN 依赖优先级（Response > Forward-Request > Request）仲裁共享 buffer。这样 DFBM 能保证 chiplet 到 interposer 的出口方向在最坏情况下仍有足够吸收能力，避免 backpressure 传播回片内 NoC 并形成跨域环。
    - 方法如何对冲 Baseline 缺陷：
      相比 MTR，DFBM 不依赖转向限制，因此不固定垂直通道选择，对非均匀 vertical channel distribution 更稳健。相比 DeFT，DFBM 不要求片内 NoC 为 upward / downward traffic 做额外 VC 隔离，而是在边界模块中管理 credit 和 buffer。相比 RC，DFBM 继承了 injection control 的 deadlock avoidance 思路，但把控制逻辑外置到 bridge，不要求供应商改片内 permission network；同时它不是持续节流，而是根据 expected credits、CVN-DB occupancy 和 congestion 状态动态节流，低负载下延迟更接近原生路径。相比 recovery 类方法，DFBM 是 avoidance 机制，不依赖死锁检测后再打断环。
    - 关键 trade-off：
      DFBM 的代价是边界处需要额外硬件：CM、Expected Credit Table、CVN-DB、priority management、credit tracking，以及在协议非确定性场景下可能需要 dummy packets 来固定 request-response 上界映射。论文用 CVN-DB 把 dedicated per-VN buffer 的面积开销从约 5% 降到约 2.5%，但共享 buffer 在极端拥塞下仍可能带来轻微性能损失。另一个 trade-off 是 DFBM 不要求知道片内 NoC 拓扑细节，但仍要求 chiplet 供应商或系统集成方提供少量协议参数，例如 coherence protocol state machine 依赖、cache controller 最大 outstanding request 数、NoC VC 数量等。

3、论文实现：
    - Baseline 如何实现：
      论文在 gem5 与 Garnet 中构建多 chiplet 模拟环境，对比三类 state-of-the-art avoidance baseline：MTR（turn-restriction based）、DeFT（virtual channel isolation）和 RC（injection control）。评估拓扑中，四个 homogeneous chiplet 通过 shared interposer 互连；chiplet 与 interposer 都采用 4x4 mesh，XY routing，3 个 VN，每个 VN 配置 2 或 4 个 VC，flow control 使用 virtual cut-through。Synthetic workloads 包括 Uniform-Random、Transpose、Bit-Rotation；full-system 使用 x86 out-of-order cores、MESI Two Level coherence、PARSEC benchmark。
    - 新设计如何实现：
      DFBM 连接 chiplet NoC 和 interposer NoC 两侧已有 VC 接口，不额外引入专用控制线。CM 维护两类 credit：Pre-Allocated Credits 用于 chiplet 主动发出的 Out-Req，数量根据 cache controller 最大 request 数预先协商；Reserved Credits 用于由外部请求触发的 passively generated packets，如 Out-Rsp 和 Out-Fwd-Req，数量与 NoC VC / expected response 需求相关。CM 首先从 coherence transaction type 解码 expected credit value，再结合 CVN-DB buffer occupancy 判断是否允许 In-Req 或 In-Fwd-Req 进入 chiplet。CVN-DB 共享 deadlock buffer，并用 VN priority 控制不同 VN 包的入队顺序，高优先级 response 先被 drain，低优先级 request 在保留空间足够时才进入。
    - 实验 / 实现平台：
      性能评估基于 gem5 和 Garnet。面积与功耗评估使用 OpenSMART 生成 NoC RTL，再通过 EDA 工具分析。论文还用 Verilog prototype 评估 DFBM 的硬件代价。系统设置包含 uniform 与 non-uniform vertical channel distribution 两类 floorplan / link 场景：uniform 情况下四个 chiplet 使用 16 条 vertical channels，non-uniform 情况下使用 12 条且分布不均，用来模拟 floorplan constraints 或 link faults 导致的边界路由器负载不均。
    - 关键实验设置与指标：
      Synthetic 评估关注 latency 与 saturation throughput。VC 数增加时所有算法吞吐提升；在高 VC 配置下，vertical channel 数量成为主要瓶颈，算法差距缩小；在低 VC 配置下，DeFT 因 VC 被方向隔离而降低有效 VC 利用率，MTR 因转向限制加剧 vertical channel 负载不均，吞吐下降。RC 和 DFBM 维持较自由的 VC / vertical channel 选择，在 uniform synthetic traffic 下相对 DeFT/MTR 有约 14% 更高 saturation throughput。Full-system 的 PARSEC 结果显示，在 uniform vertical channel distribution 下，DFBM 相对 MTR 延迟降低 1%-7%，平均约 3%；在 non-uniform 条件下延迟降低 1%-4%，平均约 2%。面积方面，MTR 最低；DeFT 至少 2 VC 的要求比 MTR 多约 48% 面积；RC 在 chiplet 内引入约 1.9% 面积开销；DFBM 若使用 dedicated per-VN deadlock buffer 约 5% 面积开销，采用 CVN-DB 后约 2.5%，且开销位于成本较低的 interposer 侧。Sensitivity study 说明 CVN-DB 容量与 VC 数需要匹配：VC=4 时容量变化影响较小，VC=2 时 buffer 容量不足会明显限制延迟和吞吐；共享 buffer 相比 dedicated buffer 只有边际性能下降。Dummy packet 占总流量比例较小，且只在 chiplet-DFBM interface 内局部传输，因此延迟和带宽影响有限。

4、pipeline/kernel 解析：
    - 新 pipeline/kernel 是什么：
      论文没有提出软件 kernel 或 GPU kernel，而是提出一条硬件边界运行路径：DFBM transaction-aware packet regulation pipeline。它由 CM 的 two-stage flow control 和 CVN-DB 的 priority-based shared buffering 组成。CM 的 Stage 1 是 coherence information extraction：从包中提取 coherence type / VN / address 等信息，并查询 Expected Credit Table 得到该事务可能触发的最大后续包数。Stage 2 是 admission arbitration：结合 pre-allocated credit、reserved credit 和 CVN-DB occupancy，决定包是通过、阻塞，还是在输出拥塞时进入 CVN-DB。CVN-DB 则按 Response > Forward-Request > Request 的优先级，把共享 deadlock buffer 的容量保留给更靠近依赖链尾部的包，从而按方向消解依赖。
    - 新 pipeline/kernel 的执行流例子：
      以一个外部 chiplet 发来的 In-Req 为例。第一步，In-Req 到达 DFBM，CM 读取其 coherence type，并在 Expected Credit Table 中查询该请求可能触发的 Out-Rsp / Out-Fwd-Req 数量上界。第二步，CM 向 CVN-DB 查询当前 buffer occupancy，并计算 reserved credit 是否足够吸收这些后续 passive packets。第三步，如果 credit 不足，CM 暂时阻塞这个 In-Req，避免它进入 chiplet 后生成无法被 DFBM 吸收的响应，从而避免 backpressure 进入片内 NoC；如果 credit 足够，In-Req 被放行进入 chiplet NoC，并消耗对应 reserved credit。第四步，chiplet 内部 cache controller 根据 MESI state machine 处理请求，可能产生 Out-Rsp 或 Out-Fwd-Req。第五步，这些响应包返回 DFBM 输出端；若 interposer 方向空闲，包直接发出；若输出端拥塞且超过阈值，包进入 CVN-DB。第六步，CVN-DB 根据 VN priority 先保证 response 类高优先级包有空间和出队机会，再允许 forward-request / request 类低优先级包进入。第七步，包成功离开 DFBM 后，CM 更新 credit。整个流程的核心效果是：DFBM 在让请求进入 chiplet 之前就为其后续响应预留边界吸收能力，保证 chiplet-to-interposer vertical channel 不成为不可释放的依赖点。
    - 一个主动请求的执行流例子：
      对 chiplet 内 core/cache 主动发起的 Out-Req，DFBM 无法从外部输入预测其产生时间，因此 CM 使用 pre-allocated credit 管理。Out-Req 进入 DFBM 时消耗该 cache controller 预先协商的 credit，并被允许发向 interposer；当对应 In-Rsp 返回 DFBM 并进入 chiplet 后，CM 释放该 credit。这个路径把主动请求数量限制在可吸收范围内，避免大量 outstanding requests 在 interposer 侧造成资源耗尽。
    - dummy packet 在流程中的作用：
      对某些 coherence state transition，同一个输入事务可能产生 0 到 K 个响应，甚至 Data 与 ACK 的组合不固定。为了让 Expected Credit Table 使用固定上界，论文引入 dummy packet，把非确定性响应数规整到协议最大 K。这样 DFBM 能用保守 credit 规则保证 worst-case absorption。代价是增加少量局部包流量，但论文的 Fig. 17 表明 dummy packets 在 PARSEC 工作负载中占比较小。
