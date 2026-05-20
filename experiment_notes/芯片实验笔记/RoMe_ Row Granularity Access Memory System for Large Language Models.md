## RoMe: Row Granularity Access Memory System for Large Language Models

- 属于芯片设计的实现是什么？实验比较什么？
  提出HBM4 RoMe chip-level优化，主要包含三项芯片设计改动：(1) HBM channel expansion：利用RoMe简化接口后每通道C/A pin从18降为5（节约13 pins/channel），将省下的416 pins聚合后增加4个额外channels，使HBM cube从32 channels扩展到36 channels，仅需额外12 pins，总带宽从2 TB/s提升到2.25 TB/s（+12.5%）。每DRAM die的channel数从8扩到9。(2) Logic die command generator placement：每个legacy channel对应一个command generator放置在logic die上。HBM4 logic die使用logic process（非DRAM process），36个command generators总面积约4268.8µm²，仅占logic die 0.003%。(3) µbump/TSV overhead：额外4个channels需要48个额外µbumps（经conservatively scaling 4× per channel），按22µm µbump pitch估算约0.14mm²。包含edge margin后DRAM die area增约12%，logic die面积同比例增加，total area overhead仅0.10%。实验比较：(1) 额外channel面积开销估算；(2) command generator面积和能耗开销；(3) 多channel下channel load balance和effective bandwidth utilization。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  面积分析：基于HBM3E specifications [34]，22µm microbump pitch [62]，conservative 4× scaling per channel for TSV/µbumps [62][77]。Command generator和MC scheduler使用Verilog + Synopsys Design Compiler 7nm [9]综合。系统级模拟使用LLMSimulator + Ramulator 2.0。论文未提供芯片设计专用模拟器开源链接。

- 模拟器模拟什么的性能，修改了什么。
  芯片面积分析：输入HBM3E die spec（DRAM die area, logic die area, per-channel µbump count, µbump pitch）→计算额外4 channels的µbump和TSV数量→加上edge margin→估算DRAM die和logic die面积增长→total area overhead = 0.10%。Command generator面积：Verilog实现→7nm综合→单generator面积×36 = 4268.8µm²→占logic die ~0.003%。论文还评估了不同VBA设计方案（3种bank group消除方式×2种PC消除方式=6种组合）的面积trade-off：采用Figure 7(d)（两个不同BG的bank组成VBA）+ Figure 8(b)（两个PC并发）避免加倍BK-BUS/BG-BUS/I/O buffer宽度，使面积开销最小。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：论文未提供芯片设计相关代码开源仓库。面积估算基于HBM3E公开规格和22µm µbump pitch公开数据。使用流程（基于论文描述）：
  1. Channel expansion面积估算：HBM4 baseline 32 channels × 120 pins/ch = 3840 pins。RoMe每channel 107 pins（120−13省下的C/A pins）。32 channels × 107 = 3424 pins，剩余416 pins → 416/107≈3.89 → 增加4 channels（多出12 pins）。每channel需~12 µbumps per data lane × 4× conservative factor = 48 µbumps → 22µm²×48 ≈ 0.023mm² × 6 rows ≈ 0.14mm² total。
  2. Command generator面积：Verilog RTL of RD_row/WR_row to ACT/RD/PRE sequence generator → Synopsys DC 7nm综合 → 单个118.6µm² × 36 = 4268.8µm² → 占logic die 0.003%。
  3. 性能验证：在Ramulator 2.0+LLMSimulator中配置36 channels → sweep batch size 8-1024 × seq_len 8K × 3 models → 验证12.5% bandwidth提升转化为9-10.4% TPOT improvement。

