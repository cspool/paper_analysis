# <span id="page-10-0"></span>7 Evaluation Results

This section first introduces the detailed settings for evaluating our proposed ReCA framework (Sec. [7.1\)](#page-10-2), and then benchmarks the performance and efficiency of our proposed ReCA optimization techniques (Sec. [7.2\)](#page-10-3) with sensitivity analysis (Sec. [7.3\)](#page-11-0), demonstrating the practical of efficient and scalable cooperative embodied AI systems.

## <span id="page-10-2"></span>7.1 Experimental Setup

Workloads. We evaluate ReCA on three state-of-the-art cooperative embodied systems, i.e., CoELA [\[86\]](#page-15-6) COMBO [\[87\]](#page-15-9), and MindAgent [\[20\]](#page-13-3). We follow the configurations and parameters reported in their studies to set up baseline systems.

Benchmarks. We evaluate ReCA on six commonly-used cooperative embodied AI benchmarks, i.e., TDW-MAT [\[15\]](#page-13-8), TDW-Cook [\[14\]](#page-13-9), TDW-Game [\[14\]](#page-13-9), CuisineWorld [\[20\]](#page-13-3), C-WAH [\[63\]](#page-15-10), and Minecraft [\[22\]](#page-13-10). These benchmarks include collaborative objective transporting, housework, gaming, etc long-horizon multi-objective planning tasks.

Metrics. To evaluate the performance and efficiency of the ReCA system, we report the average task success rate, number of steps, and end-to-end runtime. The success rate is the percentage of successful trials where a successful mission is achieved when all agents finish their subtasks within upper step limits. The average number of task steps and end-to-end runtime are reported for successful missions.

Network. We assume all embodied agents run on local machines and are interconnected via WAN. Based on realworld infrastructure, we assume a WAN latency of 75 ms and a throughput limit of 100 Mbps.

Hardware. We evaluate the LLM inference runs latency and energy consumption on Nvidia A6000 GPU, measuring power with NVML [\[69\]](#page-15-17). Control latency and power are measured using an Intel i7 CPU. We implement ReCA A-star subsystem on the Xilinx Zynq-7000 SoC ZC706 FPGA [\[80\]](#page-15-18) to assess real hardware performance.

