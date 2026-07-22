# GPreempt: GPU Preemptive Scheduling Made General and Efficient（GPU抢占优化）

# BG

> **[图片提取文字 (image.png)]:**
> ## **GPU Application**
> 
> GPUs have become essential for a wide range of computing tasks
> 
> - Computer vision
> - Machine learning
> 
> ![](_page_0_Picture_4.jpeg)
> 
> \*
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ![](_page_0_Picture_7.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image.png)

> **[图片提取文字 (image.png)]:**
> ## **GPU Sharing**
> 
> GPU applications can be divided into two categories
> 
> - ❖ Latency-Critical (LC) Applications
>   - ❖ Real-time recommendation
>   - Virtual reality
>   - **\*** .....
> 
> - ❖ Best-Effort (BE) Applications
>   - Scientific computing
>   - Graph computing
>   - Offline inference
>   - \*\* ...
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%201.png)

> **[图片提取文字 (image.png)]:**
> ## **GPU Sharing**
> 
> ## GPU applications can be divided into two categories
> 
> - Latency-Critical (LC) Applications
>   - ❖ Real-time recommendation
>   - Virtual reality
>   - **\*** .....
> 
> ![](_page_0_Figure_6.jpeg)
> 
> - ❖ Best-Effort (BE) Applications
>   - Scientific computing
>   - Graph computing
>   - Offline inference
>   - •
> 
> Co-locate different task to improve GPU utilization
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%202.png)

> **[图片提取文字 (image.png)]:**
> ## GPU Sharing is not common
> 
> GPU programming paradigm is not suitable for scheduling
> 
> Launch-and-wait model
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%203.png)

> **[图片提取文字 (image.png)]:**
> ## GPU Sharing is not common
> 
> Interference between tasks, and potentially *violating SLA guarantees*If we co-locate *multiple task* in....
> 
> 1: Multiple streams of same context
> 
> ![](_page_0_Picture_3.jpeg)
> 
> 2: Different context
> 
> BE kernels LC kernel
> 
> Streams provide no preemption LC kernels will be blocked
> 
> Task execution may be split into multiple fragments
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%204.png)

# RW

幂等kernel是重置GPU来杀死运行中kernel，kernel重启后结果不变，重置GPU所以没有等待延迟。

随机化算法/模拟就不是幂等kernel。

> **[图片提取文字 (image.png)]:**
> ## Common paradigm: Wait-based preemption
> 
> - Check before launch a kernel (kernel level)
> - \* Check and exit if LC tasks arrive at the beginning of each block (block level)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> Preempt when a kernel finishes Latency: a kernel's lifetime
> 
> ![](_page_0_Picture_5.jpeg)
> 
> Preempt when running blocks finish Latency: a block's lifetime
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%205.png)

> **[图片提取文字 (image.png)]:**
> ## Common paradigm: Reset-based preemption
> 
> - ❖ Directly kill the BE kernel on the GPU
> - \* Require *Idempotent kernels*, which produce same result in repeated runs
> 
> ![](_page_0_Picture_3.jpeg)
> 
> Exit anywhere
> 
> Latency: Near-zero
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%206.png)

> **[图片提取文字 (image.png)]:**
> ## Tradeoff between existing approaches
> 
> Existing approaches make a tradeoff between generality and efficiency
> 
> - ❖ Wait-based: Different kernel/block execution time, leading to unstable latency
> - ❖ Reset-based: Limited generality due to idempotence requirement
> 
> ![](_page_0_Figure_4.jpeg)
> 
> ![](_page_0_Figure_5.jpeg)
> 
> ![](_page_0_Figure_6.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%207.png)

> **[图片提取文字 (image.png)]:**
> ## Key idea: Switch-based preemption
> 
> Can we break the tradeoff of generality and efficiency
> 
> - ❖ Generality: We shouldn't drop the context of existing task
> - Efficiency: We shouldn't wait for the kernel / blocks to finish
> 
> Learn a lesson from the OS, context switch offers a solution
> 
> ![](_page_0_Figure_5.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%208.png)

# Method

**更细时间片**的切换保证效率，**硬件机制**时间片和**kernel-level**保证通用性；

**time slice设置**

> **[图片提取文字 (image.png)]:**
> ## Challenge: GPU scheduling is opaque to users (1)
> 
> How we use the GPU?
> 
> - We just *launch kernels* and *synchronize* to complete
> - GPU runtimes provide no scheduling interface
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%209.png)

> **[图片提取文字 (image.png)]:**
> ## Observation: Context switch is inherent in GPU
> 
> When we directly run two tasks on the same GPU?
> 
> - ❖ The GPU runs two tasks alternately in a timeslice round-robin manner
> - ❖ When there are only one task, the GPU will continually run it
> 
> ![](_page_0_Picture_4.jpeg)
> 
> Context switch is an inherent feature in the GPU, which can be leveraged to preempt the task
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ## GPreempt: Timeslice-based Preemption
> 
> Repurpose the GPU task rotation mechanism to achieve preemption?
> 
> ❖ GPU switch between task according to a predefined timeslice
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ## GPreempt: Timeslice-based Preemption
> 
> Repurpose the GPU task rotation mechanism to achieve preemption?
> 
> - ❖ GPU switch between task according to a predefined timeslice
>   - ❖ Set the BE task timeslice to the minimum, almost 0
>   - Expand the timeslice of the LC task beyond its lifetime
> - ❖ By modifying the driver code, we are able to change the timeslice
> 
> ![](_page_0_Figure_6.jpeg)
> 
> - 1. Switch is guaranteed by the hardware
> - 2. Adjusting timeslice needs no modification to the userkernel code
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2012.png)

**context切换开销**

> **[图片提取文字 (image.png)]:**
> ## Challenge: High preemption overhead (2)
> 
> ## Preemption overhead comes from
> 
> - Minimal timeslice for the BE task
>   - ❖ The minimal GPU timeslice is about 200us, LC task will wait 100us on avg
> - Massive compute contexts make switch expensive
>   - \* E.g. 44MB context of NVIDIA A100 GPU take 40us to save
> 
> ![](_page_0_Figure_6.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ## Observation: GPU computing paradigm
> 
> ## How we *offload a task to GPU*?
> 
> - Preprocessing data (often on CPU)
> - Transfer data to GPU before launch kernels (GPU DMA/copy engine)
> - Launch kernels (GPU compute engine)
> 
> ![](_page_0_Figure_5.jpeg)
> 
> Not all the procedures are done on GPU
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ## GPreempt: Hint-based Pre-preemption
> 
> Data preparation offers predictive signals for imminent LC kernel arrival
> 
> ❖ How can we preempt BE before the LC kernels launch?
> 
> ![](_page_0_Picture_3.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2015.png)

LC的preparation和BE覆盖，之后切换

> **[图片提取文字 (image.png)]:**
> ## GPreempt: Hint-based Pre-preemption
> 
> Preparation offer predictive signals for imminent LC kernel arrival
> 
> ❖ How can we preempt BE before the LC kernels launch?
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2016.png)

不如左侧？

> **[图片提取文字 (image.png)]:**
> ## GPreempt: Hint-based Pre-preemption
> 
> Preparation offer predictive signals for imminent LC kernel arrival
> 
> ❖ How can we preempt BE before the LC kernels launch?
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2017.png)

LC来时，启动不需要保存context的empty kernel

> **[图片提取文字 (image.png)]:**
> ## GPreempt: Hint-based Pre-preemption
> 
> Preempt in advance, but not too much
> 
> ❖ The empty kernel will occupy the GPU and prevent BE task execution
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2018.png)

仍然浪费GPU

> **[图片提取文字 (image.png)]:**
> ## GPreempt: Hint-based Pre-preemption
> 
> Scheduled pre-preemption: Prepare in advance, but not too much
> 
> Use a background thread to launch the preemption kernel
> 
> ![](_page_0_Picture_3.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2019.png)

> **[图片提取文字 (image.png)]:**
> ## GPreempt: Hint-based Pre-preemption
> 
> ## **Scheduled pre-preemption:** Prepare in advance, but not too much
> 
> Use a background thread to launch the preemption kernel
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2020.png)

LC来时，延迟一会启动empty kernel

# Evaluate

**工作负载：多个DNN infer、不同分布的请求队列（平稳、泊松、真实）**

> **[图片提取文字 (image.png)]:**
> **Workloads.** As detailed in Table 1, we evaluate GPREEMPT using DISB [13] and two synthetic workloads. DISB contains 6 typical DNN inference workloads, including LC-task with uniform distribution (A, B, C, D), poison distribution (E) and real-world trace (REAL). Synthetic workloads include scientific computing [23] (Y), and graph computing tasks [21] (Z), which are co-located with LC DNN inference tasks.
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ## **Evaluation Setup**
> 
> ## Workloads
> 
> - ❖ DISB: 6 typical DNN inference workloads
> - ❖ Scientific computing: Weather simulation
> - Graph computing: 4 typical graph workloads
> 
> Stateful, Non-idempotent
> 
> | WL          | LC Tasks | LC Rate | BE Tasks | BE Rate |
> |-------------|----------|---------|----------|---------|
> | A           | VGG      | 100 (C) | ResNet   | 100 (C) |
> | В           | VGG      | 220 (C) | ResNet   | 220 (C) |
> | C           | VGG      | 100 (C) | DM*      | 100 (C) |
> | D           | DM*      | 20 (C)  | DM*      | 100 (C) |
> | E           | DM*      | 20 (P)  | DM*      | 100 (P) |
> | <b>REAL</b> | DM*      | Trace   | DM*      | Trace   |
> | Y           | VGG      | 100 (C) | WS       | 150 (C) |
> | Z           | VGG      | 100 (C) | GM**     | 100 (C) |
> 
> DM: {ResNet, DenseNet, VGG, Inception, BERT}
> 
> GM: {BFS, SSSP, PageRank, CC}.
> 
> (C): Constant rate, (P): Poisson rate. "100 (C)": 100 requests/sec per task.
![image.png](GPreempt%20GPU%20Preemptive%20Scheduling%20Made%20General%20an/image%2022.png)