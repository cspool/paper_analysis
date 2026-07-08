# Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE

Zongpu Zhang\*^, Pranab Dash\*, **Qiang Xu**, Y. Charlie Hu, Jian Li, Haibing Guan

Purdue University | Shanghai Jiao Tong University

*\* equal contribution ^ work done while visiting Purdue*

![](_page_0_Picture_5.jpeg)

### Background

#### **DVFS on mobile**

DVFS governors dynamically adjust hardware **clock frequencies** based on workload demands

**Higher frequency:**

→ Faster, more power, hotter

**Lower frequency:**

→ Slower, less power, cooler

#### **Governors are independent**

- Each operates based on local metric (e.g., util)
- No cross-resource coordination

CPU governor *(e.g., EAS)*

Memory governor *(e.g., interactive)*

GPU governor *(e.g., quickstep)*

![](_page_1_Picture_13.jpeg)

![](_page_1_Picture_14.jpeg)

![](_page_1_Picture_15.jpeg)

![](_page_1_Picture_16.jpeg)

![](_page_1_Picture_17.jpeg)

![](_page_1_Picture_18.jpeg)

### LLM inference on mobile

![](_page_2_Picture_1.jpeg)

![](_page_2_Picture_2.jpeg)

![](_page_2_Picture_3.jpeg)

![](_page_2_Picture_4.jpeg)

**GPU-based inference needs all 3 components**

- GPU handles most kernels
- CPU keeps work queues fed
- Memory bandwidth shapes both prefill / decode stages

**Key question:** Can independent DVFS governors provide optimal **latency-energy** efficiency?

### Research questions

How far from optimal are default mobile governors for LLMs? **Q1:**

> Why (lack of) governor interaction causes energy inefficiency? **Q2:**

> > How can a unified governor coordinate CPU, GPU, memory efficiently? **Q3:**

![](_page_3_Picture_4.jpeg)

### Benchmarking setup

Compare **default governors** against **pinned frequency combinations** to reveal the latency-energy tradeoff.

#### **Platform setup**

![](_page_4_Picture_3.jpeg)

Pixel 7 / 7 Pro, Tensor G2

![](_page_4_Picture_5.jpeg)

![](_page_4_Picture_6.jpeg)

Batteries bypassed Screen disabled

#### **Framework and models**

![](_page_4_Picture_9.jpeg)

Llama.cpp + OpenCL

![](_page_4_Picture_11.jpeg)

TinyLlama 1.1B StableLM 3B Llama-2 7B

#### **Metrics**

![](_page_4_Picture_14.jpeg)

TTFT, TPOT, E2E latency, Energy per token

![](_page_4_Picture_16.jpeg)

### Finding 1: Default governors are far from optimal

![](_page_5_Figure_3.jpeg)

**23.0–40.4%**

**Longer latency** than optimal combinations under the same energy usage

**5.0–16.6%**

**Higher energy** than optimal combinations under the same latency

![](_page_5_Picture_8.jpeg)

#### Finding 2: Individual governors select too low frequencies

![](_page_6_Figure_1.jpeg)

![](_page_6_Figure_2.jpeg)

### Insight: Decode stage exhibits low utilization

![](_page_7_Figure_1.jpeg)

**GPU utilization at various pinned CPU frequencies**

![](_page_7_Figure_3.jpeg)

**CPU utilization at various pinned GPU frequencies**

Low utilization prompts default governors to reduce frequency in trying to bring utilization up

![](_page_7_Picture_6.jpeg)

![](_page_8_Figure_1.jpeg)

![](_page_8_Picture_2.jpeg)

![](_page_9_Figure_1.jpeg)

![](_page_9_Picture_2.jpeg)

![](_page_10_Figure_1.jpeg)

![](_page_10_Picture_2.jpeg)

![](_page_11_Figure_1.jpeg)

**Design implication: avoiding this downward spiral requires a unified CPU/GPU governor**

![](_page_11_Picture_3.jpeg)

### CORE: A unified DVFS governor for LLMs

**Goal**

Optimal frequencies under latency/energy target

**High-level idea**

Offline profile of different frequency combinations (once per model)

#### **Challenges**

- Different prefill lengths -> different frequency
- Too many frequency combinations

#### **Approach**

**● Searching ranges**

5 prefill length settings + 1 decode length setting

**● Two-step heuristic** "GPU first, CPU next"

![](_page_12_Picture_12.jpeg)

### Evaluation on ShareGPT traces

Faster inference at same or lower energy.

![](_page_13_Figure_2.jpeg)

![](_page_13_Figure_3.jpeg)

#### Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE

#### **Takeaway**

- 1. Current DVFS governors are designed to work independently
- 2. LLM inference on mobile GPU requires using all three components
- 3. As a result, default governors are far from globally optimal for mobile LLMs
- 4. Antagonistic effect drives CPU/GPU frequencies to be overly low
- 5. CORE: Simple unified coordination works

![](_page_14_Picture_7.jpeg)