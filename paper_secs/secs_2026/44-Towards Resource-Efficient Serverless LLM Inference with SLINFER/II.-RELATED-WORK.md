# II. RELATED WORK

Heterogeneous serverless computing. Designing serverless systems with heterogeneous hardware [55], [59] offers significant opportunities. Molecule [23] enables serverless computing to run seamlessly across heterogeneous computers, DSCS-Serverless [44] leverages programmable accelerators to unlock the potential of data centers, IceBreaker [60] improves cold-start by mixing heterogeneous instances, and INFaaS [58] reduces costs for serving traditional models by automatically selecting the optimal hardware architecture. In the context of LLM serving, SLINFER also identifies opportunities to leverage heterogeneous hardware effectively.

Traditional and serverless model serving systems. Before the rise of LLM, traditional model serving systems [21], [22], [25], [38], [40]—such as Clockwork [27], Cocktail [28], and SHEPHERD [73]—had introduced numerous optimizations in scheduling and resource management. Among them, BATCH [17], INFless [70], and Dilu [43] explored applying serverless paradigms. However, traditional models differ significantly from LLMs in their resource demands and execution patterns. The latter defines SLOs at token-level and executes in a multi-iteration manner with fluctuating compute/memory demand, necessitating the specialized serving systems.

In response, a wave of LLM-oriented solutions [45], [51], [53], [68], [71] has emerged. vLLM [37] enhances memory efficiency with paged-attention, Llumnix [63] dynamically schedules requests across instances, and SpotServe [46] considers preemptible instances. A series of approaches [16], [35], [54], [75] have been proposed to consider the differences between prefill and decode stages. They primarily focus on highload scenarios with a single LLM. Meanwhile, MuxServe [24] adopts static GPU sharing for multi-LLM serving but relies on predictable workloads, which does not hold in serverless settings with highly dynamic and bursty workloads. Finally, for serverless LLM serving, ServerlessLLM [26], Medusa [72], and ParaServe [42] improve cold-start but still allocate dedicated GPUs to each LLM. SLINFER focuses on resource sharing through elastic allocation and is orthogonal to them.

CPU-assisted LLM inference. Given the scarcity of GPUs, many works [39], [62], [69] explore leveraging CPUs to assist LLM inference. Early systems such as PowerInfer [62] offload infrequently accessed model parameters to the CPU. NEO [32] and FastDecode [29] further offload KV-cache along with the associated attention computations to the CPU, thereby alleviating GPU memory pressure. In these designs, CPUs primarily serve as auxiliary resources, handling lightweight or memory-bound tasks, while GPUs remain the dominant compute devices.

Recently, the emergence of CPUs equipped with matrix acceleration units (e.g., Intel AMX [15], [50]) has reshaped this landscape. LIA [36] demonstrates that AMX-enabled Intel CPUs can deliver matrix multiplication throughput comparable

![](_page_2_Figure_0.jpeg)

![](_page_2_Figure_1.jpeg)

Fig. 2: Popularity of LLMs' size from HuggingFace [5].

Fig. 3: Invocation frequencies of 25 LLMs in LMSYS [74].

![](_page_2_Figure_4.jpeg)

### III. BACKGROUND AND MOTIVATION

### A. LLM Inference Process

In LLM inference, users submit requests containing input tokens, which the inference engine processes iteratively [16], [45], [63], [68]. Each iteration generates one output token, which is streamed back to user in real time.

A request undergoes two stages [53], [54], [75]. The prefill stage occurs during the first iteration, where the engine builds the key-value (KV) cache [14], [37] and generates the first output token. In the decode stage, the engine appends to the KV-cache and generates one token per iteration. To improve concurrency, inference engines adopt continuous batching [71] to dynamically incorporate new requests into ongoing batches.

Interactive LLM serving systems should follow strict Service Level Objectives (SLOs). Two key metrics are Timeto-First-Token (TTFT) and Time-per-Output-Token (TPOT). TTFT is typically constrained to a few seconds [75] and grows with input length, while TPOT should keep up with human reading speed, which is around 250 tokens/min [16].

Once LLM serving meets above SLOs, it can operate as a reliable productivity tool like ChatGPT [7]. Ongoing contributions from open-source communities [64], [65] have further expanded the accessibility and diversity of LLMs.

