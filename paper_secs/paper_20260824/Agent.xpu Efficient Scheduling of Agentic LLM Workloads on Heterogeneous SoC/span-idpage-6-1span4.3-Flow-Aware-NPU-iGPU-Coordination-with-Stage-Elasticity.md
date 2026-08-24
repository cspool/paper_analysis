# <span id="page-6-1"></span>4.3 Flow-Aware NPU-iGPU Coordination with Stage-Elasticity

Agent.xpu principally decouples prefill and decode into specialized scheduling pipelines, facilitating the effective coordination of agentic flows. Prefill primarily leverages the NPU for compute-intensive token-wise ops, while decode resides on the iGPU to support dynamic batching and sequence growth. Such heterogeneous disaggregation is elastic to fit the real-time states of on-going reactive and proactive flows. 1 Prefill Pipeline. For reactive requests, we minimize TTFT by partitioning token-wise ops across NPU and iGPU with elastic tensor parallelism (details in 4 ). For proactive requests, token-wise ops run mainly on NPU, while dynamic ops (e.g., MHA) and residual fragments alongside chunked kernels run on iGPU. Because a single request already saturates either accelerator, prefill avoids batching and processes requests serially. This pipeline also serves as the substrate for the preemption mechanism of reactive requests ([§4.4\)](#page-7-0).

- 2 Decode Pipeline. The decode pipeline adopts continuous batching decoupled from prefill, scheduling requests at iteration granularity. Reactive and proactive requests can be co-batched under adaptive strategies ([§4.5\)](#page-8-0). Executing all decode kernels on the iGPU naturally accommodates dynamic sequence lengths and fluctuating batch composition.
- 3 Pipeline Coordination and iGPU Arbitration. On shared memory SoCs, prefill and decode share the in-place KV cache without costly cross-accelerator transfers. However, both stages may contend for the iGPU due to common dynamic operators, necessitating arbitration mechanisms. Agent.xpu adopts a *prefill-first* arbitration policy: iGPU kernels from prefill always take precedence over decode, regardless of request type. This design is motivated by: 1) decode is memory-bound and generally longer than prefill, and 2) prefill requires only a small fraction of iGPU kernels, with the bulk of computation offloaded to the NPU. Prioritizing prefill ensures that short bursts of iGPU work complete promptly, avoiding long stalls that would otherwise block the entire prefill pipeline and inflate TTFT. Decode jobs can then efficiently utilize the wide gaps between prefill bursts.
- 4 Elastic NPU-iGPU Tensor Parallelism. To reduce TTFT for reactive requests while mitigating interference with ongoing decode, Agent.xpu elastically partitions reactive prefill kernels across NPU and iGPU at runtime. Decisions are made *layer-wise* by the XPU coordinator in real-time. As detailed in Algorithm [1,](#page-7-1) if decode pipeline is idle, the coordinator solves for *nnpu* such that the NPU execution time for chunks aligns with the iGPU's execution time (assigned chunks + dynamic remainder), minimizing the makespan. Otherwise, the coordinator assigns all chunks to NPU to protect the latencysensitive decode stream on iGPU. The remainder iGPU kernel is deferred to complete no earlier than their parallel NPU counterparts, increasing the chance that decode finishes mid-layer with interference alleviated.
- 5 On-the-Fly NPU Kernel Warm-Up. At runtime, Agent.xpu opportunistically prepares dynamic NPU kernels (e.g., MHA) once a request is queued or preempted, thereby hiding compilation latency and reducing iGPU prefill load. Since prompt length is known at enqueue time, the CPU can start compiling static NPU kernels immediately; if compilation completes before prefill begins, the request switches to pure-NPU prefill, mitigating iGPU interference with decode. Compiled kernels are reclaimed once unused or expired. To eliminate potential contention introduced by NPU-iGPU co-execution, when NPU prefill kernels become memorybound (e.g., MHA with short prompt length) and overlap with memory-intensive iGPU decode, the coordinator prioritizes reactive tasks: if both pipelines share the same priority (all-reactive or all-proactive), they proceed concurrently; otherwise, work with lower priority is deferred until the higherpriority side completes for reactive latency preservation.

#### Algorithm 1 Elastic Kernel Dispatch for Reactive Prefill

```
Require: Reactive request r with input tokens x (shape: Slen(r)×dmodel),
  layer l, prefill buffer bu fp and decode status Dstatus
Ensure: Completed execution of layer l during prefill of r.
  if l is the first layer then ▷ Fill prefill buffer with r's input
     memcpy(bu fp, x, Slen(r)· dmodel ·sizeof(x.dtype))
  end if
  for each op ∈ GetOperators(l) do
     if op is token-wise (QKV Proj, FFN) then
        if Dstatus == IDLE then ▷ Scenario 1: Maximize parallelism
           nnpu ← argmin0≤i≤Nchk (r)
                                 |i·Top(Schk,NPU)−
                   (Nchk(r)−i)·Top(Schk,iGPU)−Top(Srem(r),iGPU)|
        else ▷ Scenario 2: Conservative iGPU usage
           nnpu ← Nchk(r)
           tde f er ← nnpu ·Top(Schk,NPU)−Top(Srem(r),iGPU)
        end if
        for i = 0,...,nnpu −1 do ▷ Non-blocking
           LaunchKernelAsync(op,bu fp[i · Schk : (i+1)· Schk],NPU)
        end for
        for i = nnpu,...,Nchk(r)−1 do ▷ Non-blocking
           LaunchKernelAsync(op,bu fp[i · Schk : (i+1)· Schk],iGPU)
        end for
        Sleep(tde f er) ▷ Defer iGPU kernel
        LaunchKernel(op,bu fp[Nchk(r)· Schk : Slen(r)],iGPU,
               preempt = true) ▷ Preempt (probable) decode kernels
        SyncExecution(op,NPU,iGPU)
     else ▷ Sequence-wise (MHA)
        LaunchKernel(op,bu fp[0 : Slen(r)],iGPU,preempt = True)
     end if
  end for
```

