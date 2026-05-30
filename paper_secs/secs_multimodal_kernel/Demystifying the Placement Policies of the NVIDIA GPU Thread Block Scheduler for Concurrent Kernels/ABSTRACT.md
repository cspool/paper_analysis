# ABSTRACT

In this work, we empirically derive the scheduler's behavior under concurrent workloads for NVIDIA's Pascal, Volta, and Turing microarchitectures. In contrast to past studies that suggest the scheduler uses a round-robin policy to assign thread blocks to streaming multiprocessors (SMs), we instead find that the scheduler chooses the next SM based on the SM's local resource availability. We show how this scheduling policy can lead to significant, and seemingly counter-intuitive, performance degradation; for example, a decrease of one thread per block resulted in a 3.58X increase in execution time for one kernel in our experiments. We hope that our work will be useful for improving the accuracy of GPU simulators and aid in the development of novel scheduling algorithms.

#### Categories and Subject Descriptors

C.1 [Computer Systems Organization]: Processor Architectures; C.1.4 [Processor Architectures]: Parallel Architectures; C.4 [Computer Systems Organization]: Performance of Systems

## Keywords

Concurrent kernels, GPGPUs, scheduling algorithms

## 1. INTRODUCTION

Concurrent kernel execution—i.e., running kernels from separate streams at the same time on the same device has been proposed as a means to improve the utilization of general purpose GPUs [19, 1, 16, 17, 5, 15, 7, 3, 4]. In order to take full advantage of kernel concurrency, the scheduler must make intelligent decisions to efficiently divide the GPU's limited resources among the kernels. Suboptimal decisions by the scheduler can lead to inefficiencies that impact kernel performance. However, characterizing the performance implications of such concurrency is challenging due, in large part, to the black-box nature of NVIDIA's proprietary thread block scheduler.

In this work, we use empirical observations of real hardware to infer the policies of the thread block scheduler on the Pascal, Volta, and Turing GPU microarchitectures. We find, for example, that the scheduler chooses where to assign a thread block based on the local resource availability of the streaming multiprocessors (SMs)—we call this the mostroom policy. In contrast, most literature assumes that the scheduler uses a simple round-robin policy [11, 2, 10]. We define this policy as follows:

The most-room policy dictates that a kernel block will be scheduled to the streaming multiprocessor that, at the time of scheduling, can support the most blocks from the current kernel, with only one block scheduled to that SM at a time. This calculation takes into account each SM's current resource availability, but it does not account for potential resource contention with blocks already on the SM. This policy breaks ties between SMs using a pre-defined devicespecific ordering.

Our observations lead to the following conclusion: the performance of a kernel in a concurrent workload is challenging to predict because the performance depends on factors that are external to the kernel itself. Such factors include (i) the scheduling policies of the thread block scheduler; (ii) the potential for resource contention across myriad hardware resources; and (iii) the impact of possibly unpredictable effects such as kernel launch timing.

In short, this paper makes the following main contributions:

- We characterize the behavior of the hardware thread block scheduler on NVIDIA GPUs under concurrent kernel workloads in Section 4. We introduce the mostroom policy, a previously unknown scheduling policy used to determine the placement of thread blocks on SMs.
- We examine the performance implications of the mostroom policy under concurrent workloads in Section 5. We demonstrate that the policy can result in counterintuitive performance drops with only small changes made to the structure of the concurrent kernels. For example, a decrease of one thread per block resulted in a 3.58X increase in execution time for one kernel in our experiments.
- We highlight the scheduler's impact on concurrent workloads with purpose-built kernels that emulate common classes of general purpose GPU kernels: L1-cachedependent, compute-intensive, memory-intensive, and PCIe-bandwidth-dependent. We found performance differences due to resource contention between kernels and a lack of kernel-level fairness.

Copyright is held by author/owner(s).

