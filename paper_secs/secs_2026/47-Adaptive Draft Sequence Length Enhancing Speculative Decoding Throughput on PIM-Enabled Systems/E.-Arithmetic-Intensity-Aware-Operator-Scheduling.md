# *E. Arithmetic Intensity-Aware Operator Scheduling*

With dynamic draft lengths and cross-micro-batch verification, the arithmetic intensity of each operator fluctuates at runtime due to varying draft lengths and batch sizes (Figure 6). Additionally, concurrent execution of the target and draft models further impacts operator mapping. To address these dynamics, we now establish an initial execution mapping scheme based on the system's average workload profile.

The peak compute performance and memory bandwidth of each hardware device are known in advance. Based on this information, the Scheduler initially assigns the DLM's attention operators and the TLM's FC operators to fixed hardware resources. For the DLM, each iteration generates exactly one token per request, resulting in low arithmetic intensity for its attention operations, which are therefore mapped to PIMs. For the TLM, each iteration verifies a variable number of tokens per request. By pooling tokens before verification, the FC operations become consistently compute-bound and are scheduled onto xPUs.

In contrast, whether the DLM's FC operations and the TLM's attention operations are scheduled to PIMs or xPUs is determined dynamically.

After each prediction, the Scheduler identifies which requests are eligible for additional DLM iterations and uses this to determine the effective micro-batch size. Fluctuations in the effective micro-batch size directly affect the arithmetic intensity of the DLM's FC operator. The Scheduler quickly approximates this intensity and compares it against pre-characterized thresholds—PIM compute-bound and GPU memory-bound—to decide whether the operator should be remapped.

Before the verification stage begins, the Scheduler also counts the number of draft tokens per request in the Shared Pool to estimate the arithmetic intensity of the TLM's attention operator. It then applies the same decision process as for the DLM's FC operator to determine the most suitable execution engine.

At runtime, the Scheduler dynamically remaps operators in response to variations in micro-batch size and draft length, enabling the system to consistently select the most efficient execution engine under any workload [51], [52].

