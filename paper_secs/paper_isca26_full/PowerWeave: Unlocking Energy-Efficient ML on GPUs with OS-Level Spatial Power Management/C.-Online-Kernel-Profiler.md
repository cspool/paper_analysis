# C. Online Kernel Profiler

PowerWeave transparently interposes between off-the-shelf LLM inference servers and the GPU driver, executing their kernels on a shared, spatially partitioned device. Each partition, which we refer to as a frequency domain, operates with its own independent frequency. This allows the system to dynamically and spatially control the frequency of the GPU. To achieve high energy efficiency without violating performance constraints, the system begins with a multi-stage online profiling, coordinated by the *online kernel profiler*, which operates per frequency domain.

As requests are being served, the kernel profiler initially executes all kernel instances at the maximum frequency available in their assigned set of resources, establishing a baseline performance profile under optimal conditions. While doing so, it tracks per-kernel latency and uniquely identifies kernels by their function and launch configuration, so that instances launched with different sequence lengths or batch sizes are recorded as distinct entries. This ensures that input-dependent behavior, such as the varying memory access patterns of attention kernels across sequence lengths, is captured independently rather than averaged into a single generalized curve.

The objective is to determine how each kernel's execution time responds to changes in operating frequency and to derive a frequency-latency function for each kernel. After the profiler establishes a baseline, it begins executing kernels at multiple frequency points within the same partition. This process is lightweight, as kernel execution is relatively short, typically in the range of hundreds of microseconds to a few milliseconds,

allowing the profiler to selectively monitor individual kernels while minimizing its impact on the entire request.

During this phase, new kernels may be observed for the first time. This often happens when an application launches a previously known kernel under different grid or thread block configurations. To handle these cases, the profiler employs a latency predictor that generalizes across configurations of the same kernel family, modeling latency from kernel occupancy and thread-block count, using historical data. Specifically, the latency l of a new kernel is predicted as:

$$l = waves \times \frac{l_{old}}{waves_{old}},$$

where  $l_{old}$  is the kernel latency of an already profiled instance in the same kernel family, while the *waves* of a specific kernel instance are calculated by the total launched blocks, divided by the SM occupancy (or blocks per SM), and the number of SMs allocated to that specific kernel launch. Intuitively, the number of waves corresponds to the number of thread blocks that each SM, working in parallel, must execute in sequence to complete the entire kernel.

By comparing these parameters to past executions under different configurations, the system can predict the latency of unseen kernel variants without exhaustive re-profiling. This approach leverages the regularity of machine learning workloads to enable continuous adaptation across diverse inputs.

#### D. Frequency-Latency Scaling Module

After completing the profiling phase, PowerWeave combines all of the per-kernel scaling curves it has collected into a single per-application model. This procedure relies on a heuristic, based on a first-order Taylor approximation. For a target performance degradation k, the adjusted frequency is:

$$f(k) = \frac{f_{max}}{S}, \text{where } S = 1 + \frac{k}{\sum s \cdot w}$$

Here, w denotes the weight of each kernel, defined as its contribution to the total runtime of the application, while s denotes its sensitivity factor, capturing how sharply the kernel's latency scales with changes in frequency. A higher sensitivity corresponds to a steeper slope on the kernel's frequency-slowdown curve.

Intuitively, the weights define the balance between compute-bound and memory-bound work within the application. As the prefill-to-decode ratio shifts in the workload, the weights shift accordingly, and PowerWeave updates them continuously even after the profiling phase to reflect the current workload composition. The sensitivity factor prevents kernels with low frequency sensitivity from disproportionately pulling the target frequency below the level required by frequency-sensitive kernels. Because sensitivity is a property of the kernel's instruction mix, it remains fixed after profiling. In Section VIII, we demonstrate how sensitivity and live weight updates achieve highly accurate predictions.

Since each tenant exhibits unique workload characteristics, each receives its own model, allowing the system to scale frequencies independently according to individual performance

![](_page_5_Figure_11.jpeg)

Fig. 6: PowerWeave's Governor operation over time.

requirements. This per-application curve enables dynamic adaptation to workload variability, such as fluctuating load intensity or shifting SLOs, without committing the system to a fixed slowdown assumption.

Once this stage is complete, PowerWeave stops profiling and can now start optimizing power consumption. We call this the operating phase. Whenever a kernel completes execution, PowerWeave tracks its completion time. In the scenario where there is a repeated divergence from the estimated kernel execution time, the profiling process restarts. PowerWeave's *profiling-threshold* knob is empirically set at 5%.

#### E. DVFS Controller

In the operating phase, PowerWeave relies on its DVFS Controller to modulate the GPU's frequency. The controller takes as input an application-level model built by the frequency-latency scaling module and instructions from the governor that we describe next. The governor's instructions specify how much a given application's performance may slip without violating SLOs. The DVFS controller uses this to select an operating frequency for the application such that performance degradation remains within an acceptable bound. This approach enables PowerWeave to decouple application-specific policy design in userspace from the power management control plane within the interposition layer.

#### F. PowerWeave Governor

Inference workloads can be highly unpredictable, with request rates, execution phases, and latency sensitivities that fluctuate over time and vary widely across tenants. Thus, it is necessary to adapt to these shifting conditions to sustain high SLO attainment. Because PowerWeave's DVFS controller operates above the device driver and has no direct visibility into application semantics, it relies on a global Governor to interpret runtime performance characteristics and coordinate DVFS decisions. In particular, PowerWeave's Governor sits next to the application layer, monitoring load and request latency. It also communicates the permissible performance degradation to the DVFS controller, enabling it to select the appropriate operating frequency.

Once in the operating phase, the governor follows a sequence of steps to request frequencies throughout each tenant's serving (Figure 6). First, it monitors the latencies of each application at peak frequency, establishing a reference baseline per domain. Using this baseline and the application-provided SLO, it calculates a performance slack, representing the amount of slowdown the workload can tolerate without risking an SLO violation. This slack is passed to the DVFS controller, which selects operating frequencies according to the application's slowdown-frequency function (Figure 6, Stage ⃝1 ).

However, as workloads evolve over time, the governor must continuously track per-application load and latency to adapt to changing conditions. To do so, it employs a monitoring window that detects short-term divergence. If spikes or dips appear in a tenant's request-arrival rate, the governor recomputes the slack that the tenant can safely sustain (Figure 6, Stage ⃝2 ). Suppose an application is executing at frequency f1, with a performance slack of s1%. Given a currently observed latency l1, the governor updates the requested slack to s2%, as follows:

$$s_2 = \frac{((1 - s_1) \times l_1)}{SLO},$$

Intuitively, the governor infers the theoretical latency at maximum frequency from the slip s<sup>1</sup> with (1 − s1) × l1, and divides this by the SLO target to obtain the revised allowable slowdown. The governor sends this new slack s<sup>2</sup> to the DVFS controller to request a frequency update. This process is repeated for every monitoring window.

The governor also provides fast corrective action: upon an SLO violation, it signals the DVFS controller to maximize affected-domain frequencies until latency returns to a safe margin. It then restarts the adaptation process, recomputing slack and requesting frequency updates as conditions continue to change (Figure 6, Stage ⃝3 ).

Through this continuous feedback loop, the governor ensures that PowerWeave remains robust to runtime variability, responsive to divergent tenant behaviors, and consistently able to maintain performance targets while minimizing energy.

The governor is also flexible enough to monitor multiple SLOs per partition. If different metrics require distinct scaling behaviors (e.g., TTFT vs. TPOT, or small vs. medium vs. large input prompts), the governor adopts the most conservative allowable slack to ensure that all constraints are satisfied. Overall, this approach enables a varying degree of policy design based on individual application requirements.

