# <span id="page-6-1"></span>*D. Phase Prediction Module*

To exploit the heterogeneity of PhaseWeave, the system uses lightweight hardware modules called *Phase Predictors*. These modules forecast the next execution phase of each thread at regular *epochs*. They continuously track hardware counters and system activity, and determine the upcoming phase at the end of each epoch. This means that a thread will run on a chosen chiplet for at least the epoch duration. This duration needs to balance between the fine-granularity of application phases and migration overheads. Based on our sensitivity analyses, we empirically set the epoch duration to 100µs.

The predictors are implemented in hardware to avoid frequently interrupting the CPU and maximize efficiency. They

<span id="page-6-0"></span>TABLE II: Comparison of phase prediction approaches.

| Approach              | Acc. | Storage | Compute (per epoch)      |  |  |  |  |  |
|-----------------------|------|---------|--------------------------|--|--|--|--|--|
| Threshold             |      |         |                          |  |  |  |  |  |
| Threshold-Based       | <75% | 32B     | Comparison chain         |  |  |  |  |  |
| Clustering            |      |         |                          |  |  |  |  |  |
| K-Means               | <75% | 100KB   | L2 distance + log        |  |  |  |  |  |
| K-Medoids             | <75% | 100KB   | L2 distance on centroids |  |  |  |  |  |
| HDBScan               | <75% | 180KB   | L2 distance on neighbors |  |  |  |  |  |
| HMM                   | ∼80% | 4KB     | 1000 MACs + log          |  |  |  |  |  |
| Machine Learning (ML) |      |         |                          |  |  |  |  |  |
| Multi-Armed Bandits   | <75% | 40B     | Max + simple arithmetic  |  |  |  |  |  |
| Contextual Bandits    | ∼80% | 10KB    | 500 MACs + sqrt          |  |  |  |  |  |
| Random Forest         | >90% | 8KB     | Comparisons + ptr chase  |  |  |  |  |  |

can be instantiated per core or shared across a small group of cores to reduce area and power overheads. In our implementation, we opt for per-core predictors.

Importantly, predictors are entirely *transparent* to the application, as they only use hardware and software applicationagnostic features. On hardware side, they use counters such as IPC, cache, TLB, and branch MPKI. On software side, they use frequency of system calls split into categories (*e.g.*, network- and memory-allocation system calls). Using these metrics, the predictor classifies the upcoming phase into four categories corresponding to each chiplet type: low-power, compute-, memory-, or network-dominated phases.

Phase Classification Algorithms. We explore several approaches for phase classification. [Table II](#page-6-0) summarizes these approaches in terms of accuracy, hardware storage cost, and per-epoch compute requirements.

*Threshold-based* methods apply a chain of simple rules on individual metrics. For example, high IPC with low network calls would indicate a compute phase. *Clustering techniques* such as K-Means, K-Medoids, HDBScan, and Hidden Markov Models (HMMs) group similar epochs in feature space, capturing temporal or density-based patterns. *Machine learning* approaches provide higher predictive power: Multi-Armed Bandits and Contextual Bandits formulate phase selection as a reinforcement learning problem; Random Forests combine multiple decision trees to produce a consensus prediction.

Phase classification requires finding correlations between phases and features, and tolerance against noise from a wide array of features. We observe that many of the approaches do not satisfy these requirements. Threshold-based methods are too simple and fail in defining nuanced threshold boundaries. Clustering approaches suffer from noise across features that are important in one cluster but not to others. Multi-Armed Bandits is distribution-driven and thus fails to capture relationships between phases and features. Contextual Bandits also falls short to noise across features.

Thus, in PhaseWeave, we select *Random Forests* for the Phase Predictor. This method has the best balance of classification accuracy and cost, and it has been shown that such an approach can efficiently be implemented in hardware [\[19\]](#page-13-24).

Offline Training. We train the Random Forest model *offline*, using labeled data collected from a diverse suite of microbenchmarks [\[57\]](#page-14-14). During profiling, we run each microbenchmark phase on every chiplet type to perform a sensitivity sweep and determine which hardware configu-

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Fig. 10: Microarchitecture of the Random Forest Phase Predictor in PhaseWeave.

ration delivers the best performance. We then label each training sample with its optimal chiplet. The model learns these mappings using only application-agnostic signals, *i.e.*, hardware performance counters and categorized system-call frequencies. Hence, the classifier captures general trends rather than application-specific patterns. Once trained, the model parameters are fixed and loaded into the hardware predictor at boot time, where the same model is used for all applications.

Although Random Forests are traditionally viewed as inflexible, often requiring retraining when the input distribution shifts, we find that this limitation does not apply in PhaseWeave. The phase categories used in PhaseWeave are relatively broad and stable, and the underlying hardware-level indicators that characterize them remain consistent across workloads. As a result, the trained predictor generalizes well without per-application tuning and the model accuracy over time remains high without frequent retraining or updates.

**Microarchitectural Design.** Figure 10 shows the microarchitecture of the Random Forest-based Phase Predictor integrated in PhaseWeave. The predictor is a lightweight hardware module that operates asynchronously to application cores.

The trained model is stored in a compact SRAM structure called the *Decision Storage*, which holds all internal nodes and leaves of the Random Forest. An entry encodes a decision rule, consisting of a 4-bit *feature identifier* that selects the relevant hardware or runtime metric (*e.g.*, IPC, cache MPKI, branch MPKI, TLB MPKI, system call frequency), a 16-bit *comparison threshold*, and 12-bit *indices* for its left and right child nodes. Leaf entries store a 2-bit *phase label* corresponding to one of the four execution classes in PhaseWeave (Low-Power, Memory-, Network-, or Compute-heavy). A single control bit distinguishes internal and leaf nodes.

Online Inference. At runtime, the predictor is triggered periodically at the end of each epoch  $(e.g., 100\mu s)$ . The *feature sampler* aggregates collected counters 1 and updates a local feature buffer 2. The *traversal engine* then evaluates all trees in the forest by walking their decision structures stored in the Decision Storage. The engine loads the feature values from the buffer (3a), and for each tree, it begins at the root node and iteratively compares the current feature value with the stored threshold (3b). The engine follows either the left or right index until reaching a leaf node. Each traversal requires

a fixed number of steps (*i.e.*, 5 in our implementation). The predictor traverses multiple trees *in parallel*, evaluating them using parallel comparator and fetch units.

After tree traversal, the *voting unit* aggregates the individual phase classifications 4. This unit stores counters that accumulate the number of votes for each phase class and selects the class with the highest vote as the final prediction 5. **Overhead.** Our predictor uses 15 trees that each perform 5 traversals. We measure that the predictor takes 0.02% of core area and inference only requires 75 comparisons per epoch, which takes less than 100 cycles. This cost is both negligible and not on the application's critical path.

**Need for Fine-grained Runtime Phase Detection.** We distinguish three sources of phase heterogeneity (Section II): (i) across services, (ii) across datacenter-tax operations between services, and (iii) within services. Coarse-grained mechanisms can partially address the first two. For example, services may be statically bound to a preferred chiplet type (*e.g.*, fix a mostly-network service *Nginx* on near-network chiplet). However, such approaches are insufficient for two reasons.

First, and most importantly, we observe pronounced intraservice phase heterogeneity within individual services. For instance, even when a service is globally compute-heavy, substantial sub-intervals may be memory- or network-bound. Coarse-grained placement therefore leaves significant performance and energy-efficiency gains unrealized.

Second, service behavior is input- and context-dependent (Figure 4). The same service can be compute, memory, or network-dominated based on request mix and runtime interference, making static service-to-chiplet bindings brittle.

Need for In-hardware Phase Prediction. Our characterization shows that execution phases in datacenter applications typically last only tens to hundreds of microseconds. At this timescale, even modest prediction overheads impact latency and negate the benefit of migration. Thus, phase identification must be highly accurate and have extremely low overhead.

Software-based predictors have two key limitations. First, to keep overhead low, they are limited to simple detection methods (e.g., clustering or thresholding [71], [85]). However, these methods are not accurate enough for reliable phase steering. More expressive ML models (e.g., Random Forest) take 100-1000  $\mu$ s to run in software. This cost is comparable to or longer than the phase itself, making them impractical.

Second, software-based prediction would sit on the critical path. At the end of each epoch, execution must pause while a CPU core runs the predictor and computes a decision, increasing request latency. An alternative would be to dedicate some cores for prediction. This approach avoids CPU pauses but reduces available compute, and thus lowers throughput. In contrast, PhaseWeave's Predictor module runs asynchronously and off the critical path, enabling sub-microsecond inference without interrupting the application.

#### <span id="page-7-1"></span>E. Thread Migration

When the Phase Predictor signals that a thread's upcoming phase is better matched to a different chiplet, the runtime must decide whether to relocate the thread or keep it on its current chiplet. Migrating blindly to the predicted optimal chiplet may be inefficient: a destination chiplet that is heavily loaded imposes queuing delay that outweighs architectural advantage. Thus, PhaseWeave uses a benefit-aware migration policy that explicitly weights predicted per-phase speedup against a real-time estimate of destination load.

Each chiplet exposes a software-writable *task-count* register that the OS updates whenever its local runqueue changes. Dedicated *Load State* modules read these per-chiplet counters and export them to the thread scheduler. The scheduler queries the Load State counters and combines them with offline phase-characterization data, *i.e.*, expected per-phase performance on each chiplet type. The scheduler uses these as inputs to its thread migration algorithm.

Algorithm 1 shows the procedure. Importantly, the process is not on the critical path of thread's execution and it does not directly affect workload's performance.

#### <span id="page-8-0"></span>**Algorithm 1** Benefit-aware thread migration algorithm.

```
Inputs: Chiplets C, thresh. \theta, T_{\min}, weight \lambda, switch cost C_{\text{switch}}
 1: for each thread t at epoch boundary do
          p \leftarrow \text{predicted phase for } t
          c \leftarrow current chiplet assignment of t
          for each candidate chiplet c' \in \mathcal{C} do
 4:
               S_{c'} \leftarrow \text{expected performance of phase } p \text{ on } c'
 5:
               S_c \leftarrow \text{expected performance of phase } p \text{ on } c
 6:
               Q_{c'} \leftarrow runqueue length reported by Load State of c'
 7:
               U(t,c') \leftarrow S_{c'} - S_c - \lambda \cdot Q_{c'} - C_{\text{switch}}
 8:
          end for
 9:
              \leftarrow \operatorname{arg\,max}_{c'} U(t,c')
10:
          if U(t, c^*) > \theta and residency_time(t) > T_{\min} then
11:
12:
               Migrate t from c to c^*
13:
14: end for
```

Formally, for a thread t with predicted phase p currently assigned to chiplet c, the scheduler computes the migration utility to chiplet c' as:

$$U(t,c') = S(p,c') - S(p,c) - \lambda \cdot Q(c') - C_{\text{switch}}$$
 (1)

S(p,c) denotes the expected steady-state performance (e.g., IPC) of phase p on chiplet c, Q(c') is the current runqueue length on c',  $\lambda$  converts queued threads into an expected latency penalty, and  $C_{\rm switch}$  represents the one-time context-switch cost of reassigning the thread. If c'=c, we set the utility to 0, as this case incurs no migration costs or performance changes. The scheduler selects the chiplet  $c^*=\arg\max_{c'}U(t,c')$  and performs the migration only if  $U(t,c^*)>\theta$ . To avoid oscillation due to transient prediction noise, the scheduler ensures that a given thread has a minimum residency time  $T_{\min}$  on its current chiplet. If no candidate yields positive utility, the thread remains on its current chiplet.

Upon deciding a migration, the scheduler reassigns the thread by removing it from its current executing core and enqueuing it onto the runqueue of the selected destination chiplet. The thread is reassigned to a core on the destination chiplet using a standard context switch, which can be accelerated with recently proposed hardware mechanisms [82], [83].

**OS** Integration. Chiplets in PhaseWeave are exposed to the OS as standard groups of CPU cores within a heterogeneous multicore system, similar to NUMA nodes. From the OS perspective, chiplets are simply disjoint CPU clusters (e.g., distinct CPU ID ranges), and the OS retains full control over thread scheduling. PhaseWeave does not modify OS scheduler data structures or bypass the scheduler. Instead, it provides placement recommendations through a hardware-visible interface (e.g., a dedicated MSR or memory-mapped control region). The OS periodically reads these recommendations and updates thread affinity using its native mechanisms. A migration corresponds to a conventional inter-core task migration handled entirely by the OS scheduler. All architectural state management, including page-table root updates and TLB consistency, relies on existing hardware and OS mechanisms, which is compatible with conventional OSes.

#### F. Future Extensions: Heterogeneous ISAs and Accelerators

PhaseWeave's phase detection and migration infrastructure provides a foundation for future extensions. Currently, all cores use an iso-ISA design, enabling transparent thread migration across heterogeneous chiplets with low overhead, but constraining the system to ISA-compatible cores. Future systems could extend PhaseWeave to integrate cores with *distinct ISAs or specialized accelerators*. Phase predictors could then identify phases suited to such hardware and provide guidance to a higher-level runtime or compiler for offload opportunities.

For example, a PhaseWeave server could integrate ARM cores for low-power phases or specialized accelerators [1], [27], [81]. During execution, the predictor would detect when a thread enters a phase best suited for these units. When ISA or accelerator compatibility differs, instead of migrating the thread, PhaseWeave would emit structured feedback to the runtime or compiler. Then, the software could use this information to recompile or optimize future runs, statically or dynamically offload phases to the most appropriate hardware.

