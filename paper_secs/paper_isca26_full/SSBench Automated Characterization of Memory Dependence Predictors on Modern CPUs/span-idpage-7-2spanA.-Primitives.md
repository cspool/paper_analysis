# <span id="page-7-2"></span>*A. Primitives*

JIT and Core Binding. We use operating system interfaces to pin processes to a core and to enable real-time updates of the machine code. On Linux, we use taskset to bind the test process to a specific CPU. On macOS, we set process affinity using the API pthread\_set\_qos\_class\_self\_np. We use mmap to create the instruction page and alternate its writable and executable properties using mprotect.

Timing. We measure cycles using: rdtscp on Intel; AMD's rdpru [\[14\]](#page-13-34); PMCCNTR on Arm via kernel PMU module [\[5\]](#page-13-35); S3\_2\_c15\_c0\_0 on Apple via kernel modification [\[54\]](#page-14-13); and rdcycle on RISC-V.

Test Suite Implementation. We implement the test suite with Python. The clustering algorithm is implemented using scikit-learn [\[56\]](#page-14-29), while the equation system solver is implemented using pulp [\[50\]](#page-14-30). Linear hash equations are solved through matrix operations based on numpy. During the hash solving process, we use kernel interfaces to record the virtual and physical collision addresses. If the hash function for the virtual address cannot find a solution, we attempt to solve for the hash function of the physical address.

#### *B. Noise Reduction*

To mitigate timing noise and MDP prediction table interference caused by context switches and frequency fluctuations, all MDP parameters in this work are recovered using statistical analysis. Specifically, to mitigate measurement noise, we repeat each experiment a fixed number of times for each parameter and report the mode as the final result. To further improve evaluation efficiency, we introduce an early stopping mechanism that reduces the number of repetitions when the measurement variance falls below a predefined threshold.

#### *C. Out-of-scope MDP Designs*

If the MDP adopts a design different from the assumed architecture, SSBench reports a diagnostic message during analysis. For example, if a different indexing mechanism is used, the existence analysis shows that the load PC does not influence MDP predictions. If the state machine employs no less than two counters, the 1-counter state-machine analysis fails because the resulting nonlinear equation system has no solution. If a nonlinear hash function is used, the nullspace dimension of the differential matrix becomes 0. If the MDP adopts an unknown design, the organization parameters cannot be recovered. In such cases, SSBench terminates at the corresponding stage and hints which component of the MDP deviates from the expected design, enabling further manual analysis.

