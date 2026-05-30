# IV. EVALUATION METHODOLOGY

## *A. Benchmarks*

We evaluate RELIEF against the four policies summarized in Section II using three vision and two RNN applications. The five applications, along with their input size, deadline, and laxity (when run alone), are listed in Table V. We assume the vision applications run at 60 frames per second (FPS) and thus use a deadline of 16.6 ms. Deadline for RNN applications has been borrowed from previous work [59]. Input sizes mirror prior work as well [15], [59]. Richardson-Lucy deblur is an iterative algorithm where higher iterations lead to better picture quality. We use 5 iterations to have a representative input size balanced with simulation time. Along similar lines, we assume a sequence length of 8 for both LSTM and GRU.

TABLE V: Benchmarks

| (Symbol) Benchmark               | Input / hid<br>den layer size | Deadline | Laxity  |
|----------------------------------|-------------------------------|----------|---------|
| (C) Canny edge detection [10]    | 128 x 128                     | 16.6 ms  | 13.6 ms |
| (D) RL deblur [33], [45]         | 128 x128                      | 16.6 ms  | 0.2 ms  |
| (G) GRU [13]                     | 128                           | 7 ms     | 2.3 ms  |
| (H) Harris corner detection [24] | 128 x 128                     | 16.6 ms  | 14 ms   |
| (L) LSTM [26]                    | 128                           | 7 ms     | 3.6 ms  |

#### *B. Platform*

We use gem5-SALAM [47] for our evaluation, which provides a cycle-accurate model for accelerators described in high-level C. The simulator consumes the description of an accelerator in LLVM [1] intermediate representation (IR) and a configuration file and provides statistics like execution time and energy consumption. These accelerators are then mapped into the simulated platform's physical address space, enabling access via memory-mapped registers. The simulated configuration, listed in Table VI, models a typical mobile device [38]. We model the hardware manager using an ARM Cortex-A7 based microcontroller running bare-metal C code. Cortex-A7 has an area and power overhead of 0.45mm<sup>2</sup> and <100mW [5], which can be reduced further by stripping the vector unit. The simulated platform models end-to-end execution of applications, from inserting the tasks into ready queues till the completion of each requested application. This includes interrupt handling, scheduling, driver functionality, DMA transfers, and accelerator execution. In addition to the bus-based interconnect between the accelerators listed in Table VI, we evaluate RELIEF's performance with a crossbar switch in Section V-H. The two topologies represent two ends of the interconnect cost/performance spectrum.

Our evaluation uses seven image processing accelerators, one each for the kernels shown in Figure 1. Each accelerator was

TABLE VI: Simulation setup

| Hardware<br>manager | ARM Cortex-A7 based 1.6 GHz single-core in<br>order CPU<br>32 KB 2-way L1-I; 32 KB 4-way L1-D; 64 B<br>cache line size |
|---------------------|------------------------------------------------------------------------------------------------------------------------|
| Main memory         | LPDDR5-6400; 1 16-bit channel; 1 rank; BG<br>mode; tCK = 1.25ns; burst length = 32<br>Peak bandwidth = 12.8 GB/s       |
| Interconnect        | Full-duplex bus; width = 16 B<br>Peak bandwidth = 14.9 GB/s                                                            |

designed in isolation by determining the *energy*×*delay*<sup>2</sup> *(ED*2*)* product for the execution of a single task on the accelerator, while varying the configuration in terms of the number of functional units and memory ports. The configuration with the minimum *ED*<sup>2</sup> was chosen for the design, similar to previous work [47], [53]. In practice, we expect accelerators to work on the same input size to allow for easy chaining and sharing of data by commonly used applications. Our accelerators, clocked at 1 GHz, thus, have enough scratchpad memory to work on 128x128 inputs along with double buffered output to avoid blocking on consumer accelerator reads. The precise scratchpad memory sizes are listed in Table I. For accelerators with differing input sizes, the software runtime or the hardware manager can break down tasks into smaller chunks, similar to accelerator composition in GAM+ [15].

#### *C. System load*

Combinations of the applications in Table V are often seen in real-world scenarios, e.g., Canny+LSTM is used for lane detection in self driving cars [57]. Enumerating all combinations of these applications, thus, helps us cover all their existing and potential future use cases. We experiment with four levels of contention to see how each of the policies scale. *Low* contention is just a single application, *medium* contention is all combinations of size 2, while *high* contention is all combinations of size 3. Increasing contention represents reduced ability to meet deadlines, with combinations larger than 3 meeting very few deadlines and thus not evaluated. In each of these scenarios, each application is instantiated once and the simulation ends when the last application finishes execution. The fourth level of contention, called *continuous* contention, is a modification of *high* contention where each of the three applications are run in a continuous loop to ensure each application experiences contention throughout its execution. We limit the execution time of each simulation to 50ms and report results for finished tasks. Each application is represented with a symbol in the following figures, as listed in Table V.

