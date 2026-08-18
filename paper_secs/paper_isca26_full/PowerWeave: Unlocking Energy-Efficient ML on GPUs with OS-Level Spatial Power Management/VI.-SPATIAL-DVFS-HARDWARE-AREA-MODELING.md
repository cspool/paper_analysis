# VI. SPATIAL DVFS HARDWARE AREA MODELING

To understand the practical feasibility of PowerWeave's fine-grained power mechanism, we develop a hardware area cost model that reflects the structural changes needed to realize per-domain DVFS in a technology-aware manner. This analysis complements our software design and provides grounded area estimates for our evaluation. PowerWeave targets a GPU architecture in which the SMs are divided into multiple independent voltage–frequency domains. Each domain is supplied by a local on-die regulator and may operate at its own frequency. We explicitly model the three dominant contributors shown in Figure 7: (i) on-die voltage regulators, (ii) voltage-domain boundary synchronization, and (iii) clock generation overheads. Area modeling is normalized to the target technology node using scaling factors derived from IRDS [20], [53]. We target a total silicon area of 1600 mm<sup>2</sup> , corresponding to approximately two reticle-limited dies of about 800 mm<sup>2</sup> each similar to B200 [54]. Our model can be further extended to additional dies.

DLDO Regulator Modeling. Each DVFS domain is supplied by a dedicated on-die digital low-dropout (DLDO) regulator. We model the area overhead of these regulators using a set of modeling constraints. First, we estimate the total peak current demand of the GPU as constant and independent of DVFS granularity. Next, we model the power portion of each DLDO (the PMOS pull-up array) based on an area directly proportional to the peak current it must supply. Finally, we estimate that the current is delivered by a discrete number of parallel PMOS devices, whose aggregate area therefore scales linearly with the total load current. Overall, these assumptions provide a conservative first-order analysis foundation.

For operating parameters, we set the input voltage to 1.15 V and an output range to 0.8-1.1 V, consistent with prior work [33]. We assume a maximum step size of 1% of Vout (approximately 11 mV [48]). The minimum resolution to achieve this is 128 levels; for a conservative (worst-case) area estimate, we model 256 voltage levels. After scaling to 5nm technology node, the per-domain control area becomes:

$$A_{\rm DLDO,ctrl}^{\rm 5nm} \approx \frac{A_{\rm DLDO,ctrl}^{\rm 7nm}}{S_{\rm 7nm \to 5nm}}$$

,

where S7nm→5nm is the digital area scaling factor.

Since the total power-device area remains constant, the incremental regulator overhead for N domains is therefore:

$$\Delta A_{\rm reg}(N) \approx A_{\rm DLDO,ctrl}^{\rm 5nm} \cdot (N-1)$$

This quantity represents the area overhead attributable solely to duplicated DLDO control logic.

Voltage-Domain Boundary Overhead (Level Shifters). Independent voltage islands require voltage level shifters (LSs), isolation, state retention, and synchronization FIFOs wherever signals cross DVFS boundaries. To establish an upper bound on the datapath width for crossings between the SMside L1 and the chip-level L2, we leverage characterization data of recent datacenter GPUs. Recent studies report L1 bandwidths ranging from 128 B/cycle (= 1024 bits) [32], up to 256 B/cycle (= 2048 bits) [23]. For a conservative bound (and to accommodate control/sideband signals), we set L1\_L2\_BITWIDTH = 2048 bits.

We instantiate a representative asynchronous FIFO (graycoded pointers) and augment it with voltage level shifters, isolation cells, and state retention flops, which are required for voltage-domain crossing. We model DVFS-domain links as AXI-like channels that *quiesce* traffic before a V /f change by de-asserting READY (VALID/READY two-way flow control). Under this policy, the crossing FIFO only needs to absorb a short pipeline/CDC latency and any burst tail. We therefore use a depth of 64 entries per data channel as a safe upper bound [4]. We synthesize and place-route this 2048-bit-wide FIFO macro in a 130nm process [13] to obtain a concrete area. This area is then scaled down to an equivalent area in a 5nm process node [17], [21], [22]. For a given DVFS partition (e.g., per-TPC), the total crossing overhead is:

$$A_{\text{cross,5nm}} = \frac{a_{\text{FIFO, 130nm}}}{S_{\text{dig, 130nm} \to 5nm}} + \frac{a_{\text{LS, 130nm}}}{S_{\text{ana, 130nm} \to 5nm}}$$

Where Sdig, 130nm→5nm and Sana, 130nm→5nm are the digital and analog area scaling factors. The final area overhead for N domains is then:

$$A_{\rm LS}(N) \approx A_{\rm cross,5nm} \cdot (N-1)$$

Clock-Domain Area Overhead. Increasing DVFS granularity also increases the number of independent clock domains, each serviced by a dedicated phase-locked loop (PLL) clock generator. To estimate the associated area overhead, we assume a worst-case scenario where every additional voltage domain introduces a new, fully independent clock domain with its own PLL instance. For a realistic upper bound, we adopt the area of a state-of-the-art digital PLL fabricated in a 5nm FinFET process. Specifically, a fully-synthesizable fractional-N injectionlocked PLL designed for manycore systems reports an area of 0.0036 mm<sup>2</sup> [28]. We will use this value directly as the area overhead for an additional PLL. The resulting area overhead for N DVFS domains can be expressed as:

$$A_{\rm PLL}(N) = A_{\rm PLL,unit} \cdot (N-1)$$

where APLL,unit = 0.0036 mm<sup>2</sup> is the per-domain PLL area. This term represents a conservative upper bound, as practical GPU implementations may employ a single PLL with multiple clock dividers for neighboring clock domains.

Combined Overhead Model. The total area overhead relative to a single DVFS domain is:

$$\Delta A_{\text{tot}}(N) = \Delta A_{\text{reg}}(N) + \Delta A_{\text{LS}}(N) + \Delta A_{\text{PLL}}(N).$$

PowerWeave uses this ∆Atot(N) directly in its analysis: in Section VIII we report the absolute area increase (in mm<sup>2</sup> ) and percentage of GPU die area for each DVFS granularity. Power Overhead. Beyond area, increasing the number of frequency domains introduces power overhead from duplicated control logic. We synthesize the DLDO controller at 7 nm with workload-driven activity annotation, obtaining 78 µW per regulator, approximately 11.5 mW in aggregate for 148 domains at max (one for each SM), negligible relative to the GPU's TDP (1000 W for a B200 GPU). The domain boundary crossing logic, synthesized at 130 nm, consumes 5.65 W for 148 domains as a conservative upper bound.

