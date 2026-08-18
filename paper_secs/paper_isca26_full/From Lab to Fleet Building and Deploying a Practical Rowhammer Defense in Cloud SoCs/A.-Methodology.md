# *A. Methodology*

Experimental Setup: We run full-system simulations using a setup as shown in Figure [6.](#page-9-1) The framework consists of (1) an internal emulator based on QEMU [\[5\]](#page-12-2), (2) an internal cyclelevel SoC simulator based on Asim [\[15\]](#page-13-19), and (3) an internal timing-accurate memory-controller and Rowhammer simulator (RHSim).

We run workloads inside a customized QEMU-based emulator to generate instruction traces. These traces feed into our SoC simulator, which models micro-architectural details such as out-of-order execution, branch predictors, TLBs, and the cache hierarchy. This is Microsoft's in-house simulator to evaluate all SoC components including the memory controllers, producing a trace of memory requests arriving at each controller. Finally, RHSim uses this trace to simulate each memory controller's operations.

RHSim simulates operations such as request scheduling, background refresh (REF), refresh management (RFM), and directed refresh management (DRFM). Unlike previous DRAM and Rowhammer simulators [\[13\]](#page-13-45), [\[42\]](#page-13-46), [\[49\]](#page-13-47), [\[69\]](#page-14-34), [\[86\]](#page-14-35), it does not model read, write, or precharge DDR commands, but only row activations from the input trace, constrained solely by the row cycle time (tRC). Although RHSim does not explicitly model read and write DDR commands, it still accounts for their latencies. It uses timestamps derived from the full-system SoC simulator as the *earliest* time to activate each row. This design makes RHSim much faster than prior simulators and creates worst-case conditions for a Rowhammer defense by removing any factors that could slow the rate of row activations.

Capturing instruction and memory traces allows us to run experiments efficiently. Running the full pipeline, from the workload's disk image to RHSim, as one monolithic system is slow and inefficient; even a small change forces a complete rerun. End-to-end simulations take weeks and months. By breaking the process into stages and capturing instruction traces, we can re-run only the parts we need. For example, testing new mapping functions or making configuration changes to Rowhammer defenses requires re-running only RHSim, not the entire framework.

Using the above framework, we simulate a complete system: an SoC with a full core, mesh interconnect, and DRAM configuration. In all our simulations, we set all our configuration parameters to their *production values* with one exception: the Rowhammer threshold. Because Rowhammer thresholds have steadily declined over the past decade, recent academic research estimates values to drop into the low hundreds. In contrast, the DRAM industry does not expect thresholds to fall below a few thousand. As a compromise, we evaluate Sigries using a more conservative Rowhammer threshold than the one we use in production, where lower thresholds correspond to more conservative assumptions.

Prior defenses: In addition to Sigries, we implemented seven prior defenses in RHSim: BlockHammer [\[95\]](#page-14-26), DREAM-R [\[82\]](#page-14-32), Graphene [\[62\]](#page-14-4), Hydra [\[67\]](#page-14-29), PARA/PRA [\[35\]](#page-13-17), [\[41\]](#page-13-0), PRAC [\[31\]](#page-13-20), and RRS [\[71\]](#page-14-25). We configured each defense exactly as described in prior work. For defenses with opensource implementations in prior simulators, such as Ramulator 2.0 [\[13\]](#page-13-45), we also verified that our configuration settings match the corresponding Ramulator implementation. While these prior defenses do not meet our requirements and cannot be deployed, comparing them to Sigries offers insights into its overall performance.

To mitigate Rowhammer, these defenses rely on different mechanisms. Like Sigries, DREAM-R, Graphene, Hydra, and PARA/PRA issue DRFM commands to hot rows. BlockHammer stalls "loads" and "stores" that activate a row too often. RRS swaps hot rows with randomly chosen cold rows. PRAC runs inside DRAM and raises ALERT for hot rows. In the end, all Rowhammer defenses introduce performance overhead in two ways: consuming DRAM bandwidth overhead due to DRFMs and "stalling" program execution. We use these two metrics in our evaluation.

Workloads: We use a mix of cloud-representative applications such as key-value stores (Redis, Memcached), databases (MySQL, Cassandra) and micro-benchmarks from SPEC2017 [\[80\]](#page-14-36) and PARSEC [\[7\]](#page-12-7) suites. We also use a synthetic application, *memstress*, which is a memory-intensive loop repeatedly accessing a large memory region. To maximize system utilization, we run enough instances of each workload

<span id="page-10-0"></span>

| Workload      | IPC  | LLC MPKI | RSS (GB) | Avg. # rows<br>per ref. window | R:W<br>ratio |
|---------------|------|----------|----------|--------------------------------|--------------|
| cassandra     | 2    | 20.78    | 480      | 15.63M                         | 0.7:0.3      |
| memcached     | 1.2  | 40.64    | 384      | 11.82M                         | 0.69:0.31    |
| mysql         | 1.81 | 26.93    | 102      | 6.31M                          | 0.73:0.27    |
| redis         | 2.27 | 16.78    | 526      | 11.84M                         | 0.77:0.23    |
| mcf           | 1.34 | 24.02    | 320      | 601.9K                         | 1:0          |
| XZ            | 1.68 | 10.1     | 260      | 1.67M                          | 0.65:0.35    |
| blackscholes  | 3.17 | 3.36     | 65       | 1.04M                          | 0.88:0.12    |
| bodytrack     | 4.62 | 0.78     | 22       | 527.2K                         | 0.66:0.34    |
| canneal       | 0.21 | 163.93   | 179      | 16.7M                          | 0.99:0.01    |
| dedup         | 0.77 | 66.55    | 267      | 2.53M                          | 0.67:0.33    |
| facesim       | 2.55 | 7.38     | 59       | 2.84M                          | 0.59:0.41    |
| ferret        | 3.44 | 0.88     | 25       | 1.31M                          | 0.79:0.21    |
| fluidanimate  | 4.26 | 9.16     | 80       | 5.79M                          | 0.79:0.21    |
| freqmine      | 2.13 | 2.03     | 39       | 715.9K                         | 0.68:0.32    |
| streamcluster | 1.37 | 53.7     | 29       | 3.06M                          | 0.92:0.08    |
| swaptions     | 4.61 | 0.04     | 32       | 90K                            | 1:0          |
| x264          | 5.4  | 1.57     | 29       | 1.59M                          | 0.67:0.33    |
| memstress     | 0.38 | 119.24   | 251      | 18.8M                          | 0.5:0.5      |
| decahammer    | N/A  | N/A      | N/A      | 10                             | 1:0          |
| megahammer    | N/A  | N/A      | N/A      | k                              | 1:0          |
| omni-deca     | N/A  | N/A      | N/A      | $10 \times b$                  | 1:0          |
| omni-mega     | N/A  | N/A      | N/A      | $k \times b$                   | 1:0          |

TABLE V: Workload characteristics. 2nd column (IPC) is the number of instructions retired (executed) per cycle. 3rd column (LLC MPKI) is the LLC Misses per Kilo instructions. 4th column (RRS) is the Resident Set Size. 5th column is the number of DRAM rows accessed in one tREFW (32 ms) across the entire SoC. *megahammer* uses an undisclosed, high value of k, b is the # of banks in our SoC.

to keep every core busy and generating memory traffic.

We also use four synthetic Rowhammer attacks. The first two, *decahammer* and *megahammer* target a single sub-bank; *decahammer* is a 10-sided attack (a common DDR4 bit-flip pattern) while *megahammer* is a *k*-sided attack with a high, undisclosed *k* designed to overwhelm Sigries's light mode and trigger its switch to heavy mode. We also use *omni-deca* and *omni-mega* which launch the same attacks across all banks in parallel to model massively parallel Rowhammer scenarios. Unlike all other workloads, these four attacks are generated in RHSim to create worst-case Rowhammer conditions—the highest row activation rates and no interfering traffic.

## B. High-Level Workload Characteristics

Table V provides detailed characteristics of the workloads used. Some of the workloads have a high degree of efficient instruction execution with an average IPC above 4, such as bodytrack, fluidanimate, swaptions, and x264. Others access many DRAM rows in each refresh window, and thus can stress Sigries, such as memstress, cassandra, and memcached.

**Sub-channel bandwidth utilization:** Figure 7 illustrates the average and peak sub-channel bandwidth utilization under this setup. Half the workloads (11 out of the 22) achieve peak utilization exceeding half the maximum sub-channel bandwidth. A few workloads, such as *dedup*, *facesim*, *fluidanimate*, and *memstress* are able to fully saturate the system's memory bandwidth. Both *omni-deca* and *omni-mega* continuously saturate memory bandwidth by mounting Rowhammer attacks on all sub-banks in the system.

"Hot" rows—rows with a large number of activations: Despite the high bandwidth utilization, no row is activated more than a few hundred times in a 32ms refresh window.

<span id="page-10-1"></span>![](_page_10_Figure_8.jpeg)

Fig. 7: Average and peak sub-channel bandwidth utilization.

Figure 8 shows the maximum, p90, and p50 activations per refresh window, counting only rows activated at least once. The two hottest rows appear in *mcf* (583 activations) and *xz* (326 activations). The rightmost bars show the average per-row activations across all workloads—90% of rows are activated fewer than 53 times.

The rarity of highly activated rows in our experiments may seem at odds with prior work showing that commodity benchmarks can trigger high activation rates on servers [53]. Those high rates were due to Intel's decision to store coherence directory bits in DRAM, a design choice our SoC does not share.

**Takeaway:** On our SoC, cloud workloads do not repeatedly activate the same row more than a few hundred times a refresh window (32ms).

#### C. Key Evaluation Ouestions

- Q1. Does Sigries ever transition from light to heavy mode when the system is not under attack? Since heavy mode adds significant DRAM bandwidth overhead, any transition into heavy mode would degrade system performance.
- **Q2.** Does Sigries transition from light to heavy mode under adversarial Rowhammer attacks, and if so, across how many sub-banks? Sigries's light mode handles some, but not all Rowhammer attacks. For heavy attacks, sub-banks not targeted remain in light mode, while targeted sub-banks transition into heavy mode.
- **Q3.** How does Sigries's performance compare to prior defenses? We use two metrics to compare performance across Rowhammer defenses: (1) DRAM bandwidth due to DRFM and (2) workloads' "stalling" execution time.
- **Q4.** How much area overhead does Sigries add to the hardware? A very cost-sensitive metric is the amount of area overhead a Rowhammer mitigation adds to an SoC. Even small increases can drive up silicon cost, power, and yield losses, making this a critical factor.

