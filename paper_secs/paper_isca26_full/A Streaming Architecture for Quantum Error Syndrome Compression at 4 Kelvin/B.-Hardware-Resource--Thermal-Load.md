# *B. Hardware Resource & Thermal Load*

Timing and area data for SFQ cells and superconducting passive transmission lines (PTLs) are taken from recent literature [4], [50], [57], [73], [75]. Hardware and thermal results in this section are reported for surface code distance d = 21. Functional correctness is validated through gate-level simulations in PyLSE [10] with randomized inputs over thousands of measurement rounds, using the IcePack emulator—developed for syndrome compression evaluation (Section VI-A)—as the golden reference.

SFQ logic: We analyze the three key units, which together form an IcePack tile and can be modularly combined, to guide SFQ logic selection.

Preprocessing unit: The PPU consists of memory modules and synchronization cells (Figure 12). Filtering is handled by DRO and NDRO cells (Section V-B), eliminating the need for additional logic gates. Its size scales with the number of ancilla qubits per IcePack tile, storing one bit per ancilla in the measurement, syndrome, and prediction memories.

Processing unit: The PU operates in a streaming manner, as discussed in Section V-A. Greater pipeline depth and speed improve throughput, allowing larger block sizes and reducing the number of block units per PPU. We implement the PU with clocked RSFQ gates [39] as a 7-stage pipeline, with each stage handling a different position in the syndrome bitstream. The PU's size remains constant regardless of qubit count.

Encoding unit: The ENC (Section V-C) operates at sub-GHz speed, matching index transmission. Here, xSFQ—with its clock-free logic gates [69] (Section II-B)—offers advantages, reducing JJ count (a proxy for area) and power consumption without affecting throughput. As shown in Table III, this is the least resource-demanding component.

SFQ memory: IcePack's streaming architecture favors sequential access memories. Circular shift registers are a conservative choice; they are well-studied SFQ circuits that have demonstrated correct functionality at speeds exceeding our design assumptions (16 GHz [26] vs. 10 GHz). Delay-line memories (DLMs) offer similar functionality with energy and power advantages by replacing the synchronous components (e.g., DROs) of a shift register with passive transmission lines (PTLs). To prevent skew accumulation, the memory controller is synchronous and additional DRO cells segment the PTL, where necessary. Timing analysis at 10 GHz, assuming 20%

TABLE III JJ COUNT ESTIMATES ACROSS PHYSICAL ERROR RATES AT A MEDIAN TARGET SERIALIZATION LATENCY OF T = 500 ns.

| Error | Blocks   | Ancillas | JJs per Tile |         |       | Avg. JJs per |
|-------|----------|----------|--------------|---------|-------|--------------|
| Rate  | per Tile | per Tile | PU           | PPU     | ENC   | Ancilla      |
| 10−2  | 13       | 1,593    | 1,502        | 4,917   | 769   | 4.5          |
| 10−3  | 102      | 13,024   | 1,502        | 40,005  | 987   | 3.3          |
| 10−4  | 760      | 97,250   | 1,502        | 308,230 | 1,324 | 3.2          |

![](_page_11_Figure_0.jpeg)

Fig. 16. Panel (a): Photo of the prototyped 2mm Nb delay-line storage loop, fabricated in the MITLL SFQ5ee process—used in the PU and PPU for memory and row buffers (Figures 9 and 12). Panel (b): Analog voltage-sense output with amplitude proportional to the number of SFQ pulses stored in the delay line. Results show two SFQ pulses stored and circulating at 33 GHz.

timing variability for cells and 1% for the PTL [46], [49], shows that up to 41 bits can be stored per PTL before requiring DRO insertion. This results in a  $40 \times$  reduction in the number of JJs per bit compared to a shift-register-based design.

Prior work provides theoretical analysis of DLM storage density [75]. Figure 16 presents our experimental results from a Nb-based DLM, which we fabricated in the MITLL SFQ5ee node [67] and tested at speeds up to 33 GHz.

1) JJ count: Following standard convention, JJ count is used as a first-order proxy for the active SFQ circuitry area. Table III provides a breakdown by unit for a median target serialization latency of T = 500 ns. Figure 17 shows results across a range of latencies. Even the largest configuration remains well within the capabilities of current superconducting electronics fabrication. For reference, the MITLL SFQ5ee process—the same used for the experimental validation in Figure 16—supports integration of over one million JJs per cm<sup>2</sup> [50]. The JJ count per ancilla varies from 3.2 to 13.6, with an average of 4. For comparison, syndrome-parallel local decoders such as Clique [54] require at least 96 JJs per ancilla for core logic and clocking, corresponding to 4 XORs, 2 ANDs, and 2 NOTs. This is 7-30× higher than IcePack even before accounting for additional overheads such as storage, buffering, and serialization.

IcePack's JJ advantage stems from hardware design choices, in particular its streaming microarchitecture and the extensive use of PTLs, rather than algorithmic differences. Both Clique and IcePack perform pattern matching over the same small ancilla neighborhoods, though the two use the results in different ways. A streaming implementation similar to IcePack could in principle be applied to Clique's functionality.

2) PTL area: For PTL-based memories, the area required by the PTLs is also calculated. With a low JJ count per ancilla qubit (Table III) and JJs and PTLs placed on separate layers, PTLs dominate IcePack's layout area, making their footprint the effective area estimate. This area depends on controller

![](_page_11_Figure_7.jpeg)

Fig. 17. JJ count per ancilla versus serialization latency. IcePack averages 4 JJs per ancilla, rising to 14 for  $p=10^{-2}$  at 100 ns latency. For comparison, the lightweight Clique [54] requires 96 JJs per ancilla (without accounting for storage, buffering, and serialization overheads).

speed, line velocity factor, wiring layers, and line width and pitch. IcePack uses a conservative 10 GHz speed, while other parameters follow Nb and MoN stripline specifications in SFQ5ee [67] and SC2 [66], two processes currently offered by MITLL. Extrapolating from the original DLM study [75], IcePack requires a maximum of 3,000  $\mu \rm m^2$  per ancilla qubit using Nb striplines in SFQ5ee, or a more compact 187  $\mu \rm m^2$  using MoN striplines in SC2—supporting up to 500,000 qubits per cm². Further scaling can be achieved via higher speeds, advanced processes with high-kinetic-inductance layers [24], and bonding multiple SFQ chips [12].

3) Power consumption & thermal load: Each JJ consumes 0.2 aJ per switch, with biasing adding roughly 50% overhead [74]. In the worst case, when all JJs switch every cycle, IcePack consumes 10–42 nW per ancilla. By comparison, cables draw 1 mW per Gb/s plus 10.5 mW for peripherals [70], reaching 0.1 mW per ancilla. By reducing cable bandwidth requirements, IcePack lowers the total thermal load, after accounting for JJ power consumption. Figure 18 shows the tradeoff between upstream communication thermal load

![](_page_11_Figure_11.jpeg)

Fig. 18. Thermal load per ancilla versus serialization latency. IcePack achieves a Pareto improvement across all three error rates compared to a digital readout baseline, allowing it to occupy a unique region among other approaches, as illustrated in Figure 1.

(cables plus architecture) and serialization latency, comparing a digital readout baseline with IcePack across error rates. IcePack achieves Pareto improvement—our initial objective, as illustrated in Figure 1—reducing thermal load by 11× and latency by 10×, without any loss of accuracy. Thus, it both frees thermal budget for control, readout, and other electronics [5] and shortens the delay from qubit measurement to decoding and control, which can dominate overall runtime [40].

A direct thermal-load comparison with AFS's sparse representation is not possible at this stage because AFS does not provide a related hardware implementation, although it suggests a fully parallel realization. We compare against a (strictly cheaper) streaming SFQ baseline constructed by excising IcePack-specific components—the SCU, TCU, ENC, prediction memories, and associated routing—while preserving the core Block Units (without the prediction memories) and the Priority Selector (Figure 12) required for sparse encoding. For constant tile sizes, the sparse-representation baseline utilizes 37.5% (p = 10<sup>−</sup><sup>4</sup> ), 43.7% (p = 10<sup>−</sup><sup>3</sup> ), and 63.4% (p = 10<sup>−</sup><sup>2</sup> ) fewer JJs on average than IcePack across latencies. However, these hardware savings do not translate into thermal gains. At p = 10<sup>−</sup><sup>2</sup> and 10<sup>−</sup><sup>3</sup> , JJs contribute less than 2.5% to the per-ancilla thermal load, rendering JJ count reductions thermally negligible. Conversely, the sparse representation baseline supports 3.4–4.0× fewer ancillas per cable, which inflates the dominant cable cost by the same factor. At p = 10<sup>−</sup><sup>4</sup> and T = 1, 000 ns, where the architectural share of the thermal load reaches a maximum of 16%, IcePack's 2.8× compression advantage over sparse representation yields a 2.4× improvement in total thermal cost. In all other operating points, which have a lower relative architectural cost, the gap between IcePack and sparse encoding widens and more closely follows the data-volume reduction trends of Figure 8, confirming that the additional algorithmic complexity consistently returns dividends.

- *4) Processing latency:* To evaluate streaming processing's impact on serialization latency, we use Stim traces to model the syndrome index distribution and run 100,000-cycle simulations of queue occupancy across block sizes and error rates to determine 99th-percentile latency (Figure 13). For block sizes ≤ 128, IcePack's streaming processing adds at most 17 ns, under 4% of the 500 ns target serialization latency.
- *5) Decompression at 300 K:* Decompression happens on the fly as data arrive. For Rice-Golomb codes, a counter handles unary-to-binary conversion and an accumulator maps gap lengths to absolute indices. Spatial cluster expansion requires only an addition with small opcode-dependent constants, corresponding to the offsets shown in Figure 11. For temporal clustering, an ascending-order FIFO is used to store predictions, which are checked via arithmetic comparison. We synthesized the design using Synopsys DC with the Nangate 45 nm library [62]. Post-synthesis results show a decoding latency of 2.5 ns, minor compared to the 100-1,000 ns syndrome transmission latency.

# *B. Hardware Resource & Thermal Load*

Timing and area data for SFQ cells and superconducting passive transmission lines (PTLs) are taken from recent literature [4], [50], [57], [73], [75]. Hardware and thermal results in this section are reported for surface code distance d = 21. Functional correctness is validated through gate-level simulations in PyLSE [10] with randomized inputs over thousands of measurement rounds, using the IcePack emulator—developed for syndrome compression evaluation (Section VI-A)—as the golden reference.

SFQ logic: We analyze the three key units, which together form an IcePack tile and can be modularly combined, to guide SFQ logic selection.

Preprocessing unit: The PPU consists of memory modules and synchronization cells (Figure 12). Filtering is handled by DRO and NDRO cells (Section V-B), eliminating the need for additional logic gates. Its size scales with the number of ancilla qubits per IcePack tile, storing one bit per ancilla in the measurement, syndrome, and prediction memories.

Processing unit: The PU operates in a streaming manner, as discussed in Section V-A. Greater pipeline depth and speed improve throughput, allowing larger block sizes and reducing the number of block units per PPU. We implement the PU with clocked RSFQ gates [39] as a 7-stage pipeline, with each stage handling a different position in the syndrome bitstream. The PU's size remains constant regardless of qubit count.

Encoding unit: The ENC (Section V-C) operates at sub-GHz speed, matching index transmission. Here, xSFQ—with its clock-free logic gates [69] (Section II-B)—offers advantages, reducing JJ count (a proxy for area) and power consumption without affecting throughput. As shown in Table III, this is the least resource-demanding component.

SFQ memory: IcePack's streaming architecture favors sequential access memories. Circular shift registers are a conservative choice; they are well-studied SFQ circuits that have demonstrated correct functionality at speeds exceeding our design assumptions (16 GHz [26] vs. 10 GHz). Delay-line memories (DLMs) offer similar functionality with energy and power advantages by replacing the synchronous components (e.g., DROs) of a shift register with passive transmission lines (PTLs). To prevent skew accumulation, the memory controller is synchronous and additional DRO cells segment the PTL, where necessary. Timing analysis at 10 GHz, assuming 20%

TABLE III JJ COUNT ESTIMATES ACROSS PHYSICAL ERROR RATES AT A MEDIAN TARGET SERIALIZATION LATENCY OF T = 500 ns.

| Error | Blocks   | Ancillas | JJs per Tile |         |       | Avg. JJs per |
|-------|----------|----------|--------------|---------|-------|--------------|
| Rate  | per Tile | per Tile | PU           | PPU     | ENC   | Ancilla      |
| 10−2  | 13       | 1,593    | 1,502        | 4,917   | 769   | 4.5          |
| 10−3  | 102      | 13,024   | 1,502        | 40,005  | 987   | 3.3          |
| 10−4  | 760      | 97,250   | 1,502        | 308,230 | 1,324 | 3.2          |

![](_page_11_Figure_0.jpeg)

Fig. 16. Panel (a): Photo of the prototyped 2mm Nb delay-line storage loop, fabricated in the MITLL SFQ5ee process—used in the PU and PPU for memory and row buffers (Figures 9 and 12). Panel (b): Analog voltage-sense output with amplitude proportional to the number of SFQ pulses stored in the delay line. Results show two SFQ pulses stored and circulating at 33 GHz.

timing variability for cells and 1% for the PTL [46], [49], shows that up to 41 bits can be stored per PTL before requiring DRO insertion. This results in a  $40 \times$  reduction in the number of JJs per bit compared to a shift-register-based design.

Prior work provides theoretical analysis of DLM storage density [75]. Figure 16 presents our experimental results from a Nb-based DLM, which we fabricated in the MITLL SFQ5ee node [67] and tested at speeds up to 33 GHz.

1) JJ count: Following standard convention, JJ count is used as a first-order proxy for the active SFQ circuitry area. Table III provides a breakdown by unit for a median target serialization latency of T = 500 ns. Figure 17 shows results across a range of latencies. Even the largest configuration remains well within the capabilities of current superconducting electronics fabrication. For reference, the MITLL SFQ5ee process—the same used for the experimental validation in Figure 16—supports integration of over one million JJs per cm<sup>2</sup> [50]. The JJ count per ancilla varies from 3.2 to 13.6, with an average of 4. For comparison, syndrome-parallel local decoders such as Clique [54] require at least 96 JJs per ancilla for core logic and clocking, corresponding to 4 XORs, 2 ANDs, and 2 NOTs. This is 7-30× higher than IcePack even before accounting for additional overheads such as storage, buffering, and serialization.

IcePack's JJ advantage stems from hardware design choices, in particular its streaming microarchitecture and the extensive use of PTLs, rather than algorithmic differences. Both Clique and IcePack perform pattern matching over the same small ancilla neighborhoods, though the two use the results in different ways. A streaming implementation similar to IcePack could in principle be applied to Clique's functionality.

2) PTL area: For PTL-based memories, the area required by the PTLs is also calculated. With a low JJ count per ancilla qubit (Table III) and JJs and PTLs placed on separate layers, PTLs dominate IcePack's layout area, making their footprint the effective area estimate. This area depends on controller

![](_page_11_Figure_7.jpeg)

Fig. 17. JJ count per ancilla versus serialization latency. IcePack averages 4 JJs per ancilla, rising to 14 for  $p=10^{-2}$  at 100 ns latency. For comparison, the lightweight Clique [54] requires 96 JJs per ancilla (without accounting for storage, buffering, and serialization overheads).

speed, line velocity factor, wiring layers, and line width and pitch. IcePack uses a conservative 10 GHz speed, while other parameters follow Nb and MoN stripline specifications in SFQ5ee [67] and SC2 [66], two processes currently offered by MITLL. Extrapolating from the original DLM study [75], IcePack requires a maximum of 3,000  $\mu \rm m^2$  per ancilla qubit using Nb striplines in SFQ5ee, or a more compact 187  $\mu \rm m^2$  using MoN striplines in SC2—supporting up to 500,000 qubits per cm². Further scaling can be achieved via higher speeds, advanced processes with high-kinetic-inductance layers [24], and bonding multiple SFQ chips [12].

3) Power consumption & thermal load: Each JJ consumes 0.2 aJ per switch, with biasing adding roughly 50% overhead [74]. In the worst case, when all JJs switch every cycle, IcePack consumes 10–42 nW per ancilla. By comparison, cables draw 1 mW per Gb/s plus 10.5 mW for peripherals [70], reaching 0.1 mW per ancilla. By reducing cable bandwidth requirements, IcePack lowers the total thermal load, after accounting for JJ power consumption. Figure 18 shows the tradeoff between upstream communication thermal load

![](_page_11_Figure_11.jpeg)

Fig. 18. Thermal load per ancilla versus serialization latency. IcePack achieves a Pareto improvement across all three error rates compared to a digital readout baseline, allowing it to occupy a unique region among other approaches, as illustrated in Figure 1.

(cables plus architecture) and serialization latency, comparing a digital readout baseline with IcePack across error rates. IcePack achieves Pareto improvement—our initial objective, as illustrated in Figure 1—reducing thermal load by 11× and latency by 10×, without any loss of accuracy. Thus, it both frees thermal budget for control, readout, and other electronics [5] and shortens the delay from qubit measurement to decoding and control, which can dominate overall runtime [40].

A direct thermal-load comparison with AFS's sparse representation is not possible at this stage because AFS does not provide a related hardware implementation, although it suggests a fully parallel realization. We compare against a (strictly cheaper) streaming SFQ baseline constructed by excising IcePack-specific components—the SCU, TCU, ENC, prediction memories, and associated routing—while preserving the core Block Units (without the prediction memories) and the Priority Selector (Figure 12) required for sparse encoding. For constant tile sizes, the sparse-representation baseline utilizes 37.5% (p = 10<sup>−</sup><sup>4</sup> ), 43.7% (p = 10<sup>−</sup><sup>3</sup> ), and 63.4% (p = 10<sup>−</sup><sup>2</sup> ) fewer JJs on average than IcePack across latencies. However, these hardware savings do not translate into thermal gains. At p = 10<sup>−</sup><sup>2</sup> and 10<sup>−</sup><sup>3</sup> , JJs contribute less than 2.5% to the per-ancilla thermal load, rendering JJ count reductions thermally negligible. Conversely, the sparse representation baseline supports 3.4–4.0× fewer ancillas per cable, which inflates the dominant cable cost by the same factor. At p = 10<sup>−</sup><sup>4</sup> and T = 1, 000 ns, where the architectural share of the thermal load reaches a maximum of 16%, IcePack's 2.8× compression advantage over sparse representation yields a 2.4× improvement in total thermal cost. In all other operating points, which have a lower relative architectural cost, the gap between IcePack and sparse encoding widens and more closely follows the data-volume reduction trends of Figure 8, confirming that the additional algorithmic complexity consistently returns dividends.

- *4) Processing latency:* To evaluate streaming processing's impact on serialization latency, we use Stim traces to model the syndrome index distribution and run 100,000-cycle simulations of queue occupancy across block sizes and error rates to determine 99th-percentile latency (Figure 13). For block sizes ≤ 128, IcePack's streaming processing adds at most 17 ns, under 4% of the 500 ns target serialization latency.
- *5) Decompression at 300 K:* Decompression happens on the fly as data arrive. For Rice-Golomb codes, a counter handles unary-to-binary conversion and an accumulator maps gap lengths to absolute indices. Spatial cluster expansion requires only an addition with small opcode-dependent constants, corresponding to the offsets shown in Figure 11. For temporal clustering, an ascending-order FIFO is used to store predictions, which are checked via arithmetic comparison. We synthesized the design using Synopsys DC with the Nangate 45 nm library [62]. Post-synthesis results show a decoding latency of 2.5 ns, minor compared to the 100-1,000 ns syndrome transmission latency.

