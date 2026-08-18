# *A. Architecture Overview*

We propose DIAMoND, an ASIC design that efficiently accelerates the decoding throughput of MoE models for edge AI applications. As shown in Fig.5(c), DIAMoND takes after the standard SSD architecture. It contains a Near-DRAM computing module and several In-NAND computing modules, connected with individual SSD channels. These two kinds of modules are integrated through 2.5D package method to provide adequate channel bandwidth.

For the FFN layers in an MoE model, in-NAND computing manages dynamic selected experts to minimize frequent weight off-loading, while near-DRAM computing primarily handles fixed experts. For self-attention layers, in-NAND module performs linear projection; near-DRAM module is mainly employed for KV cache storage and attention computation, which leverages the high endurance and low program-erase latency characteristics of DRAM. Sec. IV-E details a specialized self-attention workflow between in-NAND and near-DRAM module.

## *B. In-NAND computing module*

As Fig.5(b), the in-NAND equips an SSD with 16 dies distributed over 2 channels. Each channel hosts 8 dies as an SSD package. For each SSD package, 8 dies are stacked vertically, with each die providing 256 Gb of storage capacity under SLC storage pattern. Each SSD package includes a Control Die that functions as the I/O interface and involves adder trees to add the results from the 8 NAND dies. The Control Die and NAND Dies are connected through TSVs as HBF [10]. We use the ONFI 6.0 protocol as JESD230G [19], [37] for each SSD channel with 4.8 GB/s per channel.

Our design employs a NAND configuration as the stateof-the-art YMTC 232L QLC NAND Die [56]. To support in-NAND computing, the peripheral circuit of each die is augmented with ADC, Shifter and Adder, all stacked on top of the NAND die as Fig.5(a). A single ADC can be shared by multiple bitlines (BLs) through buffering and time-division multiplexing [55], [61], [65]. INT8 weights are stored in NAND cells under 2's complement format. Activations are processed bitwise by applying the voltage at the input port in 2's complement format. To enhance compute parallelism, each weight matrix is replicated across the 8 dies within a package, while each bit of an activation element is assigned to a particular die, which enables the completion of a VMM operation at 8-bit precision within a single read cycle.

# *A. Architecture Overview*

We propose DIAMoND, an ASIC design that efficiently accelerates the decoding throughput of MoE models for edge AI applications. As shown in Fig.5(c), DIAMoND takes after the standard SSD architecture. It contains a Near-DRAM computing module and several In-NAND computing modules, connected with individual SSD channels. These two kinds of modules are integrated through 2.5D package method to provide adequate channel bandwidth.

For the FFN layers in an MoE model, in-NAND computing manages dynamic selected experts to minimize frequent weight off-loading, while near-DRAM computing primarily handles fixed experts. For self-attention layers, in-NAND module performs linear projection; near-DRAM module is mainly employed for KV cache storage and attention computation, which leverages the high endurance and low program-erase latency characteristics of DRAM. Sec. IV-E details a specialized self-attention workflow between in-NAND and near-DRAM module.

## *B. In-NAND computing module*

As Fig.5(b), the in-NAND equips an SSD with 16 dies distributed over 2 channels. Each channel hosts 8 dies as an SSD package. For each SSD package, 8 dies are stacked vertically, with each die providing 256 Gb of storage capacity under SLC storage pattern. Each SSD package includes a Control Die that functions as the I/O interface and involves adder trees to add the results from the 8 NAND dies. The Control Die and NAND Dies are connected through TSVs as HBF [10]. We use the ONFI 6.0 protocol as JESD230G [19], [37] for each SSD channel with 4.8 GB/s per channel.

Our design employs a NAND configuration as the stateof-the-art YMTC 232L QLC NAND Die [56]. To support in-NAND computing, the peripheral circuit of each die is augmented with ADC, Shifter and Adder, all stacked on top of the NAND die as Fig.5(a). A single ADC can be shared by multiple bitlines (BLs) through buffering and time-division multiplexing [55], [61], [65]. INT8 weights are stored in NAND cells under 2's complement format. Activations are processed bitwise by applying the voltage at the input port in 2's complement format. To enhance compute parallelism, each weight matrix is replicated across the 8 dies within a package, while each bit of an activation element is assigned to a particular die, which enables the completion of a VMM operation at 8-bit precision within a single read cycle.

