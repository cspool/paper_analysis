# III. CASCADE: A MULTI-CHIPLET PIPELINED TFHE ACCELERATOR

We propose CASCADE, a multi-chiplet pipelined TFHE accelerator that exploits cross-HMUX pipeline parallelism. The key insight behind CASCADE is to • store all BSKs in distributed on-chip SRAMs and • execute these n HMUXs in a pipelined manner. CASCADE is provisioned with 126 MB of BSK SRAM, sufficient to accommodate encryption parameters up to the 128-bit security level. To reduce the high cost of a monolithic chip design, CASCADE introduces a distributed memory hierarchy that distributes SRAM across chiplets.

By keeping the BSKs resident in distributed on-chip SRAMs, which we call the **BSK-distributed strategy**, CAS-CADE not only eliminates most off-chip BSK transfers, but also confines intensive BSK accesses within each chiplet, preventing the BSK access conflicts and inflexibility caused by the centralized memory hierarchies of prior TFHE accelerators.

We first introduce the multi-chiplet pipeline in Sec. III. To address frequent intermediate-result transfers through D2D communication, CASCADE is co-designed with the Interleaved-Fusion mapping policy (Sec. IV) and offline scheduler (Sec. V), which are essential for hiding D2D latency.

#### A. Architecture Overview

The CASCADE architecture overview is shown in Figure 4. CASCADE consists of C HMUX Chiplets (HCs). 12 HCs are organized in a  $4\times3$  grid and interconnected in a ring topology via high-speed D2D (die-to-die) links (e.g., UCIe [25]). This ring topology allows the final chiplet  $(HC_{C-1})$  to pass data back to the first chiplet  $(HC_0)$ , enabling deep and flexible pipeline execution.

