# C. Off-chip Access Optimization

**Tile schedule.** Our architecture adopts a *GS-feature* cache, where each cache line is tagged using a 28-bit GS ID, and an additional 4 bits record the number of tiles intersected by each GS. The cache prioritizes replacing less important GSs. These 4 bits, together with the 28-bit ID, form a 32-bit aligned storage entry. This design effectively reduces off-chip accesses by leveraging the spatial locality of GSs. However, different tile scheduling trajectories influence cache hit rates. As shown in Fig. 11(a), the baseline implementation scans tiles in a row-by-row manner, exploiting horizontal locality but lacking vertical and hierarchical locality. For instance, it can reuse GSs intersecting horizontally aligned tiles (e.g.,  $1 \times 2$ ,  $1 \times 3$ ), but not vertically aligned ones (e.g.,  $2 \times 1$ ,  $3 \times 1$ ). A slight improvement is achieved using the S-trajectory, which reverses direction at the end of each row.

Inspired by Morton encoding [33], which interleaves the y- and x-axis bits (e.g., interleaving  $y_1y_0$  with  $x_1x_0$  to obtain  $y_1x_1y_0x_0$ ), we schedule tiles in increasing Morton code order, forming a Z-trajectory that better preserves 2D locality, as shown in Fig. 11(b). However, the diagonal segments of this trajectory may span large spatial distances, causing discontinuities. To mitigate this issue, and inspired by the continuity of Gray code [13], we modify the Z-trajectory into a " $\pi$ " trajectory with improved continuity, as shown in Fig. 11(c). Both Z- and  $\pi$ -trajectories exhibit hierarchical locality, as indicated by the blue arrow (trajectory of  $2 \times 2$  tiles), making them inherently scalable. This curve corresponds to the Hilbert curve [17], originally proposed for fractal geometry in 1891. We further generalize the design for different 3DGS image resolutions, as shown in Fig. 11(d). Specifically, the  $\pi$ -trajectory is applied only within each  $8 \times 8$  tile block, while block-level traversal follows the S-trajectory. For images where the tile count is not divisible by 8, the remaining tiles are scheduled using a row-wise S-trajectory.

#### VI. EVALUATION

#### <span id="page-7-0"></span>A. Experimental Methodology

Algorithm Setup. Datasets and baselines: Our implementation and GPU-based inference are built on Gsplat, a widely used and efficient 3DGS library [51]. Following [22], we evaluate on real-world scenes from the MipNeRF-360 dataset [1], including garden, bicycle, stump, bonsai, counter, kitchen, and room. We compare our order-independent transmittance method with the sort-free weight-sum 3DGS algorithm [18]. Evaluation metrics: Rendering quality is evaluated using three standard metrics: Peak Signal-to-Noise Ratio (PSNR, higher is better), Structural Similarity Index (SSIM, higher is better), and Learned Perceptual Image Patch Similarity (LPIPS, lower is better). Implementation details: Training is performed on an NVIDIA RTX 3090 GPU, with checkpoints obtained after 7000 epochs following [22]. The model is then trained for an additional 10000 epochs, initialized from these checkpoints. The MLP learning rate is set to 0.005, while Gaussian learning rates are scaled by 0.01. Thanks to its lightweight design and the convenient training framework, training is highly efficient, requiring around 30 minutes per scene on an RTX 3090 GPU. Moreover, the model is trained offline only once and can be reused across various edge deployments.

Hardware Setup. Our architecture is implemented in SystemVerilog and synthesized with Design Compiler using the TSMC 28nm CMOS library. Rendering is performed with FP16 arithmetic based on DesignWare IP [43]. The design is fully pipelined and operates at 1 GHz. A DDR5-4800 DRAM with 38.4 GB/s bandwidth is modeled using Ramulator [24], while on-chip SRAM energy and area are estimated with CACTI [35]. Total energy consumption, including both on-chip and off-chip memory accesses, is obtained using DRAMPower [3]. Latency and memory traffic are evaluated with a cycle-accurate simulator, cross-validated against RTL simulation results. Table II summarizes the design metrics, and our design occupies 3.85 mm<sup>2</sup> and consumes 1.64 W.

![](_page_8_Figure_0.jpeg)

<span id="page-8-1"></span>Fig. 11. Comparison of different tile schedule trajectories with  $4 \times 4$  size as the example.

**Baselines.** We benchmark against the NVIDIA Jetson Orin Nano edge GPU [38] and the desktop-class NVIDIA RTX 3090, demonstrating substantially lower area and power while achieving higher performance, as detailed in Sec. VI-D. We further compare against the state-of-the-art 3DGS accelerators GSCore, MetaSapiens, and GBU [25], [29], [52].

TABLE II
AREA AND POWER OF OUR DESIGN.

<span id="page-8-2"></span>

| Component               | Configuration                                                | Area [mm <sup>2</sup> ] | Power [W] |
|-------------------------|--------------------------------------------------------------|-------------------------|-----------|
| Reconfigurable PE Array | 16×16 Reconfigurable PE                                      | 2.958                   | 1.48      |
| Support Modules         | X-PE Line + Y-PE line +<br>Coord. Gen. + Div. Array (4 Div.) | 0.064                   | 0.02      |
| On-chip Buffer          | GS Feature (88KB) +<br>Output (4KB) + Depth (4KB)            | 0.826                   | 0.14      |
| Total                   |                                                              | 3.85                    | 1.64      |

