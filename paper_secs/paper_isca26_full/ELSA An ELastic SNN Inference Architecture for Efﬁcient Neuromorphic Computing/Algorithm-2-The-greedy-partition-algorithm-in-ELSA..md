# Algorithm 2: The greedy partition algorithm in ELSA.

- <span id="page-6-1"></span>Input: The required memory a and # of neuron circuits of SNN layers d, neuron core memory A, # of neuron circuits in a neural core D, # of SNN layers L, # of neural cores N and communication traffic c<sub>ij</sub> from i-th to j-th layer.
   Output: A set list of postitions for SNN layers a
- 2 Output: A set list of partitions for SNN layers s.

9 end

- 3 sort(c, reverse=True) // sort the communication traffic
- 4 foreach  $c_{ij}$  do
  5 | if  $d_i + d_j < D$  and  $a_i + a_j < A$  then
  6 |  $d_i = d_j = d_i + d_j$ ;  $a_i = a_j = a_i + a_j$ ;
  7 |  $s_i = s_i + \{i, j\}$  // add layer i, j to the partition.
  8 | end

<span id="page-6-3"></span>TABLE II: Specifications of Evaluation Benchmarks.

| Work | Topology  | Dataset             | T.S. <sup>†</sup> | #Ops   | #Sops‡ | Param. |
|------|-----------|---------------------|-------------------|--------|--------|--------|
| W1   | VGG16     | CIFAR10             | 32                | 0.66G  | 0.62G  | 32.1M  |
| W2   | VGG16     | CIFAR100            | 32                | 0.66G  | 0.62G  | 32.4M  |
| W3   | VGG16     | CIFAR10-DVS         | 32                | 1.55G  | 2.55G  | 32.1M  |
| W4   | ResNet18  | ImageNet            | 32                | 3.63G  | 3.22G  | 11.7M  |
| W5   | ResNet34  | ImageNet            | 32                | 7.36G  | 9.43G  | 21.8M  |
| W6   | ResNet50  | ImageNet            | 32                | 8.18G  | 10.04G | 25.6M  |
| W7   | ViT Small | ImageNet            | 32                | 8.50G  | 90.74G | 22.1M  |
| W8   | YOLOv2    | COCO2017<br>VOC2017 | 32                | 18.44G | 37.63G | 52.8M  |
| W9   | ResNet101 | ImageNet            | 32                | 15.60G | 19.61G | 44.5M  |

†T.S. denotes allowed maximum time-steps. ‡Sop denotes synaptic operation.

functionality and annotate the toggle rate of the gate-level netlists, where the annotated switch activities are used to estimate the energy consumption with Synopsys PrimeTime PX. For memory, the area and energy of SRAM are generated via a commercial memory compiler. The off-chip access cost is evaluated using DRAMSim3 [27] with HBM3.0 [28]. For network-on-chip, we use DSENT [29] to simulate its energy and latency. Lastly, we build a cycle-level simulator with each component parameterized from the synthesized results (*i.e.*, area, power, and latency) of ASIC designs.

- 2) Benchmarks: The evaluation benchmarks are listed in Tab. II. For the classification task, benchmarks are SNNs converted from CNN (i.e., VGG16 [42] and ResNet18/34/50/101 [43]) and Transformer (i.e., ViT Small [44]) using datasets of CIFAR-10/100 [45], CIFAR10-DVS [46], and ImageNet [47]. For the detection task, benchmarks are SNNs converted from YOLOv2 with ResNet34 as backbone on COCO2017 [48] and VOC2007 [49] datasets. Note that all SNNs in Tab. II use 4-bit quantized weights, and all the benchmark latencies are obtained after the accuracy converges in elastic inference. Evaluated SNNs are generated following SpikeZIP-TF [4].
- 3) Baselines: We compare ELSA with three categories of baselines to comprehensively demonstrate improvements:
- Elastic SNN accelerators. We predominantly compare ELSA with SNN accelerators that support elastic inference, including TrueNorth [11], Darwin [14], MorphIC [17], and PAICORE [13], to highlight the benefits of architectural innovations of ELSA. These accelerators support TBT execution and elastic inference as discussed in Sec. I, and exploit common optimizations such as near-SRAM execution, addition-only computation, and event-driven sparsity. ELSA shares the same foundations, enabling fair comparisons.

<span id="page-6-2"></span><sup>&</sup>lt;sup>3</sup>Hilbert curve [25]: a continuous, fractal space-filling curve that recursively maps a one-dimensional interval onto two-dimensional space.

TABLE III: Hardware Specifications of ELSA

<span id="page-7-0"></span>

| Component<br>Name                                                                                            | Metric                                   | Spec.                                                    | Power (µW)/<br>Percentage                                                              | Area (mm²)/<br>Percentage                                                                                      |
|--------------------------------------------------------------------------------------------------------------|------------------------------------------|----------------------------------------------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| {Proce                                                                                                       | ess Elemen                               | ts $\times$ 4 $\}$ in Singl                              | e Neural Core                                                                          |                                                                                                                |
| Weight Memory<br>Membrane Memory<br>Spike Tracer Memory<br>FireComponent<br>16-input Adder Tree<br>Sub-total | size<br>size<br>size<br>count<br>count   | 4×102.4 KB<br>4×307.2 KB<br>4×102.4 KB<br>4×128<br>4×128 | 715.0/31.2%<br>96.1/4.2%<br>13.6/0.6%<br>84.7/3.7%<br>1191.4/52.0%<br>2103.3/91.8%     | 0.487/17.49%<br>1.460/52.44%<br>0.487/17.49%<br>0.0189/0.7%<br>0.140/5.02%<br>2.59/93.03%                      |
|                                                                                                              | {Router}                                 | in Single Neura                                          | l Core                                                                                 |                                                                                                                |
| SLayerNorm Unit SSoftmax Unit FIFO Queue Flit Generator Crossbar Switch Others Sub-total                     | count<br>count<br>size<br>count<br>count | 1<br>1<br>4×512 B<br>1<br>1<br>-                         | 33.7/1.5%<br>43.1/1.9%<br>91.6/4.0%<br>2.9/0.1%<br>16.4/0.7%<br>0.2/0.0%<br>187.9/8.2% | 0.091/3.27%<br>0.096/3.45%<br>0.0013/0.047%<br>0.0011/0.040%<br>0.0017/0.061%<br>0.00015/0.0054%<br>0.19/6.97% |
| ELSA Chip                                                                                                    | #Tiles                                   | $6 \times 6$                                             | <b>82490.0</b> /100%                                                                   | 100.23/100%                                                                                                    |

- Non-elastic SNN accelerators. We further compare ELSA with SNN accelerators based on LBL execution without elastic inference capability, including Phi [33], SpinalFlow [30], SASAP [32], Prosperity [31], and C-DNN [7], to provide a more comprehensive scope of comparison.
- QANN accelerators. To highlight the benefits of SNN features (*i.e.*, event-driven sparsity and addition-only computation), we compare ELSA with multiple state-of-theart QANN accelerators spanning digital designs (Eyeriss [21], Eyeriss v2 [34], ANT [18], S-CONV [35], AIOQAB [36], Sanger [37], and ViTALiTy [38]), digital in-memory design (AEC-CIM [39]), and analog in-memory design (LLH-CIM [40]). We also compare ELSA with commercial accelerators, including Jetson AGX Orin 64GB [50], Nvidia A100 GPU [51], TPU v4 [52], and Groq [53], which have comparable chip area to ELSA.
- 4) Metrics Modeling: We model the metrics of competing designs through two steps. 1) For latency and energy values reported in the original papers, we directly use those values. 2) For cases where such metrics are not provided, we estimate them using the reported peak throughput and peak energy efficiency, to enable a relatively fair comparison.
- 5) Early Termination: ELSA conducts the early termination by a confidence-based method for classification and detection [54]–[56] to reduce latency while maintaining task accuracy. On the classification task, we use the maximum class probability as the confidence score and terminate inference at intermediate time-steps once the confidence exceeds a predefined threshold. For detection tasks, we use the objectness score produced by the detector (*e.g.*, YOLO [57]) as the confidence for early termination.

#### B. Breakdown of Components in ELSA

**Power and Area Breakdown.** Tab. III lists the power and area cost of the components used to build ELSA. ELSA organizes  $6\times6$  neural cores in a 2D-mesh, and each neural core contains 4 PEs and 1 router. The PEs and routers consume 93.03% and 6.97% of the total area, respectively. The PE area is dominated by various memories, storing weight, membrane, and spike tracer of each neuron, which takes 93.97% of the PE area and 93.03% of the entire ELSA. The PE area could be further optimized by replacing massive SRAM with other on-chip embedded memories (*e.g.*, eDRAM [58], [59])

<span id="page-7-1"></span>![](_page_7_Figure_8.jpeg)

Fig. 15: **Energy breakdown of ELSA** on the benchmark W1-7 (Tab. II). Fire Comp. is short for fire component. The Pipeline Register Energy is consumed by FIFO Oueue.

via advanced integration technology. The router is mostly occupied by SSoftmax Unit and SLayerNorm Unit (i.e., 6.72% of ELSA). The reason is that SSoftmax Unit and SLayerNorm Unit contain ST-BIF neuron circuits and memories to store spike tracer and membrane. The power of ELSA is mainly consumed by adder tree (52%) and weight memory (31.2%) since SNN inference is dominated by spike-driven addition and weight access.

Energy Breakdown of ELSA is shown in Fig. 15, where adder tree consumes most of the energy ( $29\% \sim 39\%$ ) at most of benchmarks. The second-highest energy consumption comes from memory access, including FIFO Queue and Membrane, Weight, Spike-Tracer Buffer. Thanks to the dataflow and near-memory computing design in ELSA, the off-chip DRAM access is negligible, as only the inputs of SNN are loaded from DRAM.

#### C. Comparison with SNN Accelerators

Comprehensive comparisons with prior SNN accelerators are in Tab. IV. Further comparisons across six specific benchmarks are shown in Fig. 16, where ELSA prominently outperforms prior SNN accelerators on various tasks.

Compared to elastic SNN accelerators (TrueNorth [11], Darwin [14], MorphIC [17], and PAICORE [13]), ELSA achieves the highest throughput and energy efficiency, highlighting the effectiveness of proposed architectural innovations. Specifically, compared to TrueNorth [11], ELSA breaks computation dependencies via mini-batch spiking Gustavsonproduct dataflow and BAER, thereby eliminating global synchronization barriers (i.e., 1 ms global tick in TrueNorth [11]) and significantly improving throughput from 58.0 GOPS to 4135.4 GOPS. Compared to the SOTA accelerator PAICORE [13] in various benchmarks (Fig. 16), ELSA achieves 27.4× geomean energy-saving from the mini-batch spiking Gustavson-product dataflow, and  $1.65\times$  geomean speedup from the spine/token-level fine-grained pipeline, respectively. Compared to Darwin [14], ELSA consistently achieves better performance, but falls short in bit-width precision flexibility, as Darwin [14] supports 1/2/4/8/16-bit computation for broader programmability. Note that elastic SNN accelerators generally show lower area efficiency than non-elastic accelerators (e.g., Prosperity [31]), as all weights and membrane states are stored on-chip. Nevertheless, ELSA achieves the highest area efficiency among elastic SNN accelerators, thanks to the spine/token-level pipeline improving the throughput (Fig. 22).

Compared to non-elastic SNN accelerators (C-DNN [7], SpinalFlow [30], Prosperity [31], SASAP [32], and Phi [33])

TABLE IV: Comparison with SNN accelerators.

<span id="page-8-0"></span>

|                        | SpinalFlow[30] | Prosperity[31] | SASAP[32]   | Phi[33]   | C-DNN[7]                | MorphIC[17] | TrueNorth[11] | Darwin[14]                 | PAICORE[13]   |           | ELSA                   |         |
|------------------------|----------------|----------------|-------------|-----------|-------------------------|-------------|---------------|----------------------------|---------------|-----------|------------------------|---------|
| Technology             | 28 nm          | 28 nm          | 40nm        | 28nm      | 28nm                    | 65nm        | 65nm          | 22nm                       | 28nm          |           | 28nm                   |         |
| Voltage(V)             | 0.9            | n/a            | 0.56-1.1    | n/a       | 0.7-1.1                 | 0.8-1.2     | 0.7-1.04      | 0.8                        | 0.675         |           | 0.9                    |         |
| Freq.(MHz)             | 200            | 500            | 50-200      | 500       | 50-200                  | 55-210      | 0.001△        | 333                        | 168           |           | 200-500                |         |
| SRAM Only              | No             | No             | No          | No        | No                      | Yes         | Yes           | Yes                        | Yes           |           | Yes                    |         |
| Core Number            | 1              | 1              | 2           | 1         | 64                      | 4           | 4096          | 575                        | 1024          |           | 36                     |         |
| Area(mm <sup>2</sup> ) | 2.09           | 0.529          | 2.69        | 0.662     | 20.25                   | 2.86        | 430           | 358.53                     | 537.98        |           | 100.23                 |         |
| ALU per PE             | 8-b Add,Cmp    | 8-b Add        | 8-b Add,Cmp | 8-b Add   | 1-16 bit<br>MAC,Add,CMP | 1-b Add,Cmp | 8-b Add,Cmp   | 1/2/4/8/16-bit<br>Add, Cmp | 1/8-b Add,Cmp |           | 8-b Add,<br>Cmp, Shift |         |
| Memory                 | 585 KB         | 136 KB         | n/a         | 240 KB    | 552 KB                  | 288 KB      | 51 MB         | >23.44 MB                  | 121.87 MB     |           | 72 MB                  |         |
| Scheduling             | NoPipe         | NoPipe         | NoPipe      | NoPipe    | NoPipe                  | Layer       | Layer         | Layer                      | Layer         |           | Spine/Token            |         |
| GOPS                   | 684.5          | 390.1          | $72.5^{1}$  | 242.8     | 842.83                  | 0.42        | 58.0          | 66.8                       | $1421.6^2$    | 1982.9    | 4135.4                 | 2315.1  |
| TOPS/W                 | 4.22           | 0.299          | $42.8^{1}$  | 0.286     | $24.5^{3}$              | 0.29        | 0.400         | 0.18                       | $1.156^{2}$   | 20.89     | 25.55                  | 5.10    |
| pJ/Sops‡               | n/a            | n/a            | $0.078^{1}$ | n/a       | $1.1^{3}$               | 51          | n/a           | 5.47                       | $0.865^{2}$   | 0.051     | 0.032                  | 0.020   |
| GOPS/mm <sup>2</sup>   | 327.5          | 737.4          | 27.00       | 366.8     | 41.62                   | 0.147       | 0.134         | 0.186                      | 2.642         | 19.78     | 41.26                  | 23.10   |
| Network                | ResNet34       | VGG16          | Spikformer  | VGG16     | ResNet50                | n/a         | n/a           | VGG16                      | ResNet50      | VGG16     | ResNet50*              | VIT-S   |
| Accuracy (%)           | IN@65.5        | CF10@92.3      | İN@77.1     | CF10@91.1 | IN@75.2                 | MNIST@97.8  | n/a           | CF10@90.2                  | IN@77.1       | CF10@92.3 | IN@75.6                | IN@79.1 |
| Elastic Infer.         | X              | X              | X           | X         | Х                       | ✓           | ✓             | ✓                          | ✓             | ✓         | ✓                      | ✓       |

<sup>\*:</sup> The frequency of global tick in TrueNorth [11] is 1kHZ. IN denotes the ImageNet dataset and CF10 denotes the CIFAR10 dataset.

TABLE V: Comparison of ELSA w.r.t. QANN Accelerators. All designs are evaluated with the voltage of 0.9 V.

|                        | Eyeriss <sup>†</sup> [21] | Eyeriss v2 <sup>‡</sup> [34] | ANT[18]   | S-CONV[35] | AIOQAB[36] | Sanger[37] | ViTALiTy[38] | AEC-CIM[39] | LLH-CIM[40] |          | ELSA        |                   |
|------------------------|---------------------------|------------------------------|-----------|------------|------------|------------|--------------|-------------|-------------|----------|-------------|-------------------|
| Implementation         | Digital                   | Digital                      | Digital   | Digital    | Digital    | Digital    | Digital      | Digital CIM | Analog CIM  |          | Digital     |                   |
| Technology             | 28nm                      | 28nm*                        | 28nm      | 28nm       | 28nm       | 28nm       | 28nm         | 28nm        | 22nm        |          | 28nm        |                   |
| Frequency              | 200MHz                    | 200MHz                       | 200MHz    | 400MHz     | 500MHz     | 667MHz     | 500MHz       | n/a         | 244MHz      |          | 200-500M    | Hz                |
| Area(mm <sup>2</sup> ) | 2.969                     | 1.536*                       | 4.527     | 2.69       | 0.592      | 5.194      | 5.223        | 0.468       | 0.119       |          | 100.23      |                   |
| ALU per PE             | 8-b MAC                   | 8-b MAC                      | 4.8-b MAC | 8-b MAC    | 4-b MAC    | 16-b MAC   | 16-b MAC     | 8-b MAC     | 8-b MAC     | :        | 8-b Add,Cmp | ,Shift            |
| Network                | ResNet50                  | ResNet50                     | ResNet50  | ResNet34   | ViT-S      | ViT-S      | ViT-S        | n/a         | ResNet18    | ResNet18 | ResNet50    | ViT-S             |
| GOP/s**                | 40.26                     | 153.6                        | 1210.06   | 741.93     | 132.25     | 615.50     | 2057.61      | 213.4       | 62.4        | 1347.84  | 4135.42     | 2315.14           |
| TOPS/W**               | 0.766                     | 2.336*                       | 1.880     | 4.907      | 1.789      | 0.365      | 1.25         | 22.75       | 20.7        | 29.87    | 25.55       | 5.10              |
| GOPS/mm <sup>2</sup>   | 13.56                     | 100.01*                      | 264.78    | 275.81     | 223.39     | 118.50     | 393.95       | 456.00      | 524.37      | 13.45    | 41.26       | 23.10             |
| Accuracy(%)            | IN@75.97                  | IN@75.6                      | IN@75.08  | IN@71.8    | IN@78.5    | IN@79.2    | IN@79.5      | n/a         | IN@69.25    | IN@69.5  | IN@75.6     | IN@ <b>79.1</b> △ |
| Elastic Inference      | ×                         | X                            | ×         | X          | X          | X          | ×            | X           | X           | ✓        | /           | ✓                 |

<sup>†:</sup> performance of Eyeriss is reproduced from Accelergy [41]; \(^\Delta\): ViT-S on ELSA is trained by ourselves, while other accuracies are taken from original work. IN is short for ImageNet. ‡: data from Eyeriss V2 [34]; \*: the performance is scaled to 28nm; \*\*: 1 MAC=2 OP, #time-step Sop=2 OP, TOPS/W = #OP of network / Latency per frame, ELSA is measured at 200MHz.

<span id="page-8-1"></span>![](_page_8_Figure_7.jpeg)

Fig. 16: Energy and latency comparison of SNN accelerators. Statistics are normalized w.r.t. Eyeriss [21].

without elastic inference capability, ELSA achieves the highest throughput (4.9× higher than the SOTA accelerator C-DNN [7]), since ELSA has larger on-chip hardware resources and leverages spine/token-level pipeline to reduce end-toend latency (Fig. 5). The energy efficiency is also improved by mini-batch spiking Gustavson product that reduces the memory access (Fig. 23). However, the gain is marginal (24.5 TOPS/W in C-DNN vs. 25.6 TOPS/W in ELSA), as C-DNN (LBL-based) avoids SRAM storage for membrane states.

