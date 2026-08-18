# D. Comparison with QANN Accelerators

Since SNNs adopted in ELSA are converted from QANN models with the same accuracy, we compare ELSA with existing QANN accelerators to demonstrate that the intrinsic advantages of SNNs are effectively exploited under an equal-accuracy setting. Tab. V presents the throughput and energy efficiency of ELSA operating at 200MHz, targeting low-power scenarios. Compared to the SOTA ResNet50 accelerator

<span id="page-8-3"></span>![](_page_8_Figure_12.jpeg)

Fig. 17: Energy, latency, AEDP comparison with QANN accelerators. Statistics are normalized w.r.t. Eyeriss [21].

ANT [18], ELSA delivers  $3.4\times$  higher throughput and  $13.6\times$  better efficiency. Compared to the SOTA ViT-S accelerator ViTALiTY [38], ELSA achieves  $2.8\times$  throughput (*i.e.*, 5787 GOP/s) and  $4.1\times$  efficiency improvements at an aligned 500MHz frequency. Compared to the digital-CIM design (AEC-CIM [39]) and analog-CIM design (LLH-CIM [40]), ELSA achieves  $1.31\times$  and  $1.44\times$  better energy efficiency.

As shown in Fig. 17, we compare ELSA with prior QANN accelerators across multiple benchmarks in terms of latency, energy, and AEDP. For a fair system-level comparison, we include an additional 16 MB eDRAM [60] (4.76 mm<sup>2</sup>) for weight storage. Overall, ELSA achieves the best speedup and energy efficiency on most workloads. Note that, on ViT-S, ELSA yields a higher AEDP than ViTALiTy [38], as the 9.0× larger SOP count (Tab. II) reduces both GOPS and TOPS/W.

<span id="page-8-2"></span><sup>\*:</sup> after prune. ‡: (energy/Frame)/(# of pre-synaptic and post-synaptic Spikes). 1: evaluated with 0.56 V and 50 MHZ. 2: evaluated with 4-bit synaptic weight. 3: evaluated with 0.7 V and 50 MHZ.

<span id="page-9-0"></span>TABLE VI: Comparison of Large Chips running ResNet50.

|                                  | Jetson AGX Orin | A100           | TPU V4         | Groq           | ELSA                  |
|----------------------------------|-----------------|----------------|----------------|----------------|-----------------------|
| Implementation Dataflow Arch.?   | Digital<br>No   | Digital<br>No  | Digital<br>Yes | Digital<br>Yes | Digital<br>Yes        |
| Technology                       | 8nm             | 7nm            | 7nm            | 14nm           | 28nm                  |
| On-chip Memory<br>Frequency(MHz) | 4.25MB<br>1300  | 40MB<br>1095   | 170MB<br>1050  | 230MB<br>900.0 | 72MB<br>200.0         |
| Area(mm <sup>2</sup> )           | 200.0           | 826.0          | 700.0          | 720.0          | 100.2                 |
| TOPS<br>TOPS/W                   | 10.65<br>0.177  | 624.0<br>1.560 | 275.0<br>1.432 | 750.0<br>3.125 | 4.135<br><b>25.55</b> |
| GOPS/mm <sup>2</sup>             | 50.33           | 755.4          | 392.8          | 1041.7         | 41.26                 |

<span id="page-9-1"></span>TABLE VII: Accuracy of ANN, QANN, SNN, and SNN with early termination (E.T.) on ImageNet, and the latency reduction achieved by early termination (SNN+E.T.) relative to the SNN baseline on CNN and Transformer benchmarks.

| Method    |        |        | Accuracy |               | Latency     |
|-----------|--------|--------|----------|---------------|-------------|
|           | ANN    | QANN   | SNN      | SNN+E.T.      | Reduction   |
| ResNet18  | 69.61% | 67.85% | 67.85%   | 67.79%/64.38% | 22.6%/31.0% |
| ResNet34  | 74.52% | 71.54% | 71.54%   | 71.43%/68.59% | 26.1%/39.1% |
| ResNet50  | 78.17% | 75.60% | 75.60%   | 75.52%/71.14% | 16.6%/19.3% |
| ViT Small | 81.39% | 79.07% | 79.07%   | 78.98%/76.24% | 22.3%/33.1% |

#### E. Comparison with Large Chip

Tab. VI compares ELSA with QANN accelerators with large on-chip memory and die area. Thanks to the lossless conversion algorithm, the task accuracies of QANN in GPUs/Groq/TPU, and SNN in ELSA are the same (i.e. 75.6% accuracy on ImageNet-1K with ResNet50). Compared to the edge GPU Jetson AGX Orin, ELSA achieves 144.4× energy efficiency (TOPS/W) improvement and 49.9% chip area (mm<sup>2</sup>) reduction, showing competing performance in the edge application. Compared to high-performance GPU A100 and dataflow architectures (TPU V4 and Grog), ELSA has lower throughput (TOPS) since these accelerators have better chip technology (<14 nm), higher frequency (>900 MHz), and larger area (>700 mm<sup>2</sup>). Thanks to multi-level optimizations and the inherently low energy consumption of the SNN algorithm, ELSA achieves the highest energy efficiency (8.2 × improvement compared to Groq) among them.

