# A. ANN Accelerators

**Digital Designs** [22], [63] also use Gustavson product to accelerate sparse-aware ANN by reducing memory access frequency, mitigating memory bottlenecks compared to inner/outer-product methods. Similarly, dataflow architectures such as Groq [53] and Cerebras [64] employ near-SRAM designs, storing weights on-chip to minimize memory overhead. Recent SRAM-based in-/near-memory design [65]–[67] further exploit on-chip SRAM arrays for neural-network computation. ELSA builds upon these techniques, incorporating SNN-specific optimizations, as discussed in Sec. III.

Analog In-memory Designs [40], [68] use memristive crossbar arrays to accelerate GEMM by computing directly within memory. However, the multi-row accumulation in Gustavson product may suffer from non-idealities like IR drop [69] and conductance discretization [70] in crossbar arrays, which impact accuracy. Thus, ELSA adopts a fully digital design instead of an analog in-memory implementation.

<span id="page-12-2"></span>TABLE X: Scaling study about energy breakdown.

| Components  | Detail            | ResNet18 | ResNet34 | ResNet50 | ResNet101 |
|-------------|-------------------|----------|----------|----------|-----------|
| Comput.     | Buffer            | 52.58%   | 52.59%   | 50.31%   | 48.55%    |
|             | Adder             | 34.68%   | 35.93%   | 35.70%   | 36.22%    |
|             | Neuron            | 6.25%    | 4.77%    | 4.33%    | 4.56%     |
|             | Total             | 93.50%   | 93.29%   | 90.34%   | 89.34%    |
| Communicat. | NoC Traffic       | 2.94%    | 4.09%    | 6.95%    | 8.02%     |
|             | Routing           | 0.05%    | 0.03%    | 0.03%    | 0.03%     |
|             | Total             | 2.99%    | 4.12%    | 6.98%    | 8.05%     |
| Scheduling  | Control+Scheduler | 1.21%    | 0.82%    | 1.12%    | 1.03%     |
|             | BAER              | 2.29%    | 1.77%    | 1.56%    | 1.58%     |
|             | Total             | 3.50%    | 2.59%    | 2.68%    | 2.61%     |
| pJ/SOP      | -                 | 0.038    | 0.030    | 0.032    | 0.032     |

<span id="page-12-3"></span>TABLE XI: SNN Executions Summary. "Time Advance" is the granularity at which components synchronously advance to the next time-step. "S./T.", "Calcu.", "Comm.", and "Gran." are spine/token, calculation, communication, and granularity.

| Methods        | Asynchronous Gran. Calcu. Comm. |       | Time Advance | Schedule Gran. |
|----------------|---------------------------------|-------|--------------|----------------|
| Loihi [12]     | Spike                           | Spike | Chip-level   | Network        |
| SpiNNaker [62] | Spike                           | Spike | Core-level   | Layer          |
| PAICORE [13]   | Spike                           | Spike | Core-level   | Layer          |
| ELSA (Ours)    | S./T.                           | S./T. | PE-level     | S./T.          |

#### B. SNN Accelerators

Elastic SNN Accelerators [11]–[14], [71], [72] follow TBT execution, featuring elastic inference with progressively emerged outputs. These accelerators generally adopt a multicore design [12] and exploit optimizations such as event-driven [14], addition-only [11], near-SRAM [72] and dataflow architecture [13]. However, their underlying coarse and layerwise pipeline [13] fundamentally limits the exploitation of the early-response by elastic inference, as illustrated in Fig. 5. Tab. XI summarizes the execution differences between ELSA and prior elastic accelerators. ELSA adopts spine/token-level granularity for both computation and communication, unlike the spike-level granularity used in Loihi [12], SpiNNaker [62], and PAICORE [13]. Moreover, ELSA advances the time-step at the PE-level, finer than other related works, allowing each neural core to manage the spines/tokens independently.

Non-elastic SNN Accelerators [7], [30]–[33], [73] follow LBL execution to exploit time-step parallelism and avoid costly membrane SRAM storage, thereby improving throughput and energy efficiency. Specifically, SASAP [32] scales up compute units to process spikes across all time-steps in parallel, while membrane states are immediately discarded after use [73]. However, such execution is inherently incompatible with early-response since outputs are synchronously generated before proceeding to the next layer.

#### IX. CONCLUSION

This paper presents ELSA, a near-SRAM dataflow architecture specifically designed to exploit elastic inference. By enabling early responses, ELSA is better suited for real-time applications such as autonomous driving compared to prior SNN accelerators. Experimentally, ELSA demonstrates that SNNs can outperform QANNs while maintaining comparable accuracy, reinforcing the potential of SNNs for future high-performance, low-power applications.

