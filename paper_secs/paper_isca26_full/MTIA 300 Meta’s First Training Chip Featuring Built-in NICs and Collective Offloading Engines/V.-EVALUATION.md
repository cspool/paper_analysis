# V. EVALUATION

Proper evaluation of MTIA 300 requires situating it in the fast-evolving AI accelerator landscape. Development of MTIA 300 started several years ago to compete with H100 and H200 GPUs. Its successor, MTIA 400 [34], was designed

![](_page_8_Figure_0.jpeg)

Fig. 11: Execution of an HCCL collective operation across host and device components.

to rival GB300 GPUs, while MTIA 450 and 500 [34] target industry-leading GenAI inference performance against future GPUs. In this context, Table III compares MTIA 300, H100, and H200. We evaluate MTIA 300 using both microbenchmarks and production-scale DLRM training workloads. Additionally, although MTIA 300 was optimized for DLRMs, its superior HBM capacity and bandwidth make it effective for GenAI inference, which we also evaluate in this section.

TABLE III: Evaluation testbeds.

|                            |             | MTIA 300        | H100(∗)        | H200           |
|----------------------------|-------------|-----------------|----------------|----------------|
| Peak FLOPS/s (BF16)        |             | 560 TF/s        | 780 TF/s(∗)    | 1000 TF/s      |
| HBM                        | Capacity    | 216 GB          | 96 GB          | 141 GB         |
|                            | Bandwidth   | 6.1 TB/s        | 2.4 TB/s       | 4.8 TB/s       |
| Power (accelerator / host) |             | 912W/1500W      | 500W/6500W     | 700W/8850W     |
| # of accelerators per host |             | 1               | 8              | 8              |
| Scale-up                   | Domain size | 16 accelerators | 8 accelerators | 8 accelerators |
| network                    | Bandwidth   | 800 GB/s        | 450 GB/s       | 450 GB/s       |
| Scale-out<br>network       | Bandwidth   | 200 GB/s        | 50 GB/s        | 50 GB/s        |

(\*) *To achieve better performance per watt, we use a custom H100 configuration with a 500W power cap, which reduces peak performance from 1,000 TFLOPS at 700W.*

