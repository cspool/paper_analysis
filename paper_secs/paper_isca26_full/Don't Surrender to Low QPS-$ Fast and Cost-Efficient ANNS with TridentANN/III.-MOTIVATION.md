# III. MOTIVATION

## A. High QPS != High QPS/\$

With the single-node serving and relatively high QPS, one would assume the existing SSD-based ANNS systems should

<sup>1</sup>This and following examples refer to top-10 search with 90% recall on SIFT1B, which is recommended by the BigANN benchmark [14].

yield high cost-effectiveness (in terms of QPS/\$) as well. However, our following investigation indicates otherwise.

FusionANNS stores the compressed dataset in the GPU's device memory and facilitates the distance evaluations with its high processing power. This design achieves competitive throughput (i.e., 18K QPS referred to in the paper and 22.5K QPS of our reproduction described in Subsection VIII-A) but comes with a high price (i.e., ~\$8.4K and ~\$10.3K), resulting in a mere ~2.14-2.18 QPS/\$ cost-effectiveness as shown in Table I. The cost, including the prototype in their paper and our reproduction same as [64], is mainly driven by the highend GPU (e.g., a V100-32GB costs ~\$3.0K or an A6000-48GB costs ~\$5.0K) and the dual-socket CPU chassis (e.g., 64-core and 58-core server can cost ~\$5.4K and ~\$5.3K).

Note that we cannot simply reduce such costs with cheap GPUs and/or CPUs. Since FusionANNS relies on large and high-bandwidth GPU memory (the dominant factor in GPU prices [21]) for hosting compressed vectors of the whole datasets on the GPU necessarily (e.g., 32 GB device memory is required at least for SIFT1B as [69]), and high-end CPUs (e.g., 64-core from the dual socket) for launching enough requests to saturate the processing pipeline.

DiskANN and PipeANN do not require GPU assistance and therefore achieve better cost-effectiveness (i.e., ~2.46 QPS/\$ and ~3.14 QPS/\$). However, due to the multi-hop graphbased search, their QPS is also limited (i.e., 6.9K and 8.8K QPS) compared to FusionANNS. To improve throughput, a straightforward approach is to scale up the number of CPU cores. We therefore add another CPU and use a dual-socket motherboard (i.e., the second row of DiskANN and PipeANN in Table I). Extra cores do empower them to deliver 11.7K and 16.5K OPS, respectively. Yet, the OPS/\$ remains unchanged or even decreases, since the additional cores are not fully utilized to double the throughput. With higher processing power, more requests are launched concurrently, which creates more die-level collisions inside the NVMe SSDs and higher I/O latency [46]. As a result, the QPS does not scale proportionally with the additional CPU cores.

In addition to CPU scaling, we also scale the number of SSDs (third and fourth rows). Using two SSDs can alleviate die-level collisions with lower read latency and slightly increase throughput. DiskANN and PipeANN improve from 6.9K and 8.8K to 8.4K and 11.2K QPS (QPS/\$ from  $\sim$ 2.46 and  $\sim$ 3.14 to  $\sim$ 2.89 and  $\sim$ 3.86). However, further increasing SSDs yields diminishing returns (e.g., only to 9.9K/12.5K QPS with 8 SSDs), and QPS/\$ no longer grows. This is because graph-based search incurs inherent I/O serialization and is latency-bound rather than bandwidth-bound, which prevents effectively exploiting higher SSD-side aggregate bandwidth.

**SPANN** is an interesting case. The original setup (first row) shows both the lowest QPS (i.e., 1.8K QPS) and the lowest QPS/\$ (i.e., ~0.64 QPS/\$). Since it is a clustering-based system, the performance is therefore capped by the bandwidth. Hence, we further equip SPANN with more disks (i.e., second row with 8 NVMe SSDs), which results in significant improvements in both throughput (i.e., 15.4K QPS) and cost-

| System     | Hardware Setup                           | Throughput | Cost           | QPS/\$ |
|------------|------------------------------------------|------------|----------------|--------|
| FusionANNS | 2 CPU-32core & 1 SSD<br>1 GPU-32GB (REF) | 18K QPS    | ~\$8.4K        | ~2.14  |
|            | 2 CPU-28core & 1 SSD<br>1 GPU-48GB (REP) | 22.5K QPS  | ~\$10.3K       | ~2.18  |
|            | 1 CPU-28Core & 1 SSD                     | 6.9K QPS   | ~\$2.8K        | ~2.46  |
| DiskANN    | 2 CPU-28Core & 1 SSD                     | 11.7K QPS  | ~\$5.3K        | ~2.21  |
| DISKAININ  | 1 CPU-28Core & 2 SSD                     | 8.4K QPS   | ~\$2.9K        | ~2.89  |
|            | 1 CPU-28Core & 8 SSD                     | 9.9K QPS   | ~\$3.5K        | ~2.82  |
|            | 1 CPU-28Core & 1 SSD                     | 8.8K QPS   | ~\$2.8K        | ~3.14  |
| Di A NINI  | 2 CPU-28Core & 1 SSD                     | 16.5K QPS  | ~\$5.3K        | ~3.11  |
| PipeANN    | 1 CPU-28Core & 2 SSD                     | 11.2K QPS  | ~\$2.9K        | ~3.86  |
|            | 1 CPU-28Core & 8 SSD                     | 12.5K QPS  | ~\$3.5K        | ~3.57  |
|            | 1 CPU-28Core & 1 SSD                     | 1.8K QPS   | ~\$2.8K        | ~0.64  |
| CDANIN     | 1 CPU-28Core & 8 SSD                     | 15.4K QPS  | ~\$3.5K        | ~4.40  |
| SPANN      | 1 CPU-28Core & 16 SSD                    | 17.3K QPS  | ~\$4.3K        | ~4.02  |
|            | 2 CPU-28Core & 8 SSD                     | 16.9K QPS  | ~\$6K          | ~ 2.82 |
| TRIDENTANN | 1 CPU-28core & 8 SSD                     | 22 5W ODG  | ~\$4.3K        | ~7.79  |
| (Ours)     | 2 GPU-6GB                                | 33.5K QPS  | ~ <b>⊅4.3K</b> | ~1.19  |

TABLE I: Performance and hardware cost.

effectiveness (i.e., ~4.4 QPS/\$). The drastic increase reflects the importance of a balanced system design: scaling SSD bandwidth to cooperate with compute resources appropriately can achieve both high QPS and high QPS/\$. The third and fourth rows of SPANN further validate this assumption, where adding more SSDs (i.e., 16 SSDs) or more CPU cores (i.e., 2 CPUs) leads to diminished throughput improvements and reduced cost-effectiveness, as the marginal gain is outweighed by the additional hardware cost.

#### B. The Cannikin Law in Building ANNS

At a high level, the above analysis reveals that ANNS systems follow the classical "cannikin law" (a.k.a., the wooden barrel principle), which states that the performance (QPS and QPS/\$ in our context) of a system is determined by the weakest component. In other words, we have to carefully determine the combination of GPUs, CPUs, and NVMe SSDs to avoid bottlenecks or over-provisioning of resources. Specifically, we have learned three key insights.

First, what we need is GPUs' processing power, not their expensive device memory capacity. FusionANNS uses a highend GPU to accelerate computation-intensive tasks (e.g., distance calculation) and requires a large amount of device memory to store the compressed dataset, thereby leading to a low QPS/\$. Note that the price of modern GPUs is mainly driven by the size and type (e.g., HBM or GDDR) of their device memory [21]. For example, compared to the NVIDIA V100-32GB [7] used in original FusionANNS, two NVIDIA A2000 GPUs [10] have comparable processing power<sup>2</sup> ( $\sim$ 10% difference in TFLOPS), each with 6GB GDDR (12 GB in total), and together cost only nearly  $\sim 1/4$  of a V100-32GB GPU. Moreover, in our reproduction, the NVIDIA RTX A6000-48GB [11] is the minimum option (in the same series as the A2000 used in TRIDENTANN) that satisfies FusionANNS's requirement of at least 32GB device memory for 1B-scale datasets (e.g., SIFT1B). However, its price is about 6× that of two A2000 GPUs. Hence, if we can avoid the reliance on GPU memory, we can instead use cheaper

<sup>&</sup>lt;sup>2</sup>We use 2 A2000-6GB GPUs with 16 TFLOPS in total to pair with 8 SSDs for full interconnect bandwidth, and a single V100-32GB has 14 TFLOPS.

GPUs with smaller and lower-grade device memory while still harnessing the high processing power.

Second, scaling up the computation resources (e.g., CPU cores in DiskANN and PipeANN) can help improve throughput, but can incur diminishing returns. This is because the additional CPU cores are not fully utilized due to the latency overhead contributed by the conflicts inside the NVMe SSDs. This works vice versa as well (i.e., adding more SSDs is not always helpful, as shown in the third row of SPANN). Hence, we should always match the number of CPU cores with the number of NVMe SSDs to prevent over-provisioning of resources, which hampers QPS/\$.

Third, compared to expensive GPUs and CPUs, scaling up the I/O bandwidth with NVMe SSDs is much more costeffective. Modern NVMe SSDs can provide high I/O bandwidth (e.g., 7 GiB/s for a single drive) at a low cost (e.g., \$100 for a 1-TiB PCIe-Gen4 NVMe SSD [19]). This enlightens us to rethink the design principle of ANNS systems: instead of relying on expensive computational resources, we can scale up the I/O bandwidth with low-cost NVMe SSD arrays to avoid the urgent requirement of CPU in DiskANN and PipeANN, or the large capacity of GPU memory in FusionANNS.

# III. MOTIVATION

## A. High QPS != High QPS/\$

With the single-node serving and relatively high QPS, one would assume the existing SSD-based ANNS systems should

<sup>1</sup>This and following examples refer to top-10 search with 90% recall on SIFT1B, which is recommended by the BigANN benchmark [14].

yield high cost-effectiveness (in terms of QPS/\$) as well. However, our following investigation indicates otherwise.

FusionANNS stores the compressed dataset in the GPU's device memory and facilitates the distance evaluations with its high processing power. This design achieves competitive throughput (i.e., 18K QPS referred to in the paper and 22.5K QPS of our reproduction described in Subsection VIII-A) but comes with a high price (i.e., ~\$8.4K and ~\$10.3K), resulting in a mere ~2.14-2.18 QPS/\$ cost-effectiveness as shown in Table I. The cost, including the prototype in their paper and our reproduction same as [64], is mainly driven by the highend GPU (e.g., a V100-32GB costs ~\$3.0K or an A6000-48GB costs ~\$5.0K) and the dual-socket CPU chassis (e.g., 64-core and 58-core server can cost ~\$5.4K and ~\$5.3K).

Note that we cannot simply reduce such costs with cheap GPUs and/or CPUs. Since FusionANNS relies on large and high-bandwidth GPU memory (the dominant factor in GPU prices [21]) for hosting compressed vectors of the whole datasets on the GPU necessarily (e.g., 32 GB device memory is required at least for SIFT1B as [69]), and high-end CPUs (e.g., 64-core from the dual socket) for launching enough requests to saturate the processing pipeline.

DiskANN and PipeANN do not require GPU assistance and therefore achieve better cost-effectiveness (i.e., ~2.46 QPS/\$ and ~3.14 QPS/\$). However, due to the multi-hop graphbased search, their QPS is also limited (i.e., 6.9K and 8.8K QPS) compared to FusionANNS. To improve throughput, a straightforward approach is to scale up the number of CPU cores. We therefore add another CPU and use a dual-socket motherboard (i.e., the second row of DiskANN and PipeANN in Table I). Extra cores do empower them to deliver 11.7K and 16.5K OPS, respectively. Yet, the OPS/\$ remains unchanged or even decreases, since the additional cores are not fully utilized to double the throughput. With higher processing power, more requests are launched concurrently, which creates more die-level collisions inside the NVMe SSDs and higher I/O latency [46]. As a result, the QPS does not scale proportionally with the additional CPU cores.

In addition to CPU scaling, we also scale the number of SSDs (third and fourth rows). Using two SSDs can alleviate die-level collisions with lower read latency and slightly increase throughput. DiskANN and PipeANN improve from 6.9K and 8.8K to 8.4K and 11.2K QPS (QPS/\$ from  $\sim$ 2.46 and  $\sim$ 3.14 to  $\sim$ 2.89 and  $\sim$ 3.86). However, further increasing SSDs yields diminishing returns (e.g., only to 9.9K/12.5K QPS with 8 SSDs), and QPS/\$ no longer grows. This is because graph-based search incurs inherent I/O serialization and is latency-bound rather than bandwidth-bound, which prevents effectively exploiting higher SSD-side aggregate bandwidth.

**SPANN** is an interesting case. The original setup (first row) shows both the lowest QPS (i.e., 1.8K QPS) and the lowest QPS/\$ (i.e., ~0.64 QPS/\$). Since it is a clustering-based system, the performance is therefore capped by the bandwidth. Hence, we further equip SPANN with more disks (i.e., second row with 8 NVMe SSDs), which results in significant improvements in both throughput (i.e., 15.4K QPS) and cost-

| System     | Hardware Setup                           | Throughput | Cost           | QPS/\$ |
|------------|------------------------------------------|------------|----------------|--------|
| FusionANNS | 2 CPU-32core & 1 SSD<br>1 GPU-32GB (REF) | 18K QPS    | ~\$8.4K        | ~2.14  |
|            | 2 CPU-28core & 1 SSD<br>1 GPU-48GB (REP) | 22.5K QPS  | ~\$10.3K       | ~2.18  |
|            | 1 CPU-28Core & 1 SSD                     | 6.9K QPS   | ~\$2.8K        | ~2.46  |
| DiskANN    | 2 CPU-28Core & 1 SSD                     | 11.7K QPS  | ~\$5.3K        | ~2.21  |
| DISKAININ  | 1 CPU-28Core & 2 SSD                     | 8.4K QPS   | ~\$2.9K        | ~2.89  |
|            | 1 CPU-28Core & 8 SSD                     | 9.9K QPS   | ~\$3.5K        | ~2.82  |
|            | 1 CPU-28Core & 1 SSD                     | 8.8K QPS   | ~\$2.8K        | ~3.14  |
| Di A NINI  | 2 CPU-28Core & 1 SSD                     | 16.5K QPS  | ~\$5.3K        | ~3.11  |
| PipeANN    | 1 CPU-28Core & 2 SSD                     | 11.2K QPS  | ~\$2.9K        | ~3.86  |
|            | 1 CPU-28Core & 8 SSD                     | 12.5K QPS  | ~\$3.5K        | ~3.57  |
|            | 1 CPU-28Core & 1 SSD                     | 1.8K QPS   | ~\$2.8K        | ~0.64  |
| CDANIN     | 1 CPU-28Core & 8 SSD                     | 15.4K QPS  | ~\$3.5K        | ~4.40  |
| SPANN      | 1 CPU-28Core & 16 SSD                    | 17.3K QPS  | ~\$4.3K        | ~4.02  |
|            | 2 CPU-28Core & 8 SSD                     | 16.9K QPS  | ~\$6K          | ~ 2.82 |
| TRIDENTANN | 1 CPU-28core & 8 SSD                     | 22 5W ODG  | ~\$4.3K        | ~7.79  |
| (Ours)     | 2 GPU-6GB                                | 33.5K QPS  | ~ <b>⊅4.3K</b> | ~1.19  |

TABLE I: Performance and hardware cost.

effectiveness (i.e., ~4.4 QPS/\$). The drastic increase reflects the importance of a balanced system design: scaling SSD bandwidth to cooperate with compute resources appropriately can achieve both high QPS and high QPS/\$. The third and fourth rows of SPANN further validate this assumption, where adding more SSDs (i.e., 16 SSDs) or more CPU cores (i.e., 2 CPUs) leads to diminished throughput improvements and reduced cost-effectiveness, as the marginal gain is outweighed by the additional hardware cost.

#### B. The Cannikin Law in Building ANNS

At a high level, the above analysis reveals that ANNS systems follow the classical "cannikin law" (a.k.a., the wooden barrel principle), which states that the performance (QPS and QPS/\$ in our context) of a system is determined by the weakest component. In other words, we have to carefully determine the combination of GPUs, CPUs, and NVMe SSDs to avoid bottlenecks or over-provisioning of resources. Specifically, we have learned three key insights.

First, what we need is GPUs' processing power, not their expensive device memory capacity. FusionANNS uses a highend GPU to accelerate computation-intensive tasks (e.g., distance calculation) and requires a large amount of device memory to store the compressed dataset, thereby leading to a low QPS/\$. Note that the price of modern GPUs is mainly driven by the size and type (e.g., HBM or GDDR) of their device memory [21]. For example, compared to the NVIDIA V100-32GB [7] used in original FusionANNS, two NVIDIA A2000 GPUs [10] have comparable processing power<sup>2</sup> ( $\sim$ 10% difference in TFLOPS), each with 6GB GDDR (12 GB in total), and together cost only nearly  $\sim 1/4$  of a V100-32GB GPU. Moreover, in our reproduction, the NVIDIA RTX A6000-48GB [11] is the minimum option (in the same series as the A2000 used in TRIDENTANN) that satisfies FusionANNS's requirement of at least 32GB device memory for 1B-scale datasets (e.g., SIFT1B). However, its price is about 6× that of two A2000 GPUs. Hence, if we can avoid the reliance on GPU memory, we can instead use cheaper

<sup>&</sup>lt;sup>2</sup>We use 2 A2000-6GB GPUs with 16 TFLOPS in total to pair with 8 SSDs for full interconnect bandwidth, and a single V100-32GB has 14 TFLOPS.

GPUs with smaller and lower-grade device memory while still harnessing the high processing power.

Second, scaling up the computation resources (e.g., CPU cores in DiskANN and PipeANN) can help improve throughput, but can incur diminishing returns. This is because the additional CPU cores are not fully utilized due to the latency overhead contributed by the conflicts inside the NVMe SSDs. This works vice versa as well (i.e., adding more SSDs is not always helpful, as shown in the third row of SPANN). Hence, we should always match the number of CPU cores with the number of NVMe SSDs to prevent over-provisioning of resources, which hampers QPS/\$.

Third, compared to expensive GPUs and CPUs, scaling up the I/O bandwidth with NVMe SSDs is much more costeffective. Modern NVMe SSDs can provide high I/O bandwidth (e.g., 7 GiB/s for a single drive) at a low cost (e.g., \$100 for a 1-TiB PCIe-Gen4 NVMe SSD [19]). This enlightens us to rethink the design principle of ANNS systems: instead of relying on expensive computational resources, we can scale up the I/O bandwidth with low-cost NVMe SSD arrays to avoid the urgent requirement of CPU in DiskANN and PipeANN, or the large capacity of GPU memory in FusionANNS.

