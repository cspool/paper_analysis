# <span id="page-3-1"></span>III. PERFORMANCE DISSECTION OF UVM WITH CC

The core function of the UVM driver is to handle GPUgenerated page faults. As discussed earlier, this process (batch handling) comprises several key steps: fault fetching, preprocessing, fault servicing, and issuing GPU replay to resume SM execution. Various configuration parameters influence how faults are generated and handled throughout this pipeline. Given that batch handling critically impacts UVM efficiency, we analyze its behavior under CC in this section. Our experimental methodology is described in Section [VI.](#page-9-1)

Batch Handling Performance. The current UVM driver exposes a few tunable parameters. In this study, we focus on two such parameters: *fault batching count* (B<sup>f</sup> ) and *treebased prefetching threshold* (Pt), which are discussed in Section [II-A.](#page-1-0) Given that batch handling is already expensive under UVM [\[18\]](#page-13-12), we find that CC amplifies the cost further due to overheads related to encryption [\[5\]](#page-13-8). To address this problem, the first step is to reduce the number of fault batches handled at the CPU side. For a given workload, aggressive prefetching can help reducing the fault batches as it facilitates transfer of more data per fault batch handling. However, a small value of P<sup>t</sup> may also lead to unnecessary prefetching. Under CC, this unnecessary prefetching also increases the encryption overhead. Nevertheless, our experiments (discussed next) indicate that benefits of reducing the number of faults processed can outweigh the drawbacks of increased encryption overhead.

In Figure [3,](#page-3-3) we dissect the time spent by CPU to perform batch handling. The stacked bars show the time breakdown across batch handling components under different B<sup>f</sup> and Pt, while the points report the normalized total number of fault batches processed. We first observe that regardless of the choice of P<sup>t</sup> or B<sup>f</sup> , encryption occupies a large portion of the batch handling time on CPU. For example, in CNN with aggressive prefetching, encryption accounts for more than 70% of this handling time on CPU. In GEMM, the encryption share decreased from 44% to 16% when switching from P<sup>t</sup> = 1% to 91% at B<sup>f</sup> =256. However, reduced encryption time

<span id="page-3-2"></span><sup>5</sup> If messages m<sup>1</sup> and m<sup>2</sup> are encrypted under the same nonce, producing ciphertexts c<sup>1</sup> and c2, then c1⊕c<sup>2</sup> = m1⊕m2, which can reveal information about the plaintexts.

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Fig. 4: Fault generation rate and encryption throughput per batch of GEMM. Collected on real hardware.

does not always translate to better performance. In GEMM, the number of fault batches increases from 149 to 1310, an 8.7× growth (see Figure [4\(](#page-4-0)a)), while the number of GPU-generated faults grows more than 11×. It is evident that aggressive prefetching results in a relatively lower fault generation rate. As a result, although the encrypted data volume is smaller, the additional batch handling overhead dominates. For most applications, a smaller P<sup>t</sup> (i.e., aggressive prefetching) is therefore beneficial [\[27\]](#page-13-18). CNN workloads, however, present a counterexample. In CNN, the number of fault batches grew by only 8.3%, due to its memory access properties. Meanwhile, the amount of encrypted data is reduced by 96.8%, which far outweighs the modest increase in batch handling overhead.

Observation 1: Aggressive prefetching (P<sup>t</sup> = 1%) reduces costly GPU-CPU fault batch handling-related interactions. However, it shifts the burden to encryption – i.e., time spent on encryption can exceed 70% of the total fault batch handling time on CPU. Thus, encryption is a critical performance bottleneck under CC with UVM.

Encryption Mechanism Demystified. A key difference of UVM under CC is the *encrypted paging*, where pages are migrated over PCIe in encrypted form to prevent physical attacks. Since encryption is a major source of overhead (Observation 1), we examine its implementation inside the GPU driver. Listing [1](#page-4-1) shows a simplified version of the CPU-to-GPU migration routine; it runs during servicing of every fault batch. Each TDX page is encrypted with AES-GCM (line 4). Since TDX memory is protected by the TME-MK engine, each page is first decrypted on read before CC-required AES-GCM encryption. AES-GCM always operates on 4KB pages and is bound to a push channel, where an IV is managed through *synchronization*. In our evaluation, TME-MK adds only about 40 cycles to memory-access latency thanks to its dedicated hardware implementation. After encryption, a CE copy command is injected into the channel (line 6), which triggers a DMA transfer and asynchronous GPU-side decryption that can be pipelined. As a result, decryption latency on the GPU can be hidden.

Moreover, encryption differs substantially across GPU

memory-management paths (e.g., UVM vs. non-UVM [\[6\]](#page-13-9)) that limits which optimizations apply to UVM. In GPUbased CC, crypto-libraries are used in a *data plane* for CPU– GPU memory movement (e.g., AES encryption) and a *control plane* for CC services (e.g., provisioning). For UVM, the *kernel-space* NVIDIA driver exposes a *Cryptography Services Library (CSL)* interface [\[64\]](#page-14-8). Such CPU-side encryption, which lies on the critical path (as shown in Listing [1\)](#page-4-1), is therefore expected to rely on internal CSL calls, rather than user-space OpenSSL [\[107\]](#page-15-29). Although the call chain is proprietary, the net effect is that UVM encryption is realized through the Linux Kernel Crypto API (LCA) [\[108\]](#page-15-30), which we confirm using bpftrace [\[109\]](#page-15-31) tracing of CSL interactions with libspdm [\[110\]](#page-15-32), [\[111\]](#page-15-33) and kernel AEAD encryption (crypto\_aead\_encrypt). The thread model of this kernel path also matters. Replayable UVM fault servicing (Section [II-A\)](#page-1-0) is commonly funneled through a driver-managed work queue that is serviced by a single kernel thread. As a result, encryption on the UVM fault-service chain tends to inherit this *serialization*. This structure can limit parallelism in practice and shapes which optimizations are feasible for UVM workloads. Kernel modules also do not dynamically link against user-space libcrypto, so upgrading OpenSSL does not directly affect nvidia\_uvm.ko. In contrast, for non-UVM applications, OpenSSL covers data-plane encryption, as confirmed by prior studies [\[5\]](#page-13-8), [\[6\]](#page-13-9). Unlike the kernelside UVM path, user-space cryptography can exploit CPU parallelism via standard threading (e.g., std::thread in [\[6\]](#page-13-9)). We compare these implementations in Section [VIII-A.](#page-11-0)

Observation 2: Under GPU-based CC, UVM paging encryption is a kernel-space, Linux Kernel Crypto API (LCA)-backed path that operates synchronously at 4KB page granularity with IV synchronization. This differs from user-space OpenSSL-based bulk encryption and inherently limits parallelization opportunities.

```
1 va_bb = dma_alloc_coherent();
2 for_each_va_block_page_in_region(page_index, region) {
3 void *va_src = kmap(src_page);
4 uvm_cc_cpu_encrypt(push->channel, va_bb, va_src, iv,
     ,→ PAGE_SIZE, va_tag);
5 kunmap(src_page);
6 gpu->parent->ce->decrypt(push, gpu_dst, va_bb, PAGE_SIZE,
     ,→ va_tag);
7 }
```

Listing 1: Demonstration of CPU-side encryption and GPUside decryption process in the nvidia-uvm driver.

Encryption Performance. We measure the actual UVM encryption throughput and, as expected, it remains consistent across applications. Taking GEMM as a representative case, Figure [4\(](#page-4-0)b) shows the throughput per fault batch. The average values are 1.28 GB/s (i.e., 2.98 µs for encrypting a 4 KB page) and 1.24 GB/s for P<sup>t</sup> = 1% and 91%, respectively. This is much lower than non-UVM CC data migration bandwidth (around 3.03 GB/s [\[5\]](#page-13-8)). Thus, both the implementation and the performance of encryption present a major obstacle to optimizing UVM performance.

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Fig. 5: Pre-Encryption from LÆGIS increases encryption activity leading to improved performance (i.e., saved cycles).

<span id="page-5-2"></span>![](_page_5_Figure_2.jpeg)

Fig. 6: Bars show the fraction of CPU time during which the driver thread is active versus (true) idle. Lines show the average idle duration between consecutive batches and the average encryption time per batch. The left y-axis shows time (ns) in log scale. the x-axis shows the  $\mathcal{P}_t$  level, H, M and L stands for 1%, 51% and 91%. Collected on real hardware.

**Observation 3:** GPU CC presents slow (1.3 GB/s) and synchronous software encryption in UVM, placing it on the critical path. Such low throughput presents significant performance overhead. Note that TDX hardware for AES adds only 40 cycles. Thus, the bottleneck lies in the software encryption path.

