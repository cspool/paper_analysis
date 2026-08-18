# B. Existing NVMe/TCP initiator solutions

1) Software-based approach: Software-based approaches, such as the Linux kernel NVMe/TCP initiator [1] and the SPDK NVMe/TCP initiator [65], deliver reasonable performance with optimizations such as polling, zero copy, and kernel bypassing [73, 97, 108]. The approaches consist of two components in general: an NVMe/TCP initiator stack that converts NVMe commands and completions into NVMe/TCP PDUs, and a TCP stack that transmits PDUs. Figure 3a shows its architecture. First, the host submits NVMe commands to the initiator stack. Then, the NVMe/TCP initiator stack constructs PDUs and writes them to TCP sockets. The TCP stack encapsulates the PDUs into TCP packets and transmits them to the remote target. Responses follow the reverse path.

2) On-device sidecore-based approach: On-device sidecore-based approaches offload the storage disaggregation software

TABLE I: Comparison of NVMe/TCP optimizations.

|                        | SPDK [37] | ANO [90] | Sidecore <sup>†</sup> [9] | Sidecore w/<br>Aceel.† [20] | NTI<br>(Ours) |
|------------------------|-----------|----------|---------------------------|-----------------------------|---------------|
| Host-efficiency        | X         | Δ        | О                         | О                           | O             |
| Performance            | 0         | Δ        | X                         | X                           | O             |
| Form factor            | -         | HHHL     | FHHL                      | FHHL                        | HHHL          |
| Power                  | -         | 75W      | 150W                      | 150W                        | 75W           |
| Interoperability       | △‡        | X        | O                         | O                           | O             |
| Operational resilience | 0         | O        | O                         | O                           | O             |

<sup>†:</sup> Bluefield3-DPU

stack from the host CPU to sidecores inside the device [25, 70, 84]. BlueField SNAP [9] is a representative design; it runs the full stack on the embedded ARM cores and adds the storage-emulation feature so that the PCIe card appears as a standard NVMe PCIe device to the host. Figure 3b shows its architecture. The frontend begins with the host NVMe interface, a hardware logic that exposes the DPU card as an NVMe PCIe device, enabling the NVMe transactions. When a host submits an NVMe command, the storage emulation stack fetches it. The interposition layer then translates the NVMe command into a generic block request and routes it to the backend. The backend operates in the same way as the software-based approach described in Section II-B1. Responses take the reverse path.

3) Hardware acceleration techniques: Besides relying only on sidecores in DPUs, there are techniques that accelerate specific functions with hardware. Unlike SNAP, these techniques do not target the entire storage-disaggregation stack. Instead, they focus on individual layers—for example, the storage virtualization stack [9, 52, 69], the NVMe/TCP stack [9, 90], and the TCP stack [38, 47]. Specifically within the NVMe/TCP stack, existing solutions fail to overcome fundamental architectural constraints. ANO [90], for instance, is severely restricted in its partial NVMe/TCP offload capabilities. Although it bypasses memory copies for in-order RX packets to write directly to the NVMe data buffer, this mechanism yields suboptimal small I/O performance, which remains heavily hindered by the kernel NVMe/TCP and TCP stack overhead. Crucially, ANO necessitates kernel code patching, which not only impedes deployability but also imposes significant vendor lock-in risks. Similarly, SNAP-based XLIO [20] suffers from inherent hardware and architectural limitations. Despite attempting to mitigate overheads via off-chip memory bypassing in the TX path and a user-level TCP stack to reduce context switching, its overall effectiveness is compromised. Performance remains constrained because zero-copy support is limited only to TX, and the embedded cores lack sufficient compute capability required to sustain high-performance workloads.

#### III. MOTIVATION

In this section, we introduce the motivation behind NTI. First, we identify the critical efficiency and performance limitations of existing solutions (Sections III-A, III-B). We then highlight the challenges that must be resolved to realize a solution with practical deployability (Section III-C).

<sup>‡:</sup> Unable to support kernel applications (userspace block interface)

![](_page_3_Figure_0.jpeg)

Fig. 4: CPU usage breakdown of the SPDK NVMe/TCP initiator with 4KiB random read and write FIO.

## *A. Motivation 1: CPU-inefficient software-based approaches*

While existing software-based storage disaggregation solutions achieve high performance by aggressively optimizing the I/O path, they inherently impose significant demands on CPU resources. As shown in Figure 1, the SPDK NVMe /TCP initiator requires 24 cores to deliver 200 Gbps line-rate performance, and is expected to require 96 cores for emerging 800 Gbps networks. This measurement was obtained using FIO [13] benchmarks configured with 4KiB random read I/O, 32 threads, and 256 queue depth. Figure 4 shows the CPU usage of the SPDK NVMe/TCP initiator and highlights that it is dominated by both NVMe/TCP and TCP stacks. These stacks impose a significant CPU overhead from per-packet processing, memory operations, and interactions between the storage and networking layers, which worsens at higher throughput and limits scalability. Unfortunately, this CPU inefficiency will deepen in the future, as the gap between I/O bandwidth and CPU processing capability continues to widen [67, 96, 110]. While PCIe Gen8 and 1600 Gbps Ethernet are expected to arrive in 2028 [88], CPU compute capacity will see only modest improvements. Without addressing this host CPU bottleneck, future storage disaggregation will be unable to fully utilize the next-generation network link. Thus, since the significant CPU overhead stems from both NVMe/TCP and TCP stacks, this inefficiency motivates the full offload of the storage disaggregation stack to dedicated hardware.

# B. Existing NVMe/TCP initiator solutions

1) Software-based approach: Software-based approaches, such as the Linux kernel NVMe/TCP initiator [1] and the SPDK NVMe/TCP initiator [65], deliver reasonable performance with optimizations such as polling, zero copy, and kernel bypassing [73, 97, 108]. The approaches consist of two components in general: an NVMe/TCP initiator stack that converts NVMe commands and completions into NVMe/TCP PDUs, and a TCP stack that transmits PDUs. Figure 3a shows its architecture. First, the host submits NVMe commands to the initiator stack. Then, the NVMe/TCP initiator stack constructs PDUs and writes them to TCP sockets. The TCP stack encapsulates the PDUs into TCP packets and transmits them to the remote target. Responses follow the reverse path.

2) On-device sidecore-based approach: On-device sidecore-based approaches offload the storage disaggregation software

TABLE I: Comparison of NVMe/TCP optimizations.

|                        | SPDK [37] | ANO [90] | Sidecore <sup>†</sup> [9] | Sidecore w/<br>Aceel.† [20] | NTI<br>(Ours) |
|------------------------|-----------|----------|---------------------------|-----------------------------|---------------|
| Host-efficiency        | X         | Δ        | О                         | О                           | O             |
| Performance            | 0         | Δ        | X                         | X                           | O             |
| Form factor            | -         | HHHL     | FHHL                      | FHHL                        | HHHL          |
| Power                  | -         | 75W      | 150W                      | 150W                        | 75W           |
| Interoperability       | △‡        | X        | O                         | O                           | O             |
| Operational resilience | 0         | O        | O                         | O                           | O             |

<sup>†:</sup> Bluefield3-DPU

stack from the host CPU to sidecores inside the device [25, 70, 84]. BlueField SNAP [9] is a representative design; it runs the full stack on the embedded ARM cores and adds the storage-emulation feature so that the PCIe card appears as a standard NVMe PCIe device to the host. Figure 3b shows its architecture. The frontend begins with the host NVMe interface, a hardware logic that exposes the DPU card as an NVMe PCIe device, enabling the NVMe transactions. When a host submits an NVMe command, the storage emulation stack fetches it. The interposition layer then translates the NVMe command into a generic block request and routes it to the backend. The backend operates in the same way as the software-based approach described in Section II-B1. Responses take the reverse path.

3) Hardware acceleration techniques: Besides relying only on sidecores in DPUs, there are techniques that accelerate specific functions with hardware. Unlike SNAP, these techniques do not target the entire storage-disaggregation stack. Instead, they focus on individual layers—for example, the storage virtualization stack [9, 52, 69], the NVMe/TCP stack [9, 90], and the TCP stack [38, 47]. Specifically within the NVMe/TCP stack, existing solutions fail to overcome fundamental architectural constraints. ANO [90], for instance, is severely restricted in its partial NVMe/TCP offload capabilities. Although it bypasses memory copies for in-order RX packets to write directly to the NVMe data buffer, this mechanism yields suboptimal small I/O performance, which remains heavily hindered by the kernel NVMe/TCP and TCP stack overhead. Crucially, ANO necessitates kernel code patching, which not only impedes deployability but also imposes significant vendor lock-in risks. Similarly, SNAP-based XLIO [20] suffers from inherent hardware and architectural limitations. Despite attempting to mitigate overheads via off-chip memory bypassing in the TX path and a user-level TCP stack to reduce context switching, its overall effectiveness is compromised. Performance remains constrained because zero-copy support is limited only to TX, and the embedded cores lack sufficient compute capability required to sustain high-performance workloads.

#### III. MOTIVATION

In this section, we introduce the motivation behind NTI. First, we identify the critical efficiency and performance limitations of existing solutions (Sections III-A, III-B). We then highlight the challenges that must be resolved to realize a solution with practical deployability (Section III-C).

<sup>‡:</sup> Unable to support kernel applications (userspace block interface)

![](_page_3_Figure_0.jpeg)

Fig. 4: CPU usage breakdown of the SPDK NVMe/TCP initiator with 4KiB random read and write FIO.

## *A. Motivation 1: CPU-inefficient software-based approaches*

While existing software-based storage disaggregation solutions achieve high performance by aggressively optimizing the I/O path, they inherently impose significant demands on CPU resources. As shown in Figure 1, the SPDK NVMe /TCP initiator requires 24 cores to deliver 200 Gbps line-rate performance, and is expected to require 96 cores for emerging 800 Gbps networks. This measurement was obtained using FIO [13] benchmarks configured with 4KiB random read I/O, 32 threads, and 256 queue depth. Figure 4 shows the CPU usage of the SPDK NVMe/TCP initiator and highlights that it is dominated by both NVMe/TCP and TCP stacks. These stacks impose a significant CPU overhead from per-packet processing, memory operations, and interactions between the storage and networking layers, which worsens at higher throughput and limits scalability. Unfortunately, this CPU inefficiency will deepen in the future, as the gap between I/O bandwidth and CPU processing capability continues to widen [67, 96, 110]. While PCIe Gen8 and 1600 Gbps Ethernet are expected to arrive in 2028 [88], CPU compute capacity will see only modest improvements. Without addressing this host CPU bottleneck, future storage disaggregation will be unable to fully utilize the next-generation network link. Thus, since the significant CPU overhead stems from both NVMe/TCP and TCP stacks, this inefficiency motivates the full offload of the storage disaggregation stack to dedicated hardware.

