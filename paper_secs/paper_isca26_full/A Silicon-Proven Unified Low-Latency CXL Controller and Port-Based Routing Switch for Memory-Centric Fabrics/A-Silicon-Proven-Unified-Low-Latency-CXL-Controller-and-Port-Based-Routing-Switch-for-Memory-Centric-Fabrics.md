# A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics

Miryeong Kwon, Seungjun Lee, Donghyun Gouk, Hongjoo Jung, Eojin Ryu, Seyeong Huh, Junseok Moon, Hyein Woo, Junhee Kim, Kyungkuk Nam, Jinwoo Baek, Hyunkyu Choi, Woojin Choi, Yongjin Cho, Myoungsoo Jung

Panmnesia, Inc.

https://panmnesia.com

*Abstract*—Compute Express Link (CXL) enables composable and memory-centric fabrics, but existing controllers inherit PCIederived hierarchy, boundary buffering, and deep pipelines, which constrain scalability and increase latency. This work presents a unified low-latency CXL controller and a port-based routing (PBR) switch. The controller integrates the physical, link, and transaction layers into a single pipeline operating under a unified timing reference, removing high-overhead layer-level synchronization.

A hardware-driven conversion and routing path is integrated into each port of the switch. All steps, including HBR to PBR translation, header reconstruction, routing lookup, and arbitration, are issued through fixed-cycle hardware pipelines without firmware involvement. An internal non-blocking on-chip network supports parallel forwarding and maintains constant per-hop delay regardless of port count or topology depth.

A silicon prototype fabricated in a 4 nm process achieves roughly 2.1× latency reduction and preserves deterministic behavior under congestion and multi-hop operation. Systemlevel evaluation with database and microservice workloads shows up to 2× higher throughput than HBR-based designs. These results indicate that a unified controller and hardware-automated PBR switch provide a deterministic and scalable CXL fabric for multi-host memory pooling and sharing.

# A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics

Miryeong Kwon, Seungjun Lee, Donghyun Gouk, Hongjoo Jung, Eojin Ryu, Seyeong Huh, Junseok Moon, Hyein Woo, Junhee Kim, Kyungkuk Nam, Jinwoo Baek, Hyunkyu Choi, Woojin Choi, Yongjin Cho, Myoungsoo Jung

Panmnesia, Inc.

https://panmnesia.com

*Abstract*—Compute Express Link (CXL) enables composable and memory-centric fabrics, but existing controllers inherit PCIederived hierarchy, boundary buffering, and deep pipelines, which constrain scalability and increase latency. This work presents a unified low-latency CXL controller and a port-based routing (PBR) switch. The controller integrates the physical, link, and transaction layers into a single pipeline operating under a unified timing reference, removing high-overhead layer-level synchronization.

A hardware-driven conversion and routing path is integrated into each port of the switch. All steps, including HBR to PBR translation, header reconstruction, routing lookup, and arbitration, are issued through fixed-cycle hardware pipelines without firmware involvement. An internal non-blocking on-chip network supports parallel forwarding and maintains constant per-hop delay regardless of port count or topology depth.

A silicon prototype fabricated in a 4 nm process achieves roughly 2.1× latency reduction and preserves deterministic behavior under congestion and multi-hop operation. Systemlevel evaluation with database and microservice workloads shows up to 2× higher throughput than HBR-based designs. These results indicate that a unified controller and hardware-automated PBR switch provide a deterministic and scalable CXL fabric for multi-host memory pooling and sharing.

