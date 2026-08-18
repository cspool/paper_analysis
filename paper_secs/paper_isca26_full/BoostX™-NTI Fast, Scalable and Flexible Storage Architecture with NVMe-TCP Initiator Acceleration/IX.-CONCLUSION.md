# IX. CONCLUSION

In this work, we present NTI, a fast and scalable FPGAbased DPU solution for NVMe/TCP initiators. NTI offloads the storage disaggregation stack onto a DPU card, achieving efficiency, performance, and deployability. By executing I/O path in hardware, it eliminates compute and memory bottlenecks. Moreover, it ensures deployment in real-world datacenters. Our evaluation demonstrates that NTI outperforms baseline solutions across diverse workloads.

## ACKNOWLEDGMENT

This work was partly supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2024-00395134, DPU-Centric Datacenter Architecture for Next-Generation AI Devices; and No.RS-2025-02264029, Implementation and Validation of an AI Semiconductor-Based Data Center Composable Cluster Infrastructure).

## REFERENCES

- [1] Kernel nvme/tcp host. [Online]. Available: https: //docs.redhat.com/en/documentation/red hat enterprise linux/9/ html/managing storage devices/configuring-nvme-over-fabrics-usingnvme-tcp managing-storage-devices
- [2] pgbench. [Online]. Available: https://www.postgresql.org/docs/current/ pgbench.html
- [3] "Tcp congestion control," https://www.rfc-editor.org/rfc/rfc5681.html, 2009.
- [4] (2025, August) Amd alveo™ adaptable accelerator cards. [Online]. Available: https://www.amd.com/en/products/accelerators/alveo.html
- [5] (2025, December) Amd cips sysmon. [Online]. Available: https: //docs.amd.com/r/en-US/pg352-cips/System-Monitor
- [6] (2025, December) Amd control interfaces and processing system. [Online]. Available: https://www.amd.com/en/products/adaptive-socsand-fpgas/intellectual-property/cips.html
- [7] (2025, August) Amd pensando™ dsc3-400 distributed services card. [Online]. Available: https://www.amd.com/content/dam/amd/ en/documents/pensando-technical-docs/product-briefs/pensando-dsc3 product-brief.pdf
- [8] (2025, August) Amd versal™ adaptive socs. [Online]. Available: https://www.amd.com/en/products/adaptive-socs-and-fpgas/ versal.html#overview
- [9] (2025, August) Bluefield snap. [Online]. Available: https://docs.nvidia. com/networking/display/bluefielddpuosv470/bluefield+snap
- [10] (2025, August) Clouddc superserver sys-122c-tn. [Online]. Available: https://www.supermicro.com/en/products/system/clouddc/1u/sys-122c-tn
- [11] (2025, August) Dell emc poweredge r650 spec sheet. [Online]. Available: https://www.delltechnologies.com/asset/nl-nl/products/servers/ technical-support/dell-emc-poweredge-r650-spec-sheet.pdf
- [12] (2025, August) Dell powerdge r750. [Online]. Available: https: //www.dell.com/ko-kr/shop/cty/pdp/spd/poweredge-r750/per75010a
- [13] (2025, August) fio flexible io tester. [Online]. Available: https: //git.kernel.dk/cgit/fio
- [14] (2025, August) Fpga-accelerated nvme storage solutions using the bittware 250 series accelerators. [Online]. Available: https: //www.bittware.com/ko/resources/nvme-storage/
- [15] (2025, August) Fpga-based smartnics. [Online]. Available: https: //www.napatech.com/products
- [16] (2025, August) Hpe proliant dl385 gen10 plus v2 7313 3.0ghz 16-core 1p 32gb-r mr416i-a 8sff 800w ps server. [Online]. Available: https://buy.hpe.com/kr/ko/compute/rack-servers/proliantdl300-servers/proliant-dl385-server/hpe-proliant-dl385-gen10-plusv2-server/p/1013291283
- [17] (2025, August) Hyper superserver sys-221h-tnr. [Online]. Available: https://www.supermicro.com/en/products/system/hyper/2u/ sys-221h-tnr
- [18] (2025, August) Intel® xeon® gold 6348 processor. [Online]. Available: https://www.intel.com/content/www/us/en/products/sku/212456/intelxeon-gold-6348-processor-42m-cache-2-60-ghz/specifications.html
- [19] (2025, August) Intel® xeon® gold processor. [Online]. Available: https://www.intel.com/content/www/us/en/products/details/ processors/xeon/scalable/gold.html
- [20] (2025, August) Introduction to xlio. [Online]. Available: https: //docs.nvidia.com/networking/display/xliov3312/introduction+to+xlio
- [21] (2025, August) Kioxia. [Online]. Available: www.kioxia.com

- [22] (2025, August) Kvm. [Online]. Available: https://linux-kvm.org/page/ Main Page
- [23] (2025, August) Micron. [Online]. Available: https://www.micron.com/
- [24] (2025, August) Netapp. [Online]. Available: https://www.netapp.com/
- [25] (2025, August) Nvidia bluefield-3 networking platform user guide. [Online]. Available: https://docs.nvidia.com/networking/display/bf3dpu
- [26] (2025, August) Nvm express® base specification. [Online]. Available: https://nvmexpress.org/specification/nvm-express-base-specification
- [27] (2025, August) Nvm express® base specification 5.14.1.12 asymmetric namespace access. [Online]. Available: https: //nvmexpress.org/specification/nvm-express-base-specification
- [28] (2025, August) Nvme over fabrics (of) specification (historical reference only). [Online]. Available: https://nvmexpress.org/ specification/nvme-of-specification
- [29] (2025, August) Nvme over tcp transport specification. [Online]. Available: https://nvmexpress.org/specification/tcp-transport-specification
- [30] (2025, August) Ocp nic 3.0: Thermal considerations. [Online]. Available: https://146a55aca6f00848c565 a7635525d40ac1c70300198708936b4e.ssl.cf1.rackcdn.com/images/ 5c41ae0635633a115cc8ec42fa4618801362ae93.pdf
- [31] (2025, August) Ocp oai system liquid cooling guidelines. [Online]. Available: https://www.opencompute.org/documents/oai-systemliquid-cooling-guidelines-in-ocp-template-mar-3-2023-update-pdf
- [32] (2025, August) Purestorage. [Online]. Available: https://www. purestorage.com/
- [33] (2025, August) Qemu. [Online]. Available: https://www.qemu.org/
- [34] (2025, August) Samsung. [Online]. Available: https://www.samsung. com/sec/
- [35] (2025, August) Skhynix. [Online]. Available: https://www.skhynix. com/
- [36] (2025, August) Solidigm. [Online]. Available: https://www.solidigm. com/
- [37] (2025, August) Spdk. [Online]. Available: https://spdk.io
- [38] (2025, August) Tcp offload engine (toe). [Online]. Available: https://www.chelsio.com/nic/tcp-offload-engine/
- [39] (2025, August) Transmission control protocol. [Online]. Available: https://www.ietf.org/rfc/rfc793.txt
- [40] (2025, August) vmware esxi. [Online]. Available: https://www.vmware. com/products/cloud-infrastructure/vsphere/
- [41] (2025, August) Western digital rapidflex tm c1000 nvme-of tm adapter data sheet. [Online]. Available: https://www.westerndigital.com/ko-kr/products/data-center-platforms/ rapidflex-c1000-nvme-controller?sku=c1000-nvme-controller
- [42] (2026, April) Amd pensando™ salina dpu. [Online]. Available: https://www.amd.com/content/dam/amd/en/documents/pensandotechnical-docs/product-briefs/pensando-salina-product-brief.pdf
- [43] (2026, April) Intel® infrastructure processing unit adapter e2100-ccqda2hl. [Online]. Available: https: //www.intel.com/content/www/us/en/content-details/832490/intelinfrastructure-processing-unit-adapter-e2100-ccqda2hl.html
- [44] P. Antonopoulos, A. Budovski, C. Diaconu, A. Hernandez Saenz, J. Hu, H. Kodavalla, D. Kossmann, S. Lingam, U. F. Minhas, N. Prakash *et al.*, "Socrates: The new sql server in the cloud," in *Proceedings of the 2019 International Conference on Management of Data*, 2019, pp. 1743–1756.
- [45] L. A. Barroso, U. Holzle, and P. Ranganathan, ¨ *The Datacenter as a Computer: Designing Warehouse-Scale Machines (3rd Edition)*, ser. Synthesis Lectures on Computer Architecture, M. Martonosi, Ed. Morgan & Claypool, 2018, vol. 46, https://pages.cs.wisc.edu/∼shivaram/ cs744-readings/dc-computer-v3.pdf.
- [46] Bittware, "250-soc," https://www.bittware.com/resources/buildingnvme-over-fabrics/.
- [47] J. Boo, Y. Chung, E. Baek, S. Na, C. Kim, and J. Kim, "F4t: A fast and flexible fpga-based full-stack tcp acceleration framework," in *Proceedings of the 50th Annual International Symposium on Computer Architecture*, 2023, pp. 1–13.
- [48] Q. Cai, S. Chaudhary, M. Vuppalapati, J. Hwang, and R. Agarwal, "Understanding host network stack overheads," in *Proceedings of the 2021 ACM SIGCOMM 2021 Conference*, 2021, pp. 65–77.
- [49] Chelsio, "Roce fails to scale," https://www.chelsio.com/wp-content/ uploads/resources/RoCE-Deployment-Challenges-for-Clouds.pdf, 2015.

- [50] Y. Chen, Z. Jin, Y. Wang, Y. Chen, J. Xu, H. Yu, J. Chen, W. Lin, K. Fang, K. Zhang *et al.*, "Nvmepass: A lightweight, highperformance and scalable nvme virtualization architecture with i/o queues passthrough," in *2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)*. IEEE, 2025, pp. 1395– 1407.
- [51] Y. Chen, Z. Jin, Y. Wang, Y. Chen, H. Yu, J. Xu, J. Chen, W. Lin, K. Fang, C. Wei *et al.*, "High-performance and scalable software-based nvme virtualization mechanism with i/o queues passthrough," *arXiv preprint arXiv:2304.05148*, 2023.
- [52] Y. Chen, J. Xu, C. Wei, Y. Wang, X. Yuan, Y. Zhang, X. Yu, Y. Chen, Z. Wang, S. He *et al.*, "Bm-store: A transparent and highperformance local storage architecture for bare-metal clouds enabling large-scale deployment," in *2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*. IEEE, 2023, pp. 1031–1044.
- [53] O. C¸ ic¸ek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, ¨ "3d u-net: learning dense volumetric segmentation from sparse annotation," in *International conference on medical image computing and computer-assisted intervention*. Springer, 2016, pp. 424–432.
- [54] A. Dhamija, B. Madhavan, H. Li, J. Meng, S. Khare, M. Rao, L. Brakmo, N. Spring, P. Kannan, S. Sundaresan *et al.*, "A large-scale deployment of {DCTCP}," in *21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24)*, 2024, pp. 239–252.
- [55] W. Digital, "Rapidflex nvme™-of controllers c2000," https://www.westerndigital.com/products/data-centerplatforms/rapidflex-c2000-nvme-controller?sku=1K00031.
- [56] ——, "Nvme-of™ network storage protocol: Nvme™/tcp vs. rdma with rocev2," https://documents.westerndigital.com/content/dam/doclibrary/en us/assets/public/western-digital/collateral/whitepaper/white-paper-open-flex-data24-roce-vs-tcp.pdf, 2025.
- [57] Fungible, "Fs1600," https://www.storagereview.com/review/fungiblefs1600-pushes-hyperscale-storage-to-the-data-center.
- [58] P. X. Gao, A. Narayan, S. Karandikar, J. Carreira, S. Han, R. Agarwal, S. Ratnasamy, and S. Shenker, "Network requirements for resource disaggregation," in *12th USENIX symposium on operating systems design and implementation (OSDI 16)*, 2016, pp. 249–264.
- [59] Y. Gao, Q. Li, L. Tang, Y. Xi, P. Zhang, W. Peng, B. Li, Y. Wu, S. Liu, L. Yan *et al.*, "When cloud storage meets {RDMA}," in *18th USENIX Symposium on Networked Systems Design and Implementation (NSDI 21)*, 2021, pp. 519–533.
- [60] C. Guo, H. Wu, Z. Deng, G. Soni, J. Ye, J. Padhye, and M. Lipshteyn, "Rdma over commodity ethernet at scale," in *Proceedings of the 2016 ACM SIGCOMM Conference*, 2016, pp. 202–215.
- [61] M. Gupta, "Nvme/tcp in the enterprise," https://snia.org/sites/default/ files/SDC/2021/pdfs/SNIA-SDC21-Gupta-Rajagopal-NVMe-TCP-inthe-enterprise.pdf, 2021.
- [62] Z. Guz, H. H. Li, A. Shayesteh, and V. Balakrishnan, "Performance characterization of nvme-over-fabrics storage disaggregation," *ACM Trans. Storage*, vol. 14, no. 4, 2018.
- [63] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016, pp. 770–778.
- [64] J. Hwang, Q. Cai, A. Tang, and R. Agarwal, "{TCP}{}{RDMA}:{CPU-efficient} remote storage access with i10," in *17th USENIX Symposium on Networked Systems Design and Implementation (NSDI 20)*, 2020, pp. 127–140.
- [65] Intel, "Spdk nvme-of tcp (target initiator) performance report release 24.05," https://review.spdk.io/download/performance-reports/ SPDK tcp mlx perf report 2405.pdf, 2024.
- [66] S. Jiang and M. Liu, "Building an elastic block storage over {EBOFs} using shadow views," in *22nd USENIX Symposium on Networked Systems Design and Implementation (NSDI 25)*, 2025, pp. 1137–1153.
- [67] Y. Kang and M. Liu, "Understanding and profiling {NVMe-over-TCP} using ntprof," in *22nd USENIX Symposium on Networked Systems Design and Implementation (NSDI 25)*, 2025, pp. 1117–1136.
- [68] A. Klimovic, H. Litz, and C. Kozyrakis, "Reflex: Remote flash local flash," *ACM SIGARCH Computer Architecture News*, vol. 45, no. 1, pp. 345–359, 2017.
- [69] D. Kwon, J. Boo, D. Kim, and J. Kim, "{FVM}:{FPGA-assisted} virtual device emulation for fast, scalable, and flexible storage virtualization," in *14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20)*, 2020, pp. 955–971.

- [70] X. LABS, "E-series e1," https://xsightlabs.com/wpcontent/uploads/2024/10/E1-PB-Public.pdf.
- [71] H. Li, M. Hao, S. Novakovic, V. Gogte, S. Govindan, D. R. Ports, I. Zhang, R. Bianchini, H. S. Gunawi, and A. Badam, "Leapio: Efficient and portable virtual nvme storage on arm socs," in *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2020, pp. 591–605.
- [72] Q. Li, Y. Gao, X. Wang, H. Qiu, Y. Le, D. Liu, Q. Xiang, F. Feng, P. Zhang, B. Li *et al.*, "Flor: An open high performance {RDMA} framework over heterogeneous {RNICs}," in *17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23)*, 2023, pp. 931–948.
- [73] Lightbits, "Performance improvements for nvme/tcp," https://netdevconf.info/0x14/pub/slides/52/NVMe TCP\%20netdev\ %200x14.pdf, 2020.
- [74] ——, "Nvme/tcp will open the floodgates to nvme-of deployment over the next several years," https://www.lightbitslabs.com/blog/nvme-tcpwill-open-the-floodgates-to-nvme-of-deployment/, 2021.
- [75] ——, "Nvme storage: A beginner's guide to lightning-fast data access," https://www.lightbitslabs.com/blog/nvme-storage-a-beginnersguide-to-lightning-fast-data-access/, 2025.
- [76] ——, "The rise of disaggregated storage," https://www.lightbitslabs. com/blog/the-rise-of-disaggregated-storage/, 2025.
- [77] M. Liu, H. Liu, C. Ye, X. Liao, H. Jin, Y. Zhang, R. Zheng, and L. Hu, "Towards low-latency i/o services for mixed workloads using ultra-low latency ssds," in *Proceedings of the 36th ACM International Conference on Supercomputing (ICS 22)*, 2022, pp. 13:1–13:12.
- [78] Marvell, "88sn2400 nvme-of ssd converter controller," https://www.marvell.com/products/system-solutions/nvmecontrollers.html.
- [79] ——, "Marvell fastlinq 41000," https://www.marvell.com/products/ ethernet-adapters-and-controllers/41000-ethernet-adapters.html.
- [80] A. Mathuriya, D. Bard, P. Mendygral, L. Meadows, J. Arnemann, L. Shao, S. He, T. Karn ¨ a, D. Moise, S. J. Pennycook ¨ *et al.*, "Cosmoflow: Using deep learning to learn the universe at scale," in *SC18: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2018, pp. 819–829.
- [81] R. Miao, L. Zhu, S. Ma, K. Qian, S. Zhuang, B. Li, S. Cheng, J. Gao, Y. Zhuang, P. Zhang *et al.*, "From luna to solar: the evolutions of the compute-to-storage networks in alibaba cloud," in *Proceedings of the ACM SIGCOMM 2022 Conference*, 2022, pp. 753–766.
- [82] MLCommons, "Mlperf storage," https://mlcommons.org/benchmarks/storage/.
- [83] Y. Moon, S. Lee, M. A. Jamshed, and K. Park, "{AccelTCP}: Accelerating network applications with stateful {TCP} offloading," in *17th USENIX Symposium on Networked Systems Design and Implementation (NSDI 20)*, 2020, pp. 77–92.
- [84] Napatech, "F2070x infrastructure processing unit (ipu)," https://www. napatech.com/products/f2070x-ipu/.
- [85] NetApp, "Vmware cloud foundation (vcf) on netapp," https://docs.netapp.com/us-en/netapp-solutions/pdfs/VMware Cloud Foundation VCF on NetApp.pdf, 2025.
- [86] U. of New Hampshire Interoperability Labs, "NvmeTM integrator's list," https://www.iol.unh.edu/registry/nvme.
- [87] X. Pang and J. Wang, "Understanding the performance implications of the design principles in storage-disaggregated databases," *Proceedings of the ACM on Management of Data*, vol. 2, no. 3, pp. 1–26, 2024.
- [88] PCI-SIG, "Pci-sig announces pcie 8.0 specification targeted for release by 2028," https://pcisig.com/pci-sig-announces-pcie-80-specificationtargeted-release-2028, 2025.
- [89] Pensando, "Dsc3-400 distributed services card," https: //www.amd.com/content/dam/amd/en/documents/pensando-technicaldocs/product-briefs/pensando-dsc3-product-brief.pdf.
- [90] B. Pismenny, H. Eran, A. Yehezkel, L. Liss, A. Morrison, and D. Tsafrir, "Autonomous nic offloads," in *Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems*, 2021, pp. 18–35.
- [91] S. Qiu, L. Wang, and Y. Zhang, "Exo: Accelerating storage paravirtualization with ebpf," in *SC24: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2024, pp. 1–15.
- [92] C. Ruan, Y. Zhang, C. Bi, X. Ma, H. Chen, F. Li, X. Yang, C. Li, A. Aboulnaga, and Y. Xu, "Persistent memory disaggregation for cloud-native relational databases," in *Proceedings of the 28th ACM*

- *International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, 2023, pp. 498–512.
- [93] A. Shehabi, S. J. Smith, A. Hubbard, A. Newkirk, N. Lei, M. A. B. Siddik, B. Holecek, J. Koomey, E. Masanet, and D. Sartor, "2024 united states data center energy usage report," Lawrence Berkeley National Laboratory, Berkeley, California, Tech. Rep. LBNL-2001637, 2024.
- [94] J. Shu, K. Qian, E. Zhai, X. Liu, and X. Jin, "Burstable cloud block storage with data processing units," in *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, 2024, pp. 783–799.
- [95] J. Shu, R. Zhu, Y. Ma, G. Huang, H. Mei, X. Liu, and X. Jin, "Disaggregated raid storage in modern datacenters," in *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, 2023, pp. 147–163.
- [96] A. SIGARCH, "From flops to iops: The new bottlenecks of scientific computing," https://www.sigarch.org/from-flops-to-iops-the-newbottlenecks-of-scientific-computing/, 2020.
- [97] Simplyblock, "Nvme over fabrics spdk," https://www.simplyblock.io/ product-features/nvme-over-fabrics-spdk/, 2020.
- [98] A. Skiadopoulos, Z. Xie, M. Zhao, Q. Cai, S. Agarwal, J. Adelmann, D. Ahern, C. Contavalli, M. Goldflam, V. Mayatskikh *et al.*, "Highthroughput and flexible host networking for accelerated computing," in *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, 2024, pp. 405–423.
- [99] SPDK, "Nvme multipath," https://spdk.io/doc/nvme multipath.html.
- [100] ——, "vhost target," https://spdk.io/doc/vhost.html.
- [101] X. Sun, M. Zhang, Y. Shan, K. Chen, J. Jiang, and Y. Wu, "Scalio: Scaling up {DPU-based}{JBOF} key-value store with {NVMe-oF} target offload," in *19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)*, 2025, pp. 449–464.
- [102] D. Technologies, "Nvme/tcp and smartfabric storage software," https://www.delltechnologies.com/asset/en-us/products/networking/ briefs-summaries/nvme-ip-san-solution-brief.pdf, 2023.
- [103] N. H. Technologies, "H3c lossless network best practices-6w101," https://www.h3c.com/en/Support/Resource Center/EN/Home/Public/ 00-Public/Technical Documents/Configure Deploy/Best Practices/ H3C Lossless Network BP/, 2023.
- [104] R. Thompson and L. Abracon, "Clearclock for the future of pcie," *Accessed: Sep*, 2022.
- [105] Uptime Institute, "Uptime institute global data center survey 2024," Uptime Institute, Tech. Rep., 2024, accessed: 2025-08- 20. [Online]. Available: https://uptimeinstitute.com/resources/researchand-reports/uptime-institute-global-data-center-survey-results-2024
- [106] A. Verbitski, A. Gupta, D. Saha, M. Brahmadesam, K. Gupta, R. Mittal, S. Krishnamurthy, S. Maurice, T. Kharatishvili, and X. Bao, "Amazon aurora: Design considerations for high throughput cloud-native relational databases," in *Proceedings of the 2017 ACM International Conference on Management of Data*, 2017, pp. 1041–1052.
- [107] G. Wang, L. Zhang, and W. Xu, "What can we learn from four years of data center hardware failures?" in *2017 47th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)*, 2017, pp. 25–36.
- [108] J. Xu, Y. Chen, Y. Wang, W. Shi, G. Fang, Y. Chen, H. Liao, Y. Wang, H. Lin, Z. Jin *et al.*, "Lightpool: A nvme-of-based highperformance and lightweight storage pool architecture for cloud-native distributed database," in *2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*. IEEE, 2024, pp. 983– 995.
- [109] J. Xu, Y. Qiu, Y. Chen, Y. Wang, W. Lin, Y. Lin, S. Zhao, Y. Liu, Y. Wang, and W. Chen, "Performance characterization of smartnic nvme-over-fabrics target offloading," in *Proceedings of the 17th ACM International Systems and Storage Conference*, 2024, pp. 14–24.
- [110] J. Zhang, H. Huang, L. Zhu, S. Ma, D. Rong, Y. Hou, M. Sun, C. Gu, P. Cheng, C. Shi *et al.*, "Smartds: Middle-tier-centric smartnic enabling application-aware message split for disaggregated block storage," in *Proceedings of the 50th Annual International Symposium on Computer Architecture*, 2023, pp. 1–13.
- [111] D. Zhuo, M. Ghobadi, R. Mahajan, K.-T. Forster, A. Krishnamurthy, ¨ and T. Anderson, "Understanding and mitigating packet corruption in data center networks," in *Proceedings of the Conference of the ACM Special Interest Group on Data Communication*, 2017, pp. 362–375.# IX. CONCLUSION

In this work, we present NTI, a fast and scalable FPGAbased DPU solution for NVMe/TCP initiators. NTI offloads the storage disaggregation stack onto a DPU card, achieving efficiency, performance, and deployability. By executing I/O path in hardware, it eliminates compute and memory bottlenecks. Moreover, it ensures deployment in real-world datacenters. Our evaluation demonstrates that NTI outperforms baseline solutions across diverse workloads.

## ACKNOWLEDGMENT

This work was partly supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2024-00395134, DPU-Centric Datacenter Architecture for Next-Generation AI Devices; and No.RS-2025-02264029, Implementation and Validation of an AI Semiconductor-Based Data Center Composable Cluster Infrastructure).

## REFERENCES

- [1] Kernel nvme/tcp host. [Online]. Available: https: //docs.redhat.com/en/documentation/red hat enterprise linux/9/ html/managing storage devices/configuring-nvme-over-fabrics-usingnvme-tcp managing-storage-devices
- [2] pgbench. [Online]. Available: https://www.postgresql.org/docs/current/ pgbench.html
- [3] "Tcp congestion control," https://www.rfc-editor.org/rfc/rfc5681.html, 2009.
- [4] (2025, August) Amd alveo™ adaptable accelerator cards. [Online]. Available: https://www.amd.com/en/products/accelerators/alveo.html
- [5] (2025, December) Amd cips sysmon. [Online]. Available: https: //docs.amd.com/r/en-US/pg352-cips/System-Monitor
- [6] (2025, December) Amd control interfaces and processing system. [Online]. Available: https://www.amd.com/en/products/adaptive-socsand-fpgas/intellectual-property/cips.html
- [7] (2025, August) Amd pensando™ dsc3-400 distributed services card. [Online]. Available: https://www.amd.com/content/dam/amd/ en/documents/pensando-technical-docs/product-briefs/pensando-dsc3 product-brief.pdf
- [8] (2025, August) Amd versal™ adaptive socs. [Online]. Available: https://www.amd.com/en/products/adaptive-socs-and-fpgas/ versal.html#overview
- [9] (2025, August) Bluefield snap. [Online]. Available: https://docs.nvidia. com/networking/display/bluefielddpuosv470/bluefield+snap
- [10] (2025, August) Clouddc superserver sys-122c-tn. [Online]. Available: https://www.supermicro.com/en/products/system/clouddc/1u/sys-122c-tn
- [11] (2025, August) Dell emc poweredge r650 spec sheet. [Online]. Available: https://www.delltechnologies.com/asset/nl-nl/products/servers/ technical-support/dell-emc-poweredge-r650-spec-sheet.pdf
- [12] (2025, August) Dell powerdge r750. [Online]. Available: https: //www.dell.com/ko-kr/shop/cty/pdp/spd/poweredge-r750/per75010a
- [13] (2025, August) fio flexible io tester. [Online]. Available: https: //git.kernel.dk/cgit/fio
- [14] (2025, August) Fpga-accelerated nvme storage solutions using the bittware 250 series accelerators. [Online]. Available: https: //www.bittware.com/ko/resources/nvme-storage/
- [15] (2025, August) Fpga-based smartnics. [Online]. Available: https: //www.napatech.com/products
- [16] (2025, August) Hpe proliant dl385 gen10 plus v2 7313 3.0ghz 16-core 1p 32gb-r mr416i-a 8sff 800w ps server. [Online]. Available: https://buy.hpe.com/kr/ko/compute/rack-servers/proliantdl300-servers/proliant-dl385-server/hpe-proliant-dl385-gen10-plusv2-server/p/1013291283
- [17] (2025, August) Hyper superserver sys-221h-tnr. [Online]. Available: https://www.supermicro.com/en/products/system/hyper/2u/ sys-221h-tnr
- [18] (2025, August) Intel® xeon® gold 6348 processor. [Online]. Available: https://www.intel.com/content/www/us/en/products/sku/212456/intelxeon-gold-6348-processor-42m-cache-2-60-ghz/specifications.html
- [19] (2025, August) Intel® xeon® gold processor. [Online]. Available: https://www.intel.com/content/www/us/en/products/details/ processors/xeon/scalable/gold.html
- [20] (2025, August) Introduction to xlio. [Online]. Available: https: //docs.nvidia.com/networking/display/xliov3312/introduction+to+xlio
- [21] (2025, August) Kioxia. [Online]. Available: www.kioxia.com

- [22] (2025, August) Kvm. [Online]. Available: https://linux-kvm.org/page/ Main Page
- [23] (2025, August) Micron. [Online]. Available: https://www.micron.com/
- [24] (2025, August) Netapp. [Online]. Available: https://www.netapp.com/
- [25] (2025, August) Nvidia bluefield-3 networking platform user guide. [Online]. Available: https://docs.nvidia.com/networking/display/bf3dpu
- [26] (2025, August) Nvm express® base specification. [Online]. Available: https://nvmexpress.org/specification/nvm-express-base-specification
- [27] (2025, August) Nvm express® base specification 5.14.1.12 asymmetric namespace access. [Online]. Available: https: //nvmexpress.org/specification/nvm-express-base-specification
- [28] (2025, August) Nvme over fabrics (of) specification (historical reference only). [Online]. Available: https://nvmexpress.org/ specification/nvme-of-specification
- [29] (2025, August) Nvme over tcp transport specification. [Online]. Available: https://nvmexpress.org/specification/tcp-transport-specification
- [30] (2025, August) Ocp nic 3.0: Thermal considerations. [Online]. Available: https://146a55aca6f00848c565 a7635525d40ac1c70300198708936b4e.ssl.cf1.rackcdn.com/images/ 5c41ae0635633a115cc8ec42fa4618801362ae93.pdf
- [31] (2025, August) Ocp oai system liquid cooling guidelines. [Online]. Available: https://www.opencompute.org/documents/oai-systemliquid-cooling-guidelines-in-ocp-template-mar-3-2023-update-pdf
- [32] (2025, August) Purestorage. [Online]. Available: https://www. purestorage.com/
- [33] (2025, August) Qemu. [Online]. Available: https://www.qemu.org/
- [34] (2025, August) Samsung. [Online]. Available: https://www.samsung. com/sec/
- [35] (2025, August) Skhynix. [Online]. Available: https://www.skhynix. com/
- [36] (2025, August) Solidigm. [Online]. Available: https://www.solidigm. com/
- [37] (2025, August) Spdk. [Online]. Available: https://spdk.io
- [38] (2025, August) Tcp offload engine (toe). [Online]. Available: https://www.chelsio.com/nic/tcp-offload-engine/
- [39] (2025, August) Transmission control protocol. [Online]. Available: https://www.ietf.org/rfc/rfc793.txt
- [40] (2025, August) vmware esxi. [Online]. Available: https://www.vmware. com/products/cloud-infrastructure/vsphere/
- [41] (2025, August) Western digital rapidflex tm c1000 nvme-of tm adapter data sheet. [Online]. Available: https://www.westerndigital.com/ko-kr/products/data-center-platforms/ rapidflex-c1000-nvme-controller?sku=c1000-nvme-controller
- [42] (2026, April) Amd pensando™ salina dpu. [Online]. Available: https://www.amd.com/content/dam/amd/en/documents/pensandotechnical-docs/product-briefs/pensando-salina-product-brief.pdf
- [43] (2026, April) Intel® infrastructure processing unit adapter e2100-ccqda2hl. [Online]. Available: https: //www.intel.com/content/www/us/en/content-details/832490/intelinfrastructure-processing-unit-adapter-e2100-ccqda2hl.html
- [44] P. Antonopoulos, A. Budovski, C. Diaconu, A. Hernandez Saenz, J. Hu, H. Kodavalla, D. Kossmann, S. Lingam, U. F. Minhas, N. Prakash *et al.*, "Socrates: The new sql server in the cloud," in *Proceedings of the 2019 International Conference on Management of Data*, 2019, pp. 1743–1756.
- [45] L. A. Barroso, U. Holzle, and P. Ranganathan, ¨ *The Datacenter as a Computer: Designing Warehouse-Scale Machines (3rd Edition)*, ser. Synthesis Lectures on Computer Architecture, M. Martonosi, Ed. Morgan & Claypool, 2018, vol. 46, https://pages.cs.wisc.edu/∼shivaram/ cs744-readings/dc-computer-v3.pdf.
- [46] Bittware, "250-soc," https://www.bittware.com/resources/buildingnvme-over-fabrics/.
- [47] J. Boo, Y. Chung, E. Baek, S. Na, C. Kim, and J. Kim, "F4t: A fast and flexible fpga-based full-stack tcp acceleration framework," in *Proceedings of the 50th Annual International Symposium on Computer Architecture*, 2023, pp. 1–13.
- [48] Q. Cai, S. Chaudhary, M. Vuppalapati, J. Hwang, and R. Agarwal, "Understanding host network stack overheads," in *Proceedings of the 2021 ACM SIGCOMM 2021 Conference*, 2021, pp. 65–77.
- [49] Chelsio, "Roce fails to scale," https://www.chelsio.com/wp-content/ uploads/resources/RoCE-Deployment-Challenges-for-Clouds.pdf, 2015.

- [50] Y. Chen, Z. Jin, Y. Wang, Y. Chen, J. Xu, H. Yu, J. Chen, W. Lin, K. Fang, K. Zhang *et al.*, "Nvmepass: A lightweight, highperformance and scalable nvme virtualization architecture with i/o queues passthrough," in *2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)*. IEEE, 2025, pp. 1395– 1407.
- [51] Y. Chen, Z. Jin, Y. Wang, Y. Chen, H. Yu, J. Xu, J. Chen, W. Lin, K. Fang, C. Wei *et al.*, "High-performance and scalable software-based nvme virtualization mechanism with i/o queues passthrough," *arXiv preprint arXiv:2304.05148*, 2023.
- [52] Y. Chen, J. Xu, C. Wei, Y. Wang, X. Yuan, Y. Zhang, X. Yu, Y. Chen, Z. Wang, S. He *et al.*, "Bm-store: A transparent and highperformance local storage architecture for bare-metal clouds enabling large-scale deployment," in *2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*. IEEE, 2023, pp. 1031–1044.
- [53] O. C¸ ic¸ek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, ¨ "3d u-net: learning dense volumetric segmentation from sparse annotation," in *International conference on medical image computing and computer-assisted intervention*. Springer, 2016, pp. 424–432.
- [54] A. Dhamija, B. Madhavan, H. Li, J. Meng, S. Khare, M. Rao, L. Brakmo, N. Spring, P. Kannan, S. Sundaresan *et al.*, "A large-scale deployment of {DCTCP}," in *21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24)*, 2024, pp. 239–252.
- [55] W. Digital, "Rapidflex nvme™-of controllers c2000," https://www.westerndigital.com/products/data-centerplatforms/rapidflex-c2000-nvme-controller?sku=1K00031.
- [56] ——, "Nvme-of™ network storage protocol: Nvme™/tcp vs. rdma with rocev2," https://documents.westerndigital.com/content/dam/doclibrary/en us/assets/public/western-digital/collateral/whitepaper/white-paper-open-flex-data24-roce-vs-tcp.pdf, 2025.
- [57] Fungible, "Fs1600," https://www.storagereview.com/review/fungiblefs1600-pushes-hyperscale-storage-to-the-data-center.
- [58] P. X. Gao, A. Narayan, S. Karandikar, J. Carreira, S. Han, R. Agarwal, S. Ratnasamy, and S. Shenker, "Network requirements for resource disaggregation," in *12th USENIX symposium on operating systems design and implementation (OSDI 16)*, 2016, pp. 249–264.
- [59] Y. Gao, Q. Li, L. Tang, Y. Xi, P. Zhang, W. Peng, B. Li, Y. Wu, S. Liu, L. Yan *et al.*, "When cloud storage meets {RDMA}," in *18th USENIX Symposium on Networked Systems Design and Implementation (NSDI 21)*, 2021, pp. 519–533.
- [60] C. Guo, H. Wu, Z. Deng, G. Soni, J. Ye, J. Padhye, and M. Lipshteyn, "Rdma over commodity ethernet at scale," in *Proceedings of the 2016 ACM SIGCOMM Conference*, 2016, pp. 202–215.
- [61] M. Gupta, "Nvme/tcp in the enterprise," https://snia.org/sites/default/ files/SDC/2021/pdfs/SNIA-SDC21-Gupta-Rajagopal-NVMe-TCP-inthe-enterprise.pdf, 2021.
- [62] Z. Guz, H. H. Li, A. Shayesteh, and V. Balakrishnan, "Performance characterization of nvme-over-fabrics storage disaggregation," *ACM Trans. Storage*, vol. 14, no. 4, 2018.
- [63] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016, pp. 770–778.
- [64] J. Hwang, Q. Cai, A. Tang, and R. Agarwal, "{TCP}{}{RDMA}:{CPU-efficient} remote storage access with i10," in *17th USENIX Symposium on Networked Systems Design and Implementation (NSDI 20)*, 2020, pp. 127–140.
- [65] Intel, "Spdk nvme-of tcp (target initiator) performance report release 24.05," https://review.spdk.io/download/performance-reports/ SPDK tcp mlx perf report 2405.pdf, 2024.
- [66] S. Jiang and M. Liu, "Building an elastic block storage over {EBOFs} using shadow views," in *22nd USENIX Symposium on Networked Systems Design and Implementation (NSDI 25)*, 2025, pp. 1137–1153.
- [67] Y. Kang and M. Liu, "Understanding and profiling {NVMe-over-TCP} using ntprof," in *22nd USENIX Symposium on Networked Systems Design and Implementation (NSDI 25)*, 2025, pp. 1117–1136.
- [68] A. Klimovic, H. Litz, and C. Kozyrakis, "Reflex: Remote flash local flash," *ACM SIGARCH Computer Architecture News*, vol. 45, no. 1, pp. 345–359, 2017.
- [69] D. Kwon, J. Boo, D. Kim, and J. Kim, "{FVM}:{FPGA-assisted} virtual device emulation for fast, scalable, and flexible storage virtualization," in *14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20)*, 2020, pp. 955–971.

- [70] X. LABS, "E-series e1," https://xsightlabs.com/wpcontent/uploads/2024/10/E1-PB-Public.pdf.
- [71] H. Li, M. Hao, S. Novakovic, V. Gogte, S. Govindan, D. R. Ports, I. Zhang, R. Bianchini, H. S. Gunawi, and A. Badam, "Leapio: Efficient and portable virtual nvme storage on arm socs," in *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2020, pp. 591–605.
- [72] Q. Li, Y. Gao, X. Wang, H. Qiu, Y. Le, D. Liu, Q. Xiang, F. Feng, P. Zhang, B. Li *et al.*, "Flor: An open high performance {RDMA} framework over heterogeneous {RNICs}," in *17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23)*, 2023, pp. 931–948.
- [73] Lightbits, "Performance improvements for nvme/tcp," https://netdevconf.info/0x14/pub/slides/52/NVMe TCP\%20netdev\ %200x14.pdf, 2020.
- [74] ——, "Nvme/tcp will open the floodgates to nvme-of deployment over the next several years," https://www.lightbitslabs.com/blog/nvme-tcpwill-open-the-floodgates-to-nvme-of-deployment/, 2021.
- [75] ——, "Nvme storage: A beginner's guide to lightning-fast data access," https://www.lightbitslabs.com/blog/nvme-storage-a-beginnersguide-to-lightning-fast-data-access/, 2025.
- [76] ——, "The rise of disaggregated storage," https://www.lightbitslabs. com/blog/the-rise-of-disaggregated-storage/, 2025.
- [77] M. Liu, H. Liu, C. Ye, X. Liao, H. Jin, Y. Zhang, R. Zheng, and L. Hu, "Towards low-latency i/o services for mixed workloads using ultra-low latency ssds," in *Proceedings of the 36th ACM International Conference on Supercomputing (ICS 22)*, 2022, pp. 13:1–13:12.
- [78] Marvell, "88sn2400 nvme-of ssd converter controller," https://www.marvell.com/products/system-solutions/nvmecontrollers.html.
- [79] ——, "Marvell fastlinq 41000," https://www.marvell.com/products/ ethernet-adapters-and-controllers/41000-ethernet-adapters.html.
- [80] A. Mathuriya, D. Bard, P. Mendygral, L. Meadows, J. Arnemann, L. Shao, S. He, T. Karn ¨ a, D. Moise, S. J. Pennycook ¨ *et al.*, "Cosmoflow: Using deep learning to learn the universe at scale," in *SC18: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2018, pp. 819–829.
- [81] R. Miao, L. Zhu, S. Ma, K. Qian, S. Zhuang, B. Li, S. Cheng, J. Gao, Y. Zhuang, P. Zhang *et al.*, "From luna to solar: the evolutions of the compute-to-storage networks in alibaba cloud," in *Proceedings of the ACM SIGCOMM 2022 Conference*, 2022, pp. 753–766.
- [82] MLCommons, "Mlperf storage," https://mlcommons.org/benchmarks/storage/.
- [83] Y. Moon, S. Lee, M. A. Jamshed, and K. Park, "{AccelTCP}: Accelerating network applications with stateful {TCP} offloading," in *17th USENIX Symposium on Networked Systems Design and Implementation (NSDI 20)*, 2020, pp. 77–92.
- [84] Napatech, "F2070x infrastructure processing unit (ipu)," https://www. napatech.com/products/f2070x-ipu/.
- [85] NetApp, "Vmware cloud foundation (vcf) on netapp," https://docs.netapp.com/us-en/netapp-solutions/pdfs/VMware Cloud Foundation VCF on NetApp.pdf, 2025.
- [86] U. of New Hampshire Interoperability Labs, "NvmeTM integrator's list," https://www.iol.unh.edu/registry/nvme.
- [87] X. Pang and J. Wang, "Understanding the performance implications of the design principles in storage-disaggregated databases," *Proceedings of the ACM on Management of Data*, vol. 2, no. 3, pp. 1–26, 2024.
- [88] PCI-SIG, "Pci-sig announces pcie 8.0 specification targeted for release by 2028," https://pcisig.com/pci-sig-announces-pcie-80-specificationtargeted-release-2028, 2025.
- [89] Pensando, "Dsc3-400 distributed services card," https: //www.amd.com/content/dam/amd/en/documents/pensando-technicaldocs/product-briefs/pensando-dsc3-product-brief.pdf.
- [90] B. Pismenny, H. Eran, A. Yehezkel, L. Liss, A. Morrison, and D. Tsafrir, "Autonomous nic offloads," in *Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems*, 2021, pp. 18–35.
- [91] S. Qiu, L. Wang, and Y. Zhang, "Exo: Accelerating storage paravirtualization with ebpf," in *SC24: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2024, pp. 1–15.
- [92] C. Ruan, Y. Zhang, C. Bi, X. Ma, H. Chen, F. Li, X. Yang, C. Li, A. Aboulnaga, and Y. Xu, "Persistent memory disaggregation for cloud-native relational databases," in *Proceedings of the 28th ACM*

- *International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, 2023, pp. 498–512.
- [93] A. Shehabi, S. J. Smith, A. Hubbard, A. Newkirk, N. Lei, M. A. B. Siddik, B. Holecek, J. Koomey, E. Masanet, and D. Sartor, "2024 united states data center energy usage report," Lawrence Berkeley National Laboratory, Berkeley, California, Tech. Rep. LBNL-2001637, 2024.
- [94] J. Shu, K. Qian, E. Zhai, X. Liu, and X. Jin, "Burstable cloud block storage with data processing units," in *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, 2024, pp. 783–799.
- [95] J. Shu, R. Zhu, Y. Ma, G. Huang, H. Mei, X. Liu, and X. Jin, "Disaggregated raid storage in modern datacenters," in *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, 2023, pp. 147–163.
- [96] A. SIGARCH, "From flops to iops: The new bottlenecks of scientific computing," https://www.sigarch.org/from-flops-to-iops-the-newbottlenecks-of-scientific-computing/, 2020.
- [97] Simplyblock, "Nvme over fabrics spdk," https://www.simplyblock.io/ product-features/nvme-over-fabrics-spdk/, 2020.
- [98] A. Skiadopoulos, Z. Xie, M. Zhao, Q. Cai, S. Agarwal, J. Adelmann, D. Ahern, C. Contavalli, M. Goldflam, V. Mayatskikh *et al.*, "Highthroughput and flexible host networking for accelerated computing," in *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, 2024, pp. 405–423.
- [99] SPDK, "Nvme multipath," https://spdk.io/doc/nvme multipath.html.
- [100] ——, "vhost target," https://spdk.io/doc/vhost.html.
- [101] X. Sun, M. Zhang, Y. Shan, K. Chen, J. Jiang, and Y. Wu, "Scalio: Scaling up {DPU-based}{JBOF} key-value store with {NVMe-oF} target offload," in *19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)*, 2025, pp. 449–464.
- [102] D. Technologies, "Nvme/tcp and smartfabric storage software," https://www.delltechnologies.com/asset/en-us/products/networking/ briefs-summaries/nvme-ip-san-solution-brief.pdf, 2023.
- [103] N. H. Technologies, "H3c lossless network best practices-6w101," https://www.h3c.com/en/Support/Resource Center/EN/Home/Public/ 00-Public/Technical Documents/Configure Deploy/Best Practices/ H3C Lossless Network BP/, 2023.
- [104] R. Thompson and L. Abracon, "Clearclock for the future of pcie," *Accessed: Sep*, 2022.
- [105] Uptime Institute, "Uptime institute global data center survey 2024," Uptime Institute, Tech. Rep., 2024, accessed: 2025-08- 20. [Online]. Available: https://uptimeinstitute.com/resources/researchand-reports/uptime-institute-global-data-center-survey-results-2024
- [106] A. Verbitski, A. Gupta, D. Saha, M. Brahmadesam, K. Gupta, R. Mittal, S. Krishnamurthy, S. Maurice, T. Kharatishvili, and X. Bao, "Amazon aurora: Design considerations for high throughput cloud-native relational databases," in *Proceedings of the 2017 ACM International Conference on Management of Data*, 2017, pp. 1041–1052.
- [107] G. Wang, L. Zhang, and W. Xu, "What can we learn from four years of data center hardware failures?" in *2017 47th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)*, 2017, pp. 25–36.
- [108] J. Xu, Y. Chen, Y. Wang, W. Shi, G. Fang, Y. Chen, H. Liao, Y. Wang, H. Lin, Z. Jin *et al.*, "Lightpool: A nvme-of-based highperformance and lightweight storage pool architecture for cloud-native distributed database," in *2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*. IEEE, 2024, pp. 983– 995.
- [109] J. Xu, Y. Qiu, Y. Chen, Y. Wang, W. Lin, Y. Lin, S. Zhao, Y. Liu, Y. Wang, and W. Chen, "Performance characterization of smartnic nvme-over-fabrics target offloading," in *Proceedings of the 17th ACM International Systems and Storage Conference*, 2024, pp. 14–24.
- [110] J. Zhang, H. Huang, L. Zhu, S. Ma, D. Rong, Y. Hou, M. Sun, C. Gu, P. Cheng, C. Shi *et al.*, "Smartds: Middle-tier-centric smartnic enabling application-aware message split for disaggregated block storage," in *Proceedings of the 50th Annual International Symposium on Computer Architecture*, 2023, pp. 1–13.
- [111] D. Zhuo, M. Ghobadi, R. Mahajan, K.-T. Forster, A. Krishnamurthy, ¨ and T. Anderson, "Understanding and mitigating packet corruption in data center networks," in *Proceedings of the Conference of the ACM Special Interest Group on Data Communication*, 2017, pp. 362–375.