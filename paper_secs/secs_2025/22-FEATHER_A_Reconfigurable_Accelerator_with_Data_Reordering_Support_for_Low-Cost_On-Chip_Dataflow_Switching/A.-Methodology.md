# A. Methodology

We implement *FEATHER* in Verilog and Xilinx HLS. Verilog-based implementation delivers precise microarchitecture design while HLS-based implementation enables the native usage of Xilinx IPs for buffer, control, and peripherals for better end-to-end performance on Xilinx FPGAs. We evaluate its resources on TSMC 28 nm high performance technology node using the Verilog-based implementation. We compare its end-to-end wall-clock latency against SoTAs with open-sourced end-to-end implementations on real FPGA devices. We also model *FEATHER* in Layoutloop (§V), including energy overheads, to compare it against SoTA accelerators that do not have open-sourced end-to-end deployable codes. Tab. IV summarizes our evaluation setup.

## 1) Baselines and Workloads:

Baselines for real-device evaluations. We compare *FEATHER* against Xilinx DPU [2], Gemmini [21], and Edge TPU [46], as they can be deployed in an end-to-end fashion. *FEATHER* and Xilinx DPU [2] are deployed on the same Xilinx ZCU 104 FPGA board. While Gemmini is deployed on AWS-F1 FPGA server [21] using FireSim to emulate its per-layer processing latency. Edge TPU [46] runs on a USB accelerator [1] attached to a Raspberry Pi 4B. As for all four designs, we normalize throughput by the number of PEs (i.e., MAC units) and clock frequency <sup>3</sup> for a fair comparison.

**Baselines for Layoutloop.** *FEATHER* is further compared against NVDLA [39], Eyeriss [14] and SIGMA [42] in Layoutloop. Detailed modifications/specs are listed in Tab. IV

**Workload.** BERT (representative of cloud workloads); ResNet-50 and MobiletNet-V3 (Mob-V3) as edge workloads.

- 2) FEATHER Dataflow/Layout Setup:
- **Search Space.** Dataflow design space is constructed by arbitrary nested loops as shown in Fig. 1. We use layout patterns used by prior accelerators [43] as layout space <sup>4</sup>.
- Searching Algorithm. We exhaustively search layout space for global optimal. To find optimized dataflows, we use Timeloop's internal hybrid search algorithm (exhaustive +

<sup>3</sup>Both GEMMINI and FEATHER could run at 1 GHz under TSMC 28 nm ASIC flow. However, the parallel simulation synthesis toolchain of firesim limits GEMMINI's clock frequency to 50 MHz on AWS's f1.2xlarge FPGA.

 $^4$ Conv: HWC\_C32, HWC\_W32, HWC\_H32, HWC\_C4W8, HWC\_C4H8, HWC\_W4H8, HWC\_C4W4H2; GEMM: we note input/weights/output as  $M \times K/N \times K/M \times N$  with inputs layout as MK\_K32, MK\_M32, MK\_M4K8.

![](_page_10_Figure_15.jpeg)

Fig. 12: FEATHER vs. SoTAs on real devices. We run each layer for 100 times to obtain average layer latency, and then normalize throughput by number of PE and clock frequency.

search-space pruning). Recent works [29] show that its results are comparable to sophisticated search methods [22], [23], [27] but is slower in wall clock time. We ran the search with multiple threads constrained on search size and victory conditions.

- **Performance Metric.** We use Energy-Delay-Product (EDP) as the performance metric for a dataflow/layout pair.
- Overall Search Flow. Dataflow/layout cosearch is conducted for each layer independently. The optimal dataflow-layout pair with the best EDP is chosen for each layer of ResNet-50 and Mob-V3 in the Layoutloop evaluation. For end-to-end FPGA deployment of ResNet-50, we simplify engineering efforts by selecting the two layouts with the best latency and energy efficiency on DepthWise Conv. and typical Conv., and enable FEATHER to switch between them per layer.

## B. End-to-end Real-device Latency Evaluation

- 1) FEATHER vs. Gemmini: FEATHER achieves a 3.91× geomean normalized throughput improvement than Gemmini as shown in Fig. 12, as Gemmini adopts a fixed dataflow (weights stationary with degree of parallelism being 16 in both C and M), leading to under-utilization when C of workload is not divisible by 16. The flexibility of FEATHER in the parallelism of M,C,H,W delivers its performance improvement.
- 2) FEATHER vs. Xilinx DPU: The  $2.65\times$  more throughput of FEATHER over Xilinx DPU stems from the low steady-state utilization of Xilinx DPU under convolution  $3\times3$  (75% utilization),  $7\times7$  (21.8~87.5% utilization), and FEATHER pushes both utilization to 100% and 90.4% in the steady state. This is because Xilinx's DPU with 1152 PEs only supports a single dataflow with parallelism (12,12,8) in (M,C,H/W). In deep layers with a large number of input channels (C) and kernels (M), both FEATHER and Xilinx DPU achieve a steady utilization of 100%. However, Xilinx DPU outperforms FEATHER for these layers as our controller is not as optimized as DPU's (an engineering optimization part of our future work).
- 3) FEATHER vs. Edge TPU:  $4.91\times$  speedup comes from flexibility of FEATHER in dataflow and layout.

