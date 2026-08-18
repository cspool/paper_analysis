# V. EXPERIMENTAL SETUP

#### *A. Algorithm Baselines*

We use EVBEYE [17], the first and most commonly used event-based gaze tracking dataset, to evaluate the performance of our ESS and robust ViT algorithm. The data was recorded from 27 subjects and consists of two experiments corresponding to two different types of eye motion: random saccades and smooth pursuits. The stimuli were displayed on a 40-inch, 1920×1080 pixel monitor at a standard reading distance of 40 cm. Following the standard practice to eliminate the invalid labels, such as those recorded before the experiment began [131], we excluded the "stop" and "pause" labels and the first 15 labels for each saccade beginning. The training and testing sets are created without distinguishing between the left and right eyes. The angular error is calculated over the dataset and averaged per inference.

We train the ViT algorithm using a batch size of 64 with 500 epochs. We compare our performance with two SOTA event-based gaze tracking algorithms, one reported on the EVBEYE [120] dataset and another on the OpenEDS dataset using its backbone [28] and our detector head. We also conduct experiments to compare our ESS algorithm with three sparse sampling approaches [47], [67], [98].

The ESS mask is generated offline based on the EVBEYE dataset. We randomly select 22 of the 27 subjects to create the mask, which requires about two minutes on an NVIDIA A100 GPU. The remaining subjects are unseen during mask generation. Then, we apply the same generated ESS mask to all subjects during neural network model training and testing without any user-specific recalibration or normalization. That is, the inference data is completely unseen by the ESS mask, thereby demonstrating the practical deployment feasibility of our ESS mechanism. In practical applications, the ESS mask is initialized using training data acquired from the specific hardware setup. As this regeneration process is fast and userindependent, DESSCam demonstrates good generalizability and scalability in AR/VR HMDs.

#### *B. Hardware Evaluation Methodology*

The hardware architecture of DESSCam comprises a pixel array analog front-end (top layer), pixel array sampling logic (bottom layer), PAC circuit array, peripheral circuits, and an output interface, as shown in Fig. 6. The proposed DESS-Cam is implemented in a standard CMOS 40 nm process node. The pixel front-ends, PSC, and our SSPL circuits are simulated using Cadence Virtuoso. The logarithmic amplifier and source follower are supplied at 2.5 V, while the rest of the pixel circuitry operates at 1.1 V. The SRAM peripherals are compiled using the ARM memory compiler. The PAC circuit array, peripheral circuits, and output interface are implemented in RTL. These peripheral digital circuits are synthesized, placed, and routed using a standard EDA flow with Synopsys and Cadence tools using a 22 nm process

supplied at 0.8 V. The logic power consumption is evaluated using Synopsys PrimeTimePX, incorporating fully annotated switching activity. Similar to the prior work [47], we scale the results to 40 nm process nodes using the DeepScaleTool [101], [115] for digital circuits. The analog components and SRAM cells are implemented directly using the 40 nm process library. As our DESSCam is evaluated using the standard EDA flow, we assume the off-sensor gaze estimation algorithm runs on an STM32N6x7 processor [1], which is fabricated using a 16 nm process and embeds an Arm Cortex-M55 core together with a Neural-ART NPU [12]. STM32N6x7 consumes only 6 mJ per inference when executing a standard MobileNet v2 [100], suitable for an embedded eye tracking system [16].

Our ViT model supports efficient deployment on commercial chips. Specifically, our model is quantized to INT8 precision using the learned step size quantization (LSQ) method [43]. We export the quantized model to the ONNX format and use STM32Cube.AI for heterogeneous deployment. That is, computationally intensive operations (e.g., in convolutional and linear layers) are executed on the Neural-ART NPU, while non-linear operations (e.g., LayerNorm, Softmax) are offloaded to the Cortex-M55 CPU. Although the Neural-ART NPU natively supports INT8 operations [8], it lacks native acceleration capability for transformer blocks. However, as discussed in Sec. III-C, most of our algorithm's computation is in the early convolutional layers rather than the transformer, because the convolutional layers generate sparse tokens for the transformer blocks, which significantly reduces its computational workload. Therefore, we adopt the official MobileNet v2 latency and power efficiency baseline on the STM32N6x7 [100] to estimate the deployment performance of our model. In addition, we can design a dedicated NPU to employ our model to achieve higher energy efficiency, and the NPU can be directly integrated into the 3D-stacked DVS sensor, which we leave for future work.

For our experimental setup, the ESS compression rate (uncompressed size over compressed size; 1 for full frame) is configured to 50×, the event readout threshold is set to 2, and 12 patches are used for every gaze estimation. Our evaluation metrics include power consumption, tracking latency, and hardware area, where power and latency are reported based on the samples captured during fast saccades (representing peak event rates). Our evaluation methodology is as follows.

Power Consumption. The system power is evaluated for both in-sensor and off-sensor components. The in-sensor power evaluation follows the standard DVS evaluation methodologies [48], [62], [111], which partition the power into static and dynamic contributions. Specifically, the static power is simulated under idle conditions (low event rate), which is determined by the pixel and peripheral digital circuits. The dynamic power is simulated under active imaging mode, which is decided by both the event-triggering energy and event rate. The in-sensor power consumption is defined as follows [52]:

$$P_{\text{in\_sensor}} = P_{\text{pixel\_static}} + \frac{1}{M} \sum_{s=1}^{M} \frac{N_{\text{ev}}(s) \cdot E_{\text{ts}}}{T_{\text{acc}}(s)} + P_{\text{logic}}$$
(3)

where Ppixel static denotes the static power of the pixel array. M is the total number of samples, s is the sample index, Ets is the energy for pixel event triggering and sampling, and Tacc(s) is the accumulation time for sample s. The middle term represents pixel dynamic power. Plogic includes both the static power and dynamic power of digital circuits. Note that DESSCam is an event-driven architecture, and its dynamic power varies with input. Therefore, we compute the average insensor switching activity and patch volume transferred to the off-sensor algorithm to assess dynamic power. By traversing the dataset, we obtain the average event volume per event frame, Nev, which governs pixel dynamic power and digital circuit switching activity.

The power consumption of both the MIPI interface and the off-sensor NPU is analytically estimated using the energy costs from prior work and commercial chips. For the MIPI interface, we evaluate its power using an energy cost of approximately 100 pJ per transmitted byte [73], consistent with the assumption in the SOTA research [47]. This 100 pJ/byte energy cost accounts for the standard output process of the MIPI interface, which corresponds to transmitting data from the output FIFO (Fig. 6) to the off-sensor NPU. Thus, the MIPI power is estimated by multiplying the average data rate (derived by multiplying the average patch activation rate by the AER data packet size) by the energy cost (100 pJ/byte). For the offsensor NPU, the power consumption is calculated by dividing the MAC operations per second by the energy efficiency of the STM32N6x7 processor [1]. For a fair comparison, the offsensor power of all the baselines is evaluated using our robust ViT algorithm, except for TinyTracker, which performs insensor inference using the IMX500 [4].

We first evaluate the average energy of each DVS event frame and each off-sensor inference. The power consumption is then obtained by multiplying this total energy by the frame rate. Note that although DVS's peak frame rates can reach several kHz, this rate dynamically adjusts based on the speed of eye movement. We assume the resolution of DESSCam is 346×260, consistent with the commercial DAVIS346 camera [2].

Tracking latency. For tracking latency, we account for:

- 1) The detection latency of the top-layer analog front-end.
- 2) The eventification and sampling latency in the bottom layer.
- 3) The propagation latency of adders in patch grouping.
- 4) The latency of handshaking with peripheral circuits and packeting AER data.
- 5) Interface transmission latency.
- 6) The latency of the proposed gaze tracking algorithm on the NPU.

The analog front-end latency in 1) is modeled using typical pixel triggering delays under standard light intensity [52]. The latency of eventification and sampling in 2) is obtained via Cadence Virtuoso simulations. Subsequent digital circuits are implemented in RTL, and their latency in 3) and 4) is derived from timing reports generated by Synopsys tools.

Similar to the power calculation, both the MIPI interface latency in 5) and the NPU latency in 6) are also derived from analytical estimates. The MIPI interface latency is estimated by dividing the output data volume (i.e., the AER packet size multiplied by the number of activated patches per frame) by the effective MIPI CSI-2 bandwidth, which is assumed to be 2.5 Gbps [92], [112] for a fair comparison with the SOTA work [47]. The NPU computation latency depends on the patch count per event frame and is evaluated using the same method as for power.

**Chip Area**. Each of our pixels consists of the pixel sensing circuits (PSC) in the top layer and the sparse sampling logic (SSPL) in the bottom layer. For the top layer, we claim no novelty for the PSC design and adopt the same circuits in [48], [117], so the area of the PSC is estimated based on the results reported in these prior works. For the bottom layer, we obtain the area of SSPL from its layout using Cadence Virtuoso at the 40 nm process node. The area of one PAC includes the area of 16×16 SSPL and the digital logic. The digital logic area is acquired from the synthesis area report using Design Compiler at the 22 nm node and scaled with DeepScaleTool. To obtain the equivalent pixel area, we directly divide the total PAC area by 256. Together with the areas of the peripheral circuits including SRAM readout circuits, control logic and an output interface (shown in Fig. 6), which are obtained by the Memory Compiler and Design Compiler, the whole area of the sensor is determined.

In summary, our evaluation metrics include power consumption, tracking latency, and hardware area. As shown in Table II, the power and latency of the MIPI interface and the off-sensor NPU, along with the top-layer pixel area, are derived from analytical estimation relying on prior work and commercial chips. The MIPI power and latency are estimated using the data rate (calculated from the per-frame data volume and the operating frame rate), combined with the bandwidth [92], [112] and power efficiency [73]. The NPU power and latency are estimated based on the total number of MAC operations, combined with the throughput and energy efficiency [1]. For the top-layer pixel (PSC) area, we adopt the pixel pitch reported in prior works [48], [117] with the same PSC design. For the remaining components, our evaluations are based on hardware implementation. The power and latency of the pixel (including both the top layer and the bottom layer) are evaluated via SPICE simulations using Cadence Virtuoso. The area of the bottom-layer SSPL (featuring our proposed ESS and PAC) is measured from the layout. Besides, the power, latency, and area of the digital and peripheral circuits are obtained from post-synthesis reports using Synopsys Design Compiler and Memory Compiler.

#### C. Ablation Study

To elucidate the contributions of different components in our system, we conduct ablation studies at both algorithmic and hardware levels. At the algorithmic level, we compare our ESS mechanism with three sparse sampling methods: (1) a PAC-only method without sparse sampling, (2) a widely adopted

TABLE II
EVALUATION METHODOLOGIES FOR SYSTEM COMPONENTS

| Component                 | Power     | Latency   | Area      |
|---------------------------|-----------|-----------|-----------|
| Top Layer Pixel (PSC)     | Sim.      | Sim.      | Est.      |
| Bottom Layer Pixel (SSPL) | Sim.      | Sim.      | Layout    |
| Digital & Peripherals     | Post-Syn. | Post-Syn. | Post-Syn. |
| MIPI Interface            | Est.      | Est.      | -         |
| Off-sensor NPU            | Est.      | Est.      | -         |

Est.: Analytical estimate relying on prior work and commercial chips;

Sim.: SPICE simulation using Cadence Virtuoso:

Post-Syn.: Post-synthesis using Synopsys Design Compiler and Memory Compiler.

![](_page_8_Figure_10.jpeg)

Fig. 9. End-to-end gaze prediction vs. compression rate (uncompressed size over compressed size; 1 for full frame). All the experiments are conducted using ESS. The results demonstrate that our ViT exhibits greater robustness under a high compression rate.

random sampling method used in existing algorithms [70] and sparse sampling sensor architectures [47], and (3) an event-density based denoising method [45].

For hardware evaluation, we define the following system configurations for ablation studies:

- 3D-EPV: full DESSCam architecture with ESS, PAC, and Robust ViT, executed on an off-sensor NPU.
- 3D-PV: DESSCam architecture with PAC and ViT, excluding the ESS mechanism.
- 2D-V: standard DVS architecture outputting data for ViT processing.
- BlissCam [47]: performs in-sensor sparse eventification and sampling in the analog domain, using a random sampling mechanism and standard camera imaging.
- *TinyTracker* [23]: utilizes a commercial image sensor IMX500 supporting near-sensor computing [4], with a lightweight gaze-tracking algorithm deployed in-sensor.

# V. EXPERIMENTAL SETUP

#### *A. Algorithm Baselines*

We use EVBEYE [17], the first and most commonly used event-based gaze tracking dataset, to evaluate the performance of our ESS and robust ViT algorithm. The data was recorded from 27 subjects and consists of two experiments corresponding to two different types of eye motion: random saccades and smooth pursuits. The stimuli were displayed on a 40-inch, 1920×1080 pixel monitor at a standard reading distance of 40 cm. Following the standard practice to eliminate the invalid labels, such as those recorded before the experiment began [131], we excluded the "stop" and "pause" labels and the first 15 labels for each saccade beginning. The training and testing sets are created without distinguishing between the left and right eyes. The angular error is calculated over the dataset and averaged per inference.

We train the ViT algorithm using a batch size of 64 with 500 epochs. We compare our performance with two SOTA event-based gaze tracking algorithms, one reported on the EVBEYE [120] dataset and another on the OpenEDS dataset using its backbone [28] and our detector head. We also conduct experiments to compare our ESS algorithm with three sparse sampling approaches [47], [67], [98].

The ESS mask is generated offline based on the EVBEYE dataset. We randomly select 22 of the 27 subjects to create the mask, which requires about two minutes on an NVIDIA A100 GPU. The remaining subjects are unseen during mask generation. Then, we apply the same generated ESS mask to all subjects during neural network model training and testing without any user-specific recalibration or normalization. That is, the inference data is completely unseen by the ESS mask, thereby demonstrating the practical deployment feasibility of our ESS mechanism. In practical applications, the ESS mask is initialized using training data acquired from the specific hardware setup. As this regeneration process is fast and userindependent, DESSCam demonstrates good generalizability and scalability in AR/VR HMDs.

#### *B. Hardware Evaluation Methodology*

The hardware architecture of DESSCam comprises a pixel array analog front-end (top layer), pixel array sampling logic (bottom layer), PAC circuit array, peripheral circuits, and an output interface, as shown in Fig. 6. The proposed DESS-Cam is implemented in a standard CMOS 40 nm process node. The pixel front-ends, PSC, and our SSPL circuits are simulated using Cadence Virtuoso. The logarithmic amplifier and source follower are supplied at 2.5 V, while the rest of the pixel circuitry operates at 1.1 V. The SRAM peripherals are compiled using the ARM memory compiler. The PAC circuit array, peripheral circuits, and output interface are implemented in RTL. These peripheral digital circuits are synthesized, placed, and routed using a standard EDA flow with Synopsys and Cadence tools using a 22 nm process

supplied at 0.8 V. The logic power consumption is evaluated using Synopsys PrimeTimePX, incorporating fully annotated switching activity. Similar to the prior work [47], we scale the results to 40 nm process nodes using the DeepScaleTool [101], [115] for digital circuits. The analog components and SRAM cells are implemented directly using the 40 nm process library. As our DESSCam is evaluated using the standard EDA flow, we assume the off-sensor gaze estimation algorithm runs on an STM32N6x7 processor [1], which is fabricated using a 16 nm process and embeds an Arm Cortex-M55 core together with a Neural-ART NPU [12]. STM32N6x7 consumes only 6 mJ per inference when executing a standard MobileNet v2 [100], suitable for an embedded eye tracking system [16].

Our ViT model supports efficient deployment on commercial chips. Specifically, our model is quantized to INT8 precision using the learned step size quantization (LSQ) method [43]. We export the quantized model to the ONNX format and use STM32Cube.AI for heterogeneous deployment. That is, computationally intensive operations (e.g., in convolutional and linear layers) are executed on the Neural-ART NPU, while non-linear operations (e.g., LayerNorm, Softmax) are offloaded to the Cortex-M55 CPU. Although the Neural-ART NPU natively supports INT8 operations [8], it lacks native acceleration capability for transformer blocks. However, as discussed in Sec. III-C, most of our algorithm's computation is in the early convolutional layers rather than the transformer, because the convolutional layers generate sparse tokens for the transformer blocks, which significantly reduces its computational workload. Therefore, we adopt the official MobileNet v2 latency and power efficiency baseline on the STM32N6x7 [100] to estimate the deployment performance of our model. In addition, we can design a dedicated NPU to employ our model to achieve higher energy efficiency, and the NPU can be directly integrated into the 3D-stacked DVS sensor, which we leave for future work.

For our experimental setup, the ESS compression rate (uncompressed size over compressed size; 1 for full frame) is configured to 50×, the event readout threshold is set to 2, and 12 patches are used for every gaze estimation. Our evaluation metrics include power consumption, tracking latency, and hardware area, where power and latency are reported based on the samples captured during fast saccades (representing peak event rates). Our evaluation methodology is as follows.

Power Consumption. The system power is evaluated for both in-sensor and off-sensor components. The in-sensor power evaluation follows the standard DVS evaluation methodologies [48], [62], [111], which partition the power into static and dynamic contributions. Specifically, the static power is simulated under idle conditions (low event rate), which is determined by the pixel and peripheral digital circuits. The dynamic power is simulated under active imaging mode, which is decided by both the event-triggering energy and event rate. The in-sensor power consumption is defined as follows [52]:

$$P_{\text{in\_sensor}} = P_{\text{pixel\_static}} + \frac{1}{M} \sum_{s=1}^{M} \frac{N_{\text{ev}}(s) \cdot E_{\text{ts}}}{T_{\text{acc}}(s)} + P_{\text{logic}}$$
(3)

where Ppixel static denotes the static power of the pixel array. M is the total number of samples, s is the sample index, Ets is the energy for pixel event triggering and sampling, and Tacc(s) is the accumulation time for sample s. The middle term represents pixel dynamic power. Plogic includes both the static power and dynamic power of digital circuits. Note that DESSCam is an event-driven architecture, and its dynamic power varies with input. Therefore, we compute the average insensor switching activity and patch volume transferred to the off-sensor algorithm to assess dynamic power. By traversing the dataset, we obtain the average event volume per event frame, Nev, which governs pixel dynamic power and digital circuit switching activity.

The power consumption of both the MIPI interface and the off-sensor NPU is analytically estimated using the energy costs from prior work and commercial chips. For the MIPI interface, we evaluate its power using an energy cost of approximately 100 pJ per transmitted byte [73], consistent with the assumption in the SOTA research [47]. This 100 pJ/byte energy cost accounts for the standard output process of the MIPI interface, which corresponds to transmitting data from the output FIFO (Fig. 6) to the off-sensor NPU. Thus, the MIPI power is estimated by multiplying the average data rate (derived by multiplying the average patch activation rate by the AER data packet size) by the energy cost (100 pJ/byte). For the offsensor NPU, the power consumption is calculated by dividing the MAC operations per second by the energy efficiency of the STM32N6x7 processor [1]. For a fair comparison, the offsensor power of all the baselines is evaluated using our robust ViT algorithm, except for TinyTracker, which performs insensor inference using the IMX500 [4].

We first evaluate the average energy of each DVS event frame and each off-sensor inference. The power consumption is then obtained by multiplying this total energy by the frame rate. Note that although DVS's peak frame rates can reach several kHz, this rate dynamically adjusts based on the speed of eye movement. We assume the resolution of DESSCam is 346×260, consistent with the commercial DAVIS346 camera [2].

Tracking latency. For tracking latency, we account for:

- 1) The detection latency of the top-layer analog front-end.
- 2) The eventification and sampling latency in the bottom layer.
- 3) The propagation latency of adders in patch grouping.
- 4) The latency of handshaking with peripheral circuits and packeting AER data.
- 5) Interface transmission latency.
- 6) The latency of the proposed gaze tracking algorithm on the NPU.

The analog front-end latency in 1) is modeled using typical pixel triggering delays under standard light intensity [52]. The latency of eventification and sampling in 2) is obtained via Cadence Virtuoso simulations. Subsequent digital circuits are implemented in RTL, and their latency in 3) and 4) is derived from timing reports generated by Synopsys tools.

Similar to the power calculation, both the MIPI interface latency in 5) and the NPU latency in 6) are also derived from analytical estimates. The MIPI interface latency is estimated by dividing the output data volume (i.e., the AER packet size multiplied by the number of activated patches per frame) by the effective MIPI CSI-2 bandwidth, which is assumed to be 2.5 Gbps [92], [112] for a fair comparison with the SOTA work [47]. The NPU computation latency depends on the patch count per event frame and is evaluated using the same method as for power.

**Chip Area**. Each of our pixels consists of the pixel sensing circuits (PSC) in the top layer and the sparse sampling logic (SSPL) in the bottom layer. For the top layer, we claim no novelty for the PSC design and adopt the same circuits in [48], [117], so the area of the PSC is estimated based on the results reported in these prior works. For the bottom layer, we obtain the area of SSPL from its layout using Cadence Virtuoso at the 40 nm process node. The area of one PAC includes the area of 16×16 SSPL and the digital logic. The digital logic area is acquired from the synthesis area report using Design Compiler at the 22 nm node and scaled with DeepScaleTool. To obtain the equivalent pixel area, we directly divide the total PAC area by 256. Together with the areas of the peripheral circuits including SRAM readout circuits, control logic and an output interface (shown in Fig. 6), which are obtained by the Memory Compiler and Design Compiler, the whole area of the sensor is determined.

In summary, our evaluation metrics include power consumption, tracking latency, and hardware area. As shown in Table II, the power and latency of the MIPI interface and the off-sensor NPU, along with the top-layer pixel area, are derived from analytical estimation relying on prior work and commercial chips. The MIPI power and latency are estimated using the data rate (calculated from the per-frame data volume and the operating frame rate), combined with the bandwidth [92], [112] and power efficiency [73]. The NPU power and latency are estimated based on the total number of MAC operations, combined with the throughput and energy efficiency [1]. For the top-layer pixel (PSC) area, we adopt the pixel pitch reported in prior works [48], [117] with the same PSC design. For the remaining components, our evaluations are based on hardware implementation. The power and latency of the pixel (including both the top layer and the bottom layer) are evaluated via SPICE simulations using Cadence Virtuoso. The area of the bottom-layer SSPL (featuring our proposed ESS and PAC) is measured from the layout. Besides, the power, latency, and area of the digital and peripheral circuits are obtained from post-synthesis reports using Synopsys Design Compiler and Memory Compiler.

#### C. Ablation Study

To elucidate the contributions of different components in our system, we conduct ablation studies at both algorithmic and hardware levels. At the algorithmic level, we compare our ESS mechanism with three sparse sampling methods: (1) a PAC-only method without sparse sampling, (2) a widely adopted

TABLE II
EVALUATION METHODOLOGIES FOR SYSTEM COMPONENTS

| Component                 | Power     | Latency   | Area      |
|---------------------------|-----------|-----------|-----------|
| Top Layer Pixel (PSC)     | Sim.      | Sim.      | Est.      |
| Bottom Layer Pixel (SSPL) | Sim.      | Sim.      | Layout    |
| Digital & Peripherals     | Post-Syn. | Post-Syn. | Post-Syn. |
| MIPI Interface            | Est.      | Est.      | -         |
| Off-sensor NPU            | Est.      | Est.      | -         |

Est.: Analytical estimate relying on prior work and commercial chips;

Sim.: SPICE simulation using Cadence Virtuoso:

Post-Syn.: Post-synthesis using Synopsys Design Compiler and Memory Compiler.

![](_page_8_Figure_10.jpeg)

Fig. 9. End-to-end gaze prediction vs. compression rate (uncompressed size over compressed size; 1 for full frame). All the experiments are conducted using ESS. The results demonstrate that our ViT exhibits greater robustness under a high compression rate.

random sampling method used in existing algorithms [70] and sparse sampling sensor architectures [47], and (3) an event-density based denoising method [45].

For hardware evaluation, we define the following system configurations for ablation studies:

- 3D-EPV: full DESSCam architecture with ESS, PAC, and Robust ViT, executed on an off-sensor NPU.
- 3D-PV: DESSCam architecture with PAC and ViT, excluding the ESS mechanism.
- 2D-V: standard DVS architecture outputting data for ViT processing.
- BlissCam [47]: performs in-sensor sparse eventification and sampling in the analog domain, using a random sampling mechanism and standard camera imaging.
- *TinyTracker* [23]: utilizes a commercial image sensor IMX500 supporting near-sensor computing [4], with a lightweight gaze-tracking algorithm deployed in-sensor.

