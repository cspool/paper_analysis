# VI. EVALUATION

#### A. Accuracy vs. Compression Rate

Our eye-tracking algorithm achieves a lower AE compared to the baselines across a wide range of compression rates (defined by uncompressed size/compressed size). Fig. 9 compares the angular error versus compression rate. Baseline algorithms employ the same ESS method to achieve varying compression rates. Results show that across all compression ratios, our algorithm consistently maintains the gaze estimation accuracy within  $2^{\circ}$ , showing a stronger robustness to tolerate sparse input. In particular, our robust ViT algorithm achieves an angular error of only  $0.5^{\circ}$  even under a  $50\times$  compression rate. This compression rate is used in subsequent experiments.

![](_page_9_Figure_0.jpeg)

Fig. 10. Comparison between our ESS and other sparse sampling alternatives. All experiments are conducted using our ViT algorithm. Compared to other sparse sampling methods, our ESS can retain acceptable accuracy even at high compression rates.

Furthermore, as shown in Fig. 10, our ESS sparse sampling algorithm outperforms other sparse sampling methods, especially under high compression rates. For a fair comparison, we use our robust ViT to evaluate different sparse sampling methods, including the SOTA image sensor for eye tracking with a random sparse-sampling mechanism [47], and an eventdensity based denoising method [45]. Results show that our ESS achieves lower AE compared with these baselines across all compression ratios. Besides, at 50x compression rate, the PAC-only variant (without ESS) reaches an AE of 4.7°, which is unacceptable for AR/VR applications [125]. In contrast, our ESS preserves critical global structure, achieving an AE of only 0.5° even under a 50× compression rate. This improvement stems from ESS's global correlation-aware pixel preselection, which preserves essential information under high compression rates.

#### B. Power Reduction

As shown in Fig. 11, the end-to-end power consumption of our system achieves a 2.7× reduction compared to the SOTA research, BlissCam [47], which outputs 34,257.8 pixels per frame on average (13.38% of the pixel array) by executing ROI prediction. In contrast, our method requires only a few hundred events (2% of the pixel array) for a single inference, thereby achieving much lower system power than BlissCam. There are three main reasons. Firstly, our correlation-based ESS downsamples the pixels by 50×, drastically reducing sampling requirements. Secondly, we implement efficient insensor ESS using a predefined pixel-level mask, avoiding the overhead of refreshing the entire SRAM array to generate random numbers per inference, as required in BlissCam [47]. Thirdly, DESSCam uses simple PAC circuits for token sparsity instead of an in-sensor NPU integrated in BlissCam.

To validate the effectiveness of our ESS and PAC mechanisms in power reduction, we conduct ablation studies. Compared to the 2D-V architecture, our 3D-EPV with ESS reduces power by 5.1×. This is due to: (1) disabling comparators in non-sampling pixels, as comparators dominate per-pixel power consumption [36]; (2) performing local event sampling within the pixel, eliminating the need for comparators to drive the event bus with high load capacitance. Our added SRAM contributes only 2.1% to the pixel's total static power, while

![](_page_9_Figure_6.jpeg)

Fig. 11. Power evaluation results. Our 3D-EPV eye-tracking system reduces power by 5.1× over a standard DVS-based system [111] and 2.7× over the SOTA eye tracking image sensor [47]. This improvement mainly comes from the ESS mechanism, which significantly reduces per-pixel power consumption.

its dynamic power is incurred only when the pixel is enabled for sampling.

Our SSPL-pixel implements ESS-based sampling, which eliminates the dominant power contribution from comparators and sampling circuits in non-sampling pixels by 12.6× and 9.6×, respectively, thereby reducing per-pixel power by up to 5.26×, as illustrated in Fig. 12. As a result, with only 2% active sampling pixels contributing dynamic power, our 3D-EPV achieves a 4.93× reduction in average per-pixel power compared to the no-ESS baseline (3D-PV). In summary, these ablation studies confirm that our ESS and PAC mechanisms are the primary contributors to the 5.1× power reduction over typical DVS-based eye tracking systems and the 2.7× reduction over the SOTA systems.

DESSCam enables the eye tracking system to be deployed in ultra-lightweight AR/VR HMDs, like smart glasses, which typically operate under a limited total system power budget of only tens of mW. For instance, the Meta Ray-Ban smart glasses use micro-batteries with only 154 mAh (about 593 mWh) to keep a light weight (48.6 g) [99], [108]. As a result, the power budget of the glasses is only 49.6 mW to realize all-day usability (up to 12 hours). Because the display and rendering subsystems are power-hungry, researchers are trying to reduce the power of eye tracking systems to several mW. A recent study by Sony [87] demonstrates a fully integrated neuromorphic eye-tracking system which consumes only 4.22 mW power. Commercial eye tracking systems demand higher resolutions for immersive experiences [76], such as 640×480 [27], [131], which results in higher power consumption. With this resolution, DESSCam consumes only several mW, meeting the power budget while delivering high tracking accuracy.

# VI. EVALUATION

#### A. Accuracy vs. Compression Rate

Our eye-tracking algorithm achieves a lower AE compared to the baselines across a wide range of compression rates (defined by uncompressed size/compressed size). Fig. 9 compares the angular error versus compression rate. Baseline algorithms employ the same ESS method to achieve varying compression rates. Results show that across all compression ratios, our algorithm consistently maintains the gaze estimation accuracy within  $2^{\circ}$ , showing a stronger robustness to tolerate sparse input. In particular, our robust ViT algorithm achieves an angular error of only  $0.5^{\circ}$  even under a  $50\times$  compression rate. This compression rate is used in subsequent experiments.

![](_page_9_Figure_0.jpeg)

Fig. 10. Comparison between our ESS and other sparse sampling alternatives. All experiments are conducted using our ViT algorithm. Compared to other sparse sampling methods, our ESS can retain acceptable accuracy even at high compression rates.

Furthermore, as shown in Fig. 10, our ESS sparse sampling algorithm outperforms other sparse sampling methods, especially under high compression rates. For a fair comparison, we use our robust ViT to evaluate different sparse sampling methods, including the SOTA image sensor for eye tracking with a random sparse-sampling mechanism [47], and an eventdensity based denoising method [45]. Results show that our ESS achieves lower AE compared with these baselines across all compression ratios. Besides, at 50x compression rate, the PAC-only variant (without ESS) reaches an AE of 4.7°, which is unacceptable for AR/VR applications [125]. In contrast, our ESS preserves critical global structure, achieving an AE of only 0.5° even under a 50× compression rate. This improvement stems from ESS's global correlation-aware pixel preselection, which preserves essential information under high compression rates.

#### B. Power Reduction

As shown in Fig. 11, the end-to-end power consumption of our system achieves a 2.7× reduction compared to the SOTA research, BlissCam [47], which outputs 34,257.8 pixels per frame on average (13.38% of the pixel array) by executing ROI prediction. In contrast, our method requires only a few hundred events (2% of the pixel array) for a single inference, thereby achieving much lower system power than BlissCam. There are three main reasons. Firstly, our correlation-based ESS downsamples the pixels by 50×, drastically reducing sampling requirements. Secondly, we implement efficient insensor ESS using a predefined pixel-level mask, avoiding the overhead of refreshing the entire SRAM array to generate random numbers per inference, as required in BlissCam [47]. Thirdly, DESSCam uses simple PAC circuits for token sparsity instead of an in-sensor NPU integrated in BlissCam.

To validate the effectiveness of our ESS and PAC mechanisms in power reduction, we conduct ablation studies. Compared to the 2D-V architecture, our 3D-EPV with ESS reduces power by 5.1×. This is due to: (1) disabling comparators in non-sampling pixels, as comparators dominate per-pixel power consumption [36]; (2) performing local event sampling within the pixel, eliminating the need for comparators to drive the event bus with high load capacitance. Our added SRAM contributes only 2.1% to the pixel's total static power, while

![](_page_9_Figure_6.jpeg)

Fig. 11. Power evaluation results. Our 3D-EPV eye-tracking system reduces power by 5.1× over a standard DVS-based system [111] and 2.7× over the SOTA eye tracking image sensor [47]. This improvement mainly comes from the ESS mechanism, which significantly reduces per-pixel power consumption.

its dynamic power is incurred only when the pixel is enabled for sampling.

Our SSPL-pixel implements ESS-based sampling, which eliminates the dominant power contribution from comparators and sampling circuits in non-sampling pixels by 12.6× and 9.6×, respectively, thereby reducing per-pixel power by up to 5.26×, as illustrated in Fig. 12. As a result, with only 2% active sampling pixels contributing dynamic power, our 3D-EPV achieves a 4.93× reduction in average per-pixel power compared to the no-ESS baseline (3D-PV). In summary, these ablation studies confirm that our ESS and PAC mechanisms are the primary contributors to the 5.1× power reduction over typical DVS-based eye tracking systems and the 2.7× reduction over the SOTA systems.

DESSCam enables the eye tracking system to be deployed in ultra-lightweight AR/VR HMDs, like smart glasses, which typically operate under a limited total system power budget of only tens of mW. For instance, the Meta Ray-Ban smart glasses use micro-batteries with only 154 mAh (about 593 mWh) to keep a light weight (48.6 g) [99], [108]. As a result, the power budget of the glasses is only 49.6 mW to realize all-day usability (up to 12 hours). Because the display and rendering subsystems are power-hungry, researchers are trying to reduce the power of eye tracking systems to several mW. A recent study by Sony [87] demonstrates a fully integrated neuromorphic eye-tracking system which consumes only 4.22 mW power. Commercial eye tracking systems demand higher resolutions for immersive experiences [76], such as 640×480 [27], [131], which results in higher power consumption. With this resolution, DESSCam consumes only several mW, meeting the power budget while delivering high tracking accuracy.

