# II. BACKGROUND AND MOTIVATION

We first introduce the gaze-tracked foveated rendering (TFR) technology (Sec. II-A). Then we present the key per-

#### Motion-to-Photon Latency (MPL)

![](_page_1_Figure_8.jpeg)

Fig. 2. Eye tracking for gaze-tracked foveated rendering. Eye tracking enables the GPU to render only the foveal region in high resolution, while the peripheral regions are rendered at low resolution. This approach reduces rendering workload by up to 72%.

![](_page_1_Figure_10.jpeg)

Fig. 3. Current commercial devices face latency and power bottlenecks. (a) In current commercial devices, eye-tracking latency dominates the end-to-end tracking latency. (b) Reducing latency incurs a high power cost, revealing an inherent tradeoff. Achieving the 1 kHz tracking frequency demands 96 W system power consumption, which is unacceptable in AR/VR HMDs.

formance metrics of eye tracking and our optimization goals (Sec. II-B).

#### A. Gaze-tracked foveated rendering (TFR)

TFR adopts gaze direction data from eye-tracking to render the foveal region at high resolution, while blurring the peripheral regions [88], as shown in Fig. 2. Experiments conducted by Tobii on Pico headsets demonstrate that TFR reduces GPU rendering overhead by up to 72%, with an average reduction of 60% [9].

Although TFR significantly reduces rendering overhead, it is hard to deploy TFR in commercial devices [104], [136]. The main reason is its high motion-to-photon latency (MPL), which includes eye-tracking latency and TFR & display latency, that is, the delay between a virtual action and its perceived visual feedback [13], [130], as illustrated in Fig. 2. Prior studies have proved that MPL should be less than 5 ms to avoid visual discomfort in AR/VR HMDs [19], [59], [66], [139]. However, the MPL of some AR/VR HMDs (e.g., Vive Pro Eye) is up to 79 ms [114]. As illustrated in Fig. 3(a), the MPL is mainly dominated by the eye-tracking latency, which accounts for up to 63.3% of MPL in commercial systems [114] and 77.7% in the SOTA research [44], [47].

To reduce the latency of eye tracking, researchers increase the frame rate of image sensors, but a higher frame rate brings higher power consumption [3], [6], [7], as depicted in Fig. 3(b). Such a latency–power tradeoff makes it difficult to improve the performance of a frame-based eye tracking system. In order to break the tradeoff, we propose DESSCam, leveraging the temporally low-latency characteristics of DVS and a spatially sparse in-sensor ESS mechanism.

#### B. Eye Tracking Metrics

**Tracking Latency.** Eye tracking latency includes the image sensor delay and the NPU delay. The sensor delay originates from the pixel array, readout circuit, and output interface, while the NPU delay scales with the computational complexity of the gaze tracking algorithm. As mentioned in Sec. II-A, eye tracking latency dominates the MPL in AR/VR systems. Low eye-tracking latency is crucial for an immersive user experience. So far, most image sensors for eye-tracking are frame-based [47], [55], [73], [110], [143], which account for 88.3% of the tracking latency [47]. Therefore, our goal is to reduce eye tracking latency by optimizing the image sensor design.

Power Consumption. The power consumption of an eye tracking system includes the power of the image sensor and the NPU. The sensor power includes the power of the pixel array, readout circuits, and output interface, while the NPU power is associated with the computational workload. Most efforts focus on reducing the NPU computational workload by noise suppression [53], region-of-interest (ROI) segmentation [120], or designing lightweight eye tracking algorithms [30], [90], [128]. However, the power of image sensors dominates the power of an eye tracking system [83]. Specifically, DVS consumes the most power of event-based eye-tracking systems [33], [120]. This is because DVS pixels are always on and consume much power [48], [117]. As a result, even under low event activity, DVS consumes tens of mW [52], [111], [117]. In this work, we aim to reduce the power consumption of the DVS pixel array.

**Angular Error.** The angular error (AE) metric is widely used to evaluate the accuracy of gaze tracking systems. It quantifies the angular deviation between the predicted gaze direction and the ground-truth gaze direction [61]. AE is defined as the angular difference between two 3D gaze vectors: the predicted gaze vector and the actual gaze vector, typically calculated as follows [68], [120]:

$$AE = \arccos\left(\frac{\mathbf{v}_{pred}}{|\mathbf{v}_{pred}|} \cdot \frac{\mathbf{v}_{gt}}{|\mathbf{v}_{gt}|}\right) \tag{1}$$

Here,  $\mathbf{v}_{\text{pred}} = (x_{\text{pred}}, y_{\text{pred}}, L_0)$  and  $\mathbf{v}_{\text{gt}} = (x_{\text{gt}}, y_{\text{gt}}, L_0)$ .  $(x_{\text{gt}}, y_{\text{gt}})$  and  $(x_{\text{pred}}, y_{\text{pred}})$  denote the ground-truth and predicted gaze coordinates, respectively.  $L_0$  represents the Euclidean distance between the subject and the screen onto which the gaze point is projected. A smaller AE enables TFR to focus rendering on a compact region, enhancing user immersion while improving computational efficiency [136]. To achieve a low AE even at high data sparsity, we need to design a robust algorithm.

#### C. Summary

Our goal is to significantly reduce the power consumption and latency of eye tracking while ensuring accuracy. To achieve this goal, we propose DESSCam, which reduces latency through the temporally adaptive sampling capability of DVS and reduces power through pixel-wise ESS. Furthermore, we propose a robust ViT algorithm to realize a small AE under high data sparsity.

# II. BACKGROUND AND MOTIVATION

We first introduce the gaze-tracked foveated rendering (TFR) technology (Sec. II-A). Then we present the key per-

#### Motion-to-Photon Latency (MPL)

![](_page_1_Figure_8.jpeg)

Fig. 2. Eye tracking for gaze-tracked foveated rendering. Eye tracking enables the GPU to render only the foveal region in high resolution, while the peripheral regions are rendered at low resolution. This approach reduces rendering workload by up to 72%.

![](_page_1_Figure_10.jpeg)

Fig. 3. Current commercial devices face latency and power bottlenecks. (a) In current commercial devices, eye-tracking latency dominates the end-to-end tracking latency. (b) Reducing latency incurs a high power cost, revealing an inherent tradeoff. Achieving the 1 kHz tracking frequency demands 96 W system power consumption, which is unacceptable in AR/VR HMDs.

formance metrics of eye tracking and our optimization goals (Sec. II-B).

#### A. Gaze-tracked foveated rendering (TFR)

TFR adopts gaze direction data from eye-tracking to render the foveal region at high resolution, while blurring the peripheral regions [88], as shown in Fig. 2. Experiments conducted by Tobii on Pico headsets demonstrate that TFR reduces GPU rendering overhead by up to 72%, with an average reduction of 60% [9].

Although TFR significantly reduces rendering overhead, it is hard to deploy TFR in commercial devices [104], [136]. The main reason is its high motion-to-photon latency (MPL), which includes eye-tracking latency and TFR & display latency, that is, the delay between a virtual action and its perceived visual feedback [13], [130], as illustrated in Fig. 2. Prior studies have proved that MPL should be less than 5 ms to avoid visual discomfort in AR/VR HMDs [19], [59], [66], [139]. However, the MPL of some AR/VR HMDs (e.g., Vive Pro Eye) is up to 79 ms [114]. As illustrated in Fig. 3(a), the MPL is mainly dominated by the eye-tracking latency, which accounts for up to 63.3% of MPL in commercial systems [114] and 77.7% in the SOTA research [44], [47].

To reduce the latency of eye tracking, researchers increase the frame rate of image sensors, but a higher frame rate brings higher power consumption [3], [6], [7], as depicted in Fig. 3(b). Such a latency–power tradeoff makes it difficult to improve the performance of a frame-based eye tracking system. In order to break the tradeoff, we propose DESSCam, leveraging the temporally low-latency characteristics of DVS and a spatially sparse in-sensor ESS mechanism.

#### B. Eye Tracking Metrics

**Tracking Latency.** Eye tracking latency includes the image sensor delay and the NPU delay. The sensor delay originates from the pixel array, readout circuit, and output interface, while the NPU delay scales with the computational complexity of the gaze tracking algorithm. As mentioned in Sec. II-A, eye tracking latency dominates the MPL in AR/VR systems. Low eye-tracking latency is crucial for an immersive user experience. So far, most image sensors for eye-tracking are frame-based [47], [55], [73], [110], [143], which account for 88.3% of the tracking latency [47]. Therefore, our goal is to reduce eye tracking latency by optimizing the image sensor design.

Power Consumption. The power consumption of an eye tracking system includes the power of the image sensor and the NPU. The sensor power includes the power of the pixel array, readout circuits, and output interface, while the NPU power is associated with the computational workload. Most efforts focus on reducing the NPU computational workload by noise suppression [53], region-of-interest (ROI) segmentation [120], or designing lightweight eye tracking algorithms [30], [90], [128]. However, the power of image sensors dominates the power of an eye tracking system [83]. Specifically, DVS consumes the most power of event-based eye-tracking systems [33], [120]. This is because DVS pixels are always on and consume much power [48], [117]. As a result, even under low event activity, DVS consumes tens of mW [52], [111], [117]. In this work, we aim to reduce the power consumption of the DVS pixel array.

**Angular Error.** The angular error (AE) metric is widely used to evaluate the accuracy of gaze tracking systems. It quantifies the angular deviation between the predicted gaze direction and the ground-truth gaze direction [61]. AE is defined as the angular difference between two 3D gaze vectors: the predicted gaze vector and the actual gaze vector, typically calculated as follows [68], [120]:

$$AE = \arccos\left(\frac{\mathbf{v}_{pred}}{|\mathbf{v}_{pred}|} \cdot \frac{\mathbf{v}_{gt}}{|\mathbf{v}_{gt}|}\right) \tag{1}$$

Here,  $\mathbf{v}_{\text{pred}} = (x_{\text{pred}}, y_{\text{pred}}, L_0)$  and  $\mathbf{v}_{\text{gt}} = (x_{\text{gt}}, y_{\text{gt}}, L_0)$ .  $(x_{\text{gt}}, y_{\text{gt}})$  and  $(x_{\text{pred}}, y_{\text{pred}})$  denote the ground-truth and predicted gaze coordinates, respectively.  $L_0$  represents the Euclidean distance between the subject and the screen onto which the gaze point is projected. A smaller AE enables TFR to focus rendering on a compact region, enhancing user immersion while improving computational efficiency [136]. To achieve a low AE even at high data sparsity, we need to design a robust algorithm.

#### C. Summary

Our goal is to significantly reduce the power consumption and latency of eye tracking while ensuring accuracy. To achieve this goal, we propose DESSCam, which reduces latency through the temporally adaptive sampling capability of DVS and reduces power through pixel-wise ESS. Furthermore, we propose a robust ViT algorithm to realize a small AE under high data sparsity.

