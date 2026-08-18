# <span id="page-6-3"></span>C. SoC Integration of ECHO Accelerator

The computational flow of ECHO is illustrated in Figure 10 and Figure 11. The ECHO accelerator operates in a **Hybrid mode** comprising **Monocular–Inertial (MI)** and **Inertial–Only (IO)** modes. In MI mode, SLAM-based pose estimation is performed using monocular images and IMU data, as shown in Figure 6 (b). Between MI steps, IO mode employs the lightweight RNN-based pose estimation module in Section III-D to generate high-frequency pose estimates

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Fig. 11: The computational flows of: (a) MI mode and (b) IO mode. The step numbers are shown in circles.

with shorter intervals. The two modes operate in an interleaved manner, resulting in a reduced  $T_{IN}$  in Equation 1. We also pipeline the sensing, pose estimation, audio rendering, and audio output stages to improve steady-state throughput. Next, we describe MI and IO modes separately.

For MI mode, as shown in Figure 11 (a), in Step 1, the SLAM camera captures a frame and passes it to CPU. CPU constructs the image pyramid and partitions each level into small cells which are then written to system memory in Step 2. In Step 3, ECHO accelerator reads the image cells from memory into its internal buffer and performs FAST detection using its ORB extractor module. The extracted points are stored back to memory in Step 4. Following this, CPU continues the remaining computations in the SLAM tracking thread and performs point filtering described in Section III-C (Step 5). During local map tracking, as part of the iterative pose optimization process, CPU invokes ECHO accelerator to compute reprojection errors and Jacobians for the 2D-3D correspondences (Step 6). Once these results are returned, CPU completes the Gauss-Newton iteration to refine the pose. After that, the optimized pose is used to perform acoustic foveation and spatial clustering of audio sources. These clustered audio sources, along with the optimized pose, are then passed to GPU (Step 7). In Step 8, GPU performs SS using the clustered audio sources and the listener pose to generate the final binaural audio stream. Finally, the rendered audio is sent to DAC and played back through the headphones (Step 9).

The operations in IO mode are shown in Figure 11 (b). The ECHO accelerator directly reads IMU measurements (Step 1) and computes the updated pose with RNN (Step 2). The estimated pose is then passed to CPU, which proceeds with the same subsequent stages as in Steps 7–9 of the MI mode, including audio source clustering (Step 3), audio rendering (Step 4), and audio output (Step 5).

Figure 10 illustrates the ECHO workflow in Hybrid mode. Compared to the workflow in Figure 6 (b), ECHO executes pose estimation more frequently, substantially reducing  $T_{IN}$ . In addition, the techniques introduced in Section III significantly decrease both pose estimation and audio rendering latencies. Let  $T_P^{MI}$  and  $T_P^{IO}$  denote the pose estimation latencies for the MI and IO modes, respectively, and  $T_S^{MI}$  and  $T_S^{IO}$  represent the sensing latencies in the two modes. Given the small data volumes exchanged between SoC components, communication latency is negligible. The motion-to-sound latency  $T_{m-s}$  can be approximated under two scenarios. When motion occurs between two consecutive SLAM

<span id="page-7-1"></span>TABLE II: Hybrid mode results in ATE (m)\RRE (°).

| $ATE \backslash RRE$ | ЕСНО                | ORB-SLAM3           | VINS-Fusion | HybVIO      | OKVIS       |
|----------------------|---------------------|---------------------|-------------|-------------|-------------|
| AEA 1                | 0.072\2.461         | 0.063\2.389         | 1.020\2.655 | 0.997\4.271 | 0.198\2.707 |
| AEA 2                | <b>0.068</b> \2.817 | 0.070\2.830         | 1.259\7.319 | 1.273\3.882 | 0.225\2.812 |
| AEA 3                | <b>0.038</b> \1.675 | 0.040\1.699         | 0.462\1.669 | 0.491\5.928 | 0.067\1.928 |
| AEA 4                | <b>0.042</b> \1.581 | 0.070\ <b>1.512</b> | 0.884\6.140 | 0.910\2.187 | 0.078\1.904 |
| TUM 1                | 0.015\0.513         | <b>0.011</b> \0.612 | 0.125\0.533 | 0.032\0.481 | 0.084\0.495 |
| TUM 2                | 0.021\ <b>0.559</b> | <b>0.015</b> \0.626 | 0.089\0.630 | 0.032\0.680 | 0.115\0.761 |
| TUM 3                | 0.020\ <b>0.645</b> | <b>0.019</b> \0.711 | 1.235\1.662 | 0.094\0.675 | 0.073\0.783 |
| TUM 4                | 0.020\ <b>0.476</b> | <b>0.017</b> \0.537 | 0.064\0.490 | 0.026\0.698 | 0.040\0.567 |
| TUM 5                | 0.019\ <b>0.461</b> | 0.013\0.555         | 0.138\0.526 | 0.034\0.523 | 0.062\0.554 |
| TUM 6                | 0.010\ <b>0.399</b> | <b>0.007</b> \0.466 | 0.130\0.429 | 0.028\0.419 | 0.047\0.524 |
| Average              | 0.033\1.014         | <b>0.033</b> \1.194 | 0.541\2.205 | 0.392\1.974 | 0.099\1.303 |

camera captures and IO mode is used for pose estimation,  $T_{m-s}^{IO} = T_{IN} + T_S^{IO} + T_P^{IO} + T_{R1} + T_{R2} + T_O$ . When motion occurs and is detected using MI mode,  $T_{m-s}^{MI} = T_{IN} + T_S^{MI} + T_P^{MI} + T_{R1} + T_{R2} + T_O$ . The overall motion-to-sound latency is given by  $T_{m-s} = \max(T_{m-s}^{MI}, T_{m-s}^{IO})$ . Typically,  $T_S^{MI} + T_P^{MI}$  far exceeds  $T_S^{IO} + T_P^{IO}$ . Therefore, the overall motion-to-sound latency is dominated by the occurrence of  $T_{m-s}^{MI}$ .

