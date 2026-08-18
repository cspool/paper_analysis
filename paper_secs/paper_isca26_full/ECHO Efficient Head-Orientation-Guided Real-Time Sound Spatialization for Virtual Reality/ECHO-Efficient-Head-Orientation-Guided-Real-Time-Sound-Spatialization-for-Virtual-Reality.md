# ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality

Haiyu Wang *Tandon School of Engineering New York University* NY, USA hw3689@nyu.edu

Tianhua Xia *Tandon School of Engineering New York University* NY, USA tx856@nyu.edu

Sai Qian Zhang *Tandon School of Engineering New York University* NY, USA sai.zhang@nyu.edu

*Abstract*—Immersive virtual reality (VR) experiences depend on the seamless coordination of visual and auditory feedback under tight latency constraints. While decades of research have refined real-time graphics pipelines, low-latency sound spatialization remains computationally intensive and relatively underexplored, despite its equal importance to user presence. Auditory perception is highly sensitive to head orientation, environmental acoustics, and timing precision. On compact, resource-limited head-mounted displays (HMDs), the heavy computational workload often causes spatial audio to lag behind visual rendering, degrading perceptual realism and user comfort.

We present ECHO, an efficient head-orientation-guided framework for real-time sound spatialization that leverages natural head dynamics to reduce redundant computation. ECHO dynamically combines audio sources based on the listener's orientation and co-optimizes head pose estimation algorithms with hardware for low-latency performance. Integrated as a plug-in within VR HMD SoCs, ECHO achieves up to 2.91× lower sound spatialization latency while maintaining high-fidelity auditory realism across diverse acoustic environments.

