# 2 Related work

## 2.1 Video Enhancement

Video enhancement aims to exploit sub-pixel information from contextual frames to improve the quality and resolution of videos, which primarily includes video frame interpolation, video superresolution, and spatio-temporal video super-resolution.

Video Frame Interpolation (VFI) targets generating the intermediate frames between given consecutive inputs. Early CNN-based methods [\[1,](#page-10-0) [18,](#page-11-0) [23,](#page-11-1) [40,](#page-12-1) [77\]](#page-14-3) mainly rely on optical flow estimation or direct frame synthesis, but often suffer from limited receptive fields and poor handling of large motion. Therefore, Transformer-based approaches [\[35,](#page-12-0) [47,](#page-13-0) [74\]](#page-14-2) have been proposed to model long-range dependencies, significantly improving the quality and detail of generated video.

Video Super-Resolution (VSR) aims to reconstruct high-resolution (HR) video from low-resolution (LR) inputs. Early VSR methods primarily used explicit optical flow alignment [\[3,](#page-10-1) [5,](#page-10-9) [56\]](#page-13-9), dynamic filtering [\[24\]](#page-11-3), deformable convolutions [\[60\]](#page-13-10), and temporal attention mechanisms [\[28,](#page-11-6) [65\]](#page-14-8). With the increasing prominence of the Transformer's powerful representation capabilities, numerous Transformer-based VSR methods have been proposed, achieving progressive success. For example, PSRT [\[55\]](#page-13-11) leverages a multi-frame self-attention mechanism to jointly process features from the current input frame and the propagated features. MIA [\[76\]](#page-14-9) further boosts performance by leveraging masked intra-frame and inter-frame attention blocks to better use of previously enhanced features.

Spatio-Temporal Video Super-Resolution (STVSR) aims to simultaneously enhance spatial and temporal resolution, combining VSR and VFI, and presents greater challenges. Among the most representative real-time Transformer-based models is RSTT [\[12\]](#page-10-4), which achieves state-of-the-art performance by constructing feature dictionaries from different levels of encoders and repeatedly querying them during the decoding stage.

Although powerful transformer-based models have demonstrated superiority in enhancing spatial resolution and perceptual quality, their high computational cost hinders practical deployment. This paper is the first to propose an efficient model compression method specifically for video enhancement to facilitate its deployment.

