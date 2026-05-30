# <span id="page-8-0"></span>C. Experimental details

### C.1. Training data

Our training process contains three stage. In the first stage, we pretrain our model on vision-language alignment using the single image instruction tuning dataset, LLaVA-OneVision [\[30\]](#page-10-1). In the second stage, we train our model on a vision-focused video-language dataset without audio. Specifically, we construct the training dataset using LLaVA-Video [\[85\]](#page-12-10), VideoChat2-IT [\[34\]](#page-10-9) and MovieChat [\[52\]](#page-11-2). In the third stage, we train our model on an audio-visual video understanding dataset to enable the model to comprehend multiple modalities jointly. Our training data is collected from Music-AVQA [\[31\]](#page-10-11), AVQA [\[70\]](#page-12-11), AVSD [\[3\]](#page-9-6), Long-VALE [\[17\]](#page-9-2) and AVInstruct [\[73\]](#page-12-12). We also sample a subset from the data used in stage 2 to retain the capabilities learned in previous stages. The detailed data sources are listed in Table [7.](#page-9-14)

### C.2. Implementation details

We mainly conduct experiments with two backbone LLMs: Qwen2-7B [\[49\]](#page-11-0) and LLaMA3.2-3B [\[41\]](#page-10-0). We sample 1 frame per second for each video. Following previous work [\[51,](#page-11-3) [54\]](#page-11-1), we use DINOv2 [\[46\]](#page-11-12) and SigLIP [\[75\]](#page-12-13) as visual encoders, and obtain 144 aggregated tokens per frame. For audio encoding, following the implementation in BEATs [\[8\]](#page-9-8), we resample the raw audio waveform to 16,000 Hz, and extract audio tokens using the pretrained BEATs encoder, resulting in about 50 tokens per second. We set the maximum number of scene segments to 24, and the number of query tokens to 16 by default. We use the pretrained BERT [\[13\]](#page-9-11) to initialize the Q-Former.

The models are trained for one epoch in each stage. During training, the visual and audio encoders are kept frozen, while the temporal compressor and the MLLMs are trained. In the first two stages, we train the full model parameters. In the third stage, we apply Low-Rank Adaptation (LoRA) [\[25\]](#page-10-23) to reduce GPU memory consumption. The detailed hyperparameter settings used during model training are presented in Table [5.](#page-8-5)

<span id="page-8-5"></span>

| Training Stage             |      | Stage 1 Stage 2 Stage 3 |      |
|----------------------------|------|-------------------------|------|
| Max Sequence Length        |      | 8192                    |      |
| Number of Video Frames     |      | 1 fps                   |      |
| Number of Segmented Scenes |      | 24                      |      |
| Visual Tokens per Frame    |      | 144                     |      |
| Audio Tokens per Frame     | 50   |                         |      |
| Context Tokens per Frame   | 16   |                         |      |
| Optimizer                  |      | AdamW [42]              |      |
| Learning Rate              | 1e-5 | 1e-5                    | 2e-5 |
| Learning Rate Schedule     |      | Cosine Decay            |      |
| Warmup Ratio               |      | 0.03                    |      |
| Training Mode              | Full | Full                    | LoRA |

Table 5. Hyperparameters Used in Model Training.

### C.3. Evaluation setup

Following the approach in [\[10\]](#page-9-1), we adopt an LLM assisted evaluation for AVSD. We also provide an example as one shot. For LVCoT, we set the number of segments to 3 by default.

