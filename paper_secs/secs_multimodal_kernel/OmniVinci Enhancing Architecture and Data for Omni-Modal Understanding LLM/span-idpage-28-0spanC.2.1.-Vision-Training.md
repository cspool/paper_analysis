# <span id="page-28-0"></span>*C.2.1. Vision Training*

The modality-specific vision training aims to train the model with visual understanding ability. We follow NVILA [\[69\]](#page-16-9) training recipe including five stages:

**Stage 1 | Vision Projector Alignment.** This stage learns to project visual information through a visual projector. This stage ensures that the visual embeddings are compatible with the language model's token embeddings, which is essential for smooth downstream integration. The model is trained on image-text pairs with simple captioning-style supervision, setting a baseline understanding of visual semantics. Only the vision projector is tuned during this process.

**Stage 2 | Vision Encoder Alignment.** With the projector aligned, the model now focuses on enhancing the vision encoder's capacity to process diverse visual content. In this stage we train only the vision encoder and visual projector.

**Stage 3 | Vision Pre-Training.** During this core stage, the model is trained on large-scale multimodal data to learn how to interpret and generate image descriptions. The vision encoder is kept frozen, while the vision projector and the LLM are fine-tuned.

**Stage 4 | Image Instruction Tuning.** In this stage the model is fine-tuned with vision instruction-following capabilities. It is trained to answer multimodal questions, generate captions, reason over scenes, interpret documents, and more. Training data covers a broad range of multimodal capabilities. It includes high-quality instructional examples to align the model with human preferences, datasets for generating rich image captions, and tasks that develop logical and visual reasoning skills. The model is also trained to interpret documents and embedded text, answer general and knowledge-based visual questions, and handle diagrams, visual dialogues, and multimodal instructions. In this stage, all model parameters are fine-tuned.

**Stage 5 | Video Instruction Tuning.** In the final vision alignment stage, the model is adapted to video understanding. The goal here is to enable temporal reasoning and visual understanding over sequences of frames. This includes tasks such as activity recognition, multi-frame object tracking, and answering time-sensitive questions. The whole model is fine-tuned.

Through this vision alignment process, we obtain the "vision preliminary checkpoint" with well-trained vision encoder, projector, and language model.

## <span id="page-28-1"></span>*C.2.2. Audio Training*

Starting from the language model in the above vision preliminary checkpoint we next train the audio understanding ability of our model, which involves (i) audio projector and encoder alignment step followed by (ii) audio instruction tuning.

**Stage 1 | Audio Projector & Encoder Alignment.** This phase focuses on aligning audio encoder and its associated compression layer. We keep the parameters of the language model and vision side fixed. Training consumes 50K audio-language pairs curated from public datasets spanning across audio-based (music, non-speech sound, and speech) question answering, speech-to-text captioning, and automatic speech recognition. By training on this heterogeneous dataset, we encourage the audio projection module to learn a unified representation that aligns well with the language model's semantic space.

**Stage 2 | Audio Instruction Tuning.** During the second stage of training, the audio encoder, audio projection module, and language model are fine-tuned in a unified, end-to-end manner. This joint optimization allows the system to develop a comprehensive and deeply integrated understanding of audio. This stage consumes a comprehensive audio-SFT dataset overseeing 9.6 million samples, including but not limited to audio-based question answering (AudioEntailmentQA [\[25\]](#page-14-12), Clotho-AQA [\[67\]](#page-16-16), DCASE-2025-train [\[109\]](#page-18-14), etc.), audio captioning (AudioCaps [\[52\]](#page-15-14), Clotho-v2 [\[29\]](#page-14-13), Miradata [\[50\]](#page-15-15)-recaptioned, etc.), speech emotion recognition (CREMA-D [\[9\]](#page-13-13), IEMOCAP [\[8\]](#page-12-6), MELD [\[82\]](#page-17-17), etc.), automatic speech recognition (CV-ASR [\[5\]](#page-12-7), Europarl-ASR [\[54\]](#page-15-16), LibriSpeech-ASR [\[81\]](#page-17-8), etc.), and speech translation (MuST-C [\[26\]](#page-14-14), Emilia [\[45\]](#page-15-17), etc.). This allows the model to learn both low-level acoustic features and high-level semantic representations, enabling robust generalization across multiple audio understanding tasks and versatile capabilities in interpreting complex auditory inputs. At this point, we find that the model's ability to perform visual understanding tasks

<span id="page-29-1"></span>is worse, which motivates us to pursue the subsequent omni-modal joint training.

## **C.3. Omni-Modal Joint Training Details**

We adopt a cosine learning rate schedule, preceded by a linear warm-up phase over the first 3% of the training data. The base learning rate is set to 2 × 10−<sup>5</sup> . During training, the vision and audio encoders are kept frozen. The total token count is approximately 200 billion.

