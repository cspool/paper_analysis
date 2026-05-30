# 4 Experiment

We conducted a comprehensive experimental evaluation of ML-Mamba through four aspects: benchmarking evaluation: We used six commonly used visual language model (VLM) benchmarks to evaluate the effectiveness of the proposed method. These benchmarks include four open-ended visual question answering tasks that require different reasoning abilities, as well as two closed set prediction tasks that involve determining spatial relationships of objects and detecting visual illusions.

- Efficiency evaluation: We conducted a comparative evaluation of ML-Mamba and other Transformer based models at similar model sizes to validate our model's improvement in efficiency.
- Ablation study: We further explored some design choices in the model structure through ablation studies to determine which components have a significant impact on model performance.
- Comparison of answer generation quality: We have provided specific examples to demonstrate the comparison of our model with other models in terms of answer generation quality. Through these experiments, we comprehensively evaluated the performance and advantages of ML-Mamba.

#### 4.1 Experimental Setup

Table [1](#page-8-0) details the hyperparameters of the ML-Mamba model. For the visual encoder part, DINOv2 adopts the same ViT structure as in its original paper, namely a ViT-Large model with 304M parameters, pretrained on the LVD-142M dataset. SigLIP uses a slightly larger shape-optimized version than ViT-Large. The resolution of the input images is set to 384x384, with the number of visual tokens being 729.

The backbone of the LLM is initialized using the pretrained weights from the Mamba-2 model, while the multimodal connectors (MSC) and projectors are always randomly initialized. We chose an open-source model weight from the Huggingface platform to initialize our model as the LLM backbone for our proposed model.

The entire training process took approximately 31 hours on 8 NVIDIA A100 80GB GPUs. During training, we used Pytorch's fully shared data parallel framework [\[53\]](#page-14-14) and adopted automatic mixed precision with FP32 and BF16 for distributed training. The batch size was set to 64. We used the AdamW [\[37\]](#page-13-14) optimizer and updated the network parameters using a learning rate with cosine decay. The learning rate was set to 2 × 10<sup>−</sup><sup>5</sup> , the decay factor was 0.1, and the warm-up ratio was 0.03. The model was trained for 2 epochs with supervised fine-tuning.

