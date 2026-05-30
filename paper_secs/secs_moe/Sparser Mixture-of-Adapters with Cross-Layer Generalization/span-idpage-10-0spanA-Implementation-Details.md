# <span id="page-10-0"></span>A Implementation Details

We conduct training on an NVIDIA A100 GPU, using a batch size of 128 and a micro-batch size of 4. The learning rate is set to 3e-4 and optimized with the AdamW optimizer, incorporating a warmup of 80 steps and a cosine scheduler with restarts to manage the learning rate decay. LoRA ranks are set to 8 for fine-tuning Phi-3 and 16 for the other models. Training spans 3 epochs.

For all models, we use 8 pre-layer experts (N = 8) and set the number of layers n<sup>l</sup> = 8. The hyperparameter α is swept over the range {0.005, 0.01, 0.05, 0.1}.

We fine-tune four pre-trained models: Phi-2, Phi-3, Gemma, and OLMo. LoRA is applied to different modules in each model:

- For the Phi-2 model, the target modules include q\_proj, k\_proj, and v\_proj.
- For the Phi-3 model, we target the qkv\_proj module.
- For the Gemma model, the k\_proj module is targeted.
- For the OLMo model, we focus on the att\_proj module.

## B Redundancy Analysis

To evaluate the redundancy within the Mixture of LoRA model, we performed two sets of experiments: one focused on progressively masking experts across different layers, and the other on deactivating all but one layer of experts.

Table [5](#page-11-0) shows the performance change when randomly masking experts within specific layers at different masking ratios, where 100% masking represents using the backbone model alone. The results, averaged across 8 commonsense reasoning datasets, show minimal performance degradation as experts are progressively masked. Even with high masking ratios, the performance drop remains within a small margin, suggesting a high level of redundancy in the expert layers. Notably, the variance across datasets is also low, indicating that the model remains robust despite significant expert masking.

In Table [6,](#page-13-0) we explore the extreme case of using experts from only a single layer. Interestingly, for the BoolQ dataset, activating experts in Layer 16 outperformed using all layers, suggesting that

certain layers are more critical to performance than others. However, for most other datasets, deactivating all but one layer led to notable performance drops, particularly in later layers such as Layer 32. This analysis highlights that while some layers may be redundant, others play a key role in task-specific performance, and the importance of each layer can vary across datasets.

These findings emphasize the potential for reducing model complexity by selectively utilizing experts without significant performance trade-offs.

