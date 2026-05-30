# Description and Reproduction Details of Baselines

All baselines in our comparison follow a common two-stage pipeline: upcycling  $\rightarrow$  post-training. In the post-training stage, to ensure a fair comparison, all models are trained solely on the extended calibration dataset  $D_{\rm cal}$  (as constructed in our framework), using the AdamW optimizer for 6 epochs with a learning rate of 5e-5. In the post-training phase, all modules are trainable, and the learning rate remains consistent.

The upcycling strategies adopted by each baseline are as follows:

- (a) BTX constructs experts by directly reusing the FFN weights from each dense model. The shared backbone is formed by linearly averaging all model weights.
- (b) BAM reuses FFN weights and partial attention weights  $(W^q, W^o)$  to construct experts, while the remaining weights are linearly averaged to form the shared backbone.
- (c) Drop-Upcycling reuses FFN weights and applies Gaussian perturbation to randomly selected parameters to prevent expert homogenization. The remaining weights are reused and averaged to build the shared backbone, and the expert modules are also updated during training.

## **Additional Experimental Results**

For small-scale models, the Llama series uses Llama3.2 1B for all versions  $(M_1,M_2,M_3,M_4)$  to isolate the impact of varying training tasks, data, and parameters on model performance.