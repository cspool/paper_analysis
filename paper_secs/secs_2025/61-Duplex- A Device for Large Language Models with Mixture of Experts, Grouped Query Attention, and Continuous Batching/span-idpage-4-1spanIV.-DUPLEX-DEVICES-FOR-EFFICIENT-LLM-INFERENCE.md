# <span id="page-4-1"></span>IV. DUPLEX: DEVICES FOR EFFICIENT LLM INFERENCE

To address the challenges in the hetero system, we propose Duplex, which configures separate processing units for high-Op/B and low-Op/B operations that share device memories. Appropriate processing units are selected for each operation based on the stage. The low-Op/B unit handles the MoE layers during the decoding-only stage as well as the attention layers of the decoding-only stage and of decoding sequences in the mixed stage. The high-Op/B unit manages the rest. We opt for an HBM-based system to provide high memory bandwidth.

