# A Appendix

### A.1 Training Details

Due to the need to evaluate accuracy and token count on a validation set every n steps, our validation set consists of 512 questions sampled from past questions in AIME-1983 to AIME-2023. The original ratio for shortcot and longcot is set to 0.5:0.5, with an evaluation interval of every 32 steps. The model is allowed to train for a total of 2000 steps, and the learning rate is set as a constant at 1e-5. For the 7B and 14B models, we conducted training on two 8-GPU (80GB) machines, with one 8-GPU machine performing vllm inference and the other performing training. Every n steps, parameter synchronization is executed using vllm's parameter sync function. For the 32B model, we performed SFT on five 8-GPU nodes and deployed the inference service on one node.

We ultimately select the checkpoint with the shortest token length among those whose accuracy on the validation set is no less than 30% of that achieved by the original long CoT.

