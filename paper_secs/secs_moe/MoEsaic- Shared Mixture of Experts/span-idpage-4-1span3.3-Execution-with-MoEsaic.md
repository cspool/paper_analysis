# <span id="page-4-1"></span>3.3 Execution with MoEsaic

During inference, MoEsaic batches the requests from multiple clients directed toward different model instances. It also propagates each client's model id along with the request through the execution of all layers so the routing mechanism can select the correct gate(s).

Since several requests belonging to different clients are processed in a single batch, it prolongs the execution of a batch when compared to a per-client dedicated MoE instance. Below, we describe the source of the increased latency and the techniques that we employ to reduce the MoEsaic's adverse impact on inference performance.

<span id="page-4-0"></span>3.3.1 Fused Gate. A typical MoE model contains as many gates as the number of layers. This means based on the model type it needs to invoke about 50 or so gates for each iteration. This problem becomes worse with MoEsaic, which simultaneously serves tens of MoE model instances. Therefore, it needs to invoke tens of gates for each layer. The repeated invocation of the CUDA kernels for gates results in incremental increase in latency w.r.t. the number of served MoE models.

<span id="page-4-2"></span>

| Model        | Count | GPU Memory<br>(GBs) | GPUs<br>(40GB) |
|--------------|-------|---------------------|----------------|
| Mixtral 4x7B | 4     | 224                 | 8              |
| MoEsaic      | 4     | 140                 | 4              |

Table 1: GPU memory shows the amount of memory required for the model parameters. The model has 2 shared experts. The GPU count is based on minimum possible tensor-parallel mode.

To address this problem, we implement a fused gate, which combines several gates into a single fused gate, where multiple routing requests are processed in a batch. The combined routing efficiently executes several requests in parallel with little impact on the routing latency. Figure [3](#page-3-0) shows the organization of the fused gate. With a fused gate, MoEsaic maintains a gate mapping for each model instance, so the output of the fused gate can be correctly interpreted to select the same experts as the original gate.

3.3.2 Batching of Requests. To deduplicate experts, MoEsaic assigns unique identity to each expert in model structure. As a result, even the identical experts that share the underlying tensors are represented by different nn.Parameter structure in each expert. Therefore, the triton kernel that implements the processing of experts in vLLM, performs the processing of requests for each expert independently. When serving several MoEs, this means large number of experts are invoked for processing a batch of requests, even if they share the GPU memory.

To avoid the separate processing, MoEsaic create a merged representation of identical experts after model initialization so that they are represented by a single nn.Parameter. When processing an inference request, each MoE's gate maps the identity of the expert to its new merged representation. As a result of the merged representation, the requests towards the deduplicated experts are batched for processing even if they belong to different clients. Due to their high performance parallel architecture of GPUs, it can process these larger batches more efficiently.

3.3.3 Security Implications. We expect that MoEsaic will be a hosted by a service provider, while clients only need to submit their models for serving. In such a deployment, the customers or users do not have access to the infrastructure hosting the models. Therefore, they cannot read other clients' data (e.g., requests or activations) or model parameters.

