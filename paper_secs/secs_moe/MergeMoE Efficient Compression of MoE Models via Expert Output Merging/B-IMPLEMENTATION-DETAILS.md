# B IMPLEMENTATION DETAILS

Similar to M-SMoE, when reducing the number of experts from N to M, we maintain N references of experts while letting them point to M real experts. In that way, the matrix A is implicit encoded. In addition, for the compression matrix T1, we calculate it in the GPU memory with the least square method. To maximize the number of samples used while avoiding out-of-GPU-memory errors, we adopt the BFloat32 data type. We perform the compression layer by layer. For each layer, we use Torch hooks to obtain intermediate activations, perform the least square method and release the memory after computation. The merging process traverses the layers from back to front because merging the later layers does not affect the activations of the earlier layers.

