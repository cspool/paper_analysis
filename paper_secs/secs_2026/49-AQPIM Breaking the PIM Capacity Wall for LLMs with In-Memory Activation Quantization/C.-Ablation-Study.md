# *C. Ablation Study*

To evaluate the effectiveness of the importance-weighted kmeans clustering and vector splitting optimization, we conduct ablation studies comparing four configurations: standard PQ, AQPIM without weighting, AQPIM without pre-sorting, and AQPIM. Table IV presents the result for high compression scenarios (128 centroids). We observe that applying both weighting and pre-sorting significantly contributes to accuracy, particularly under aggressive compression situations. The overhead introduced by these techniques is minimal: importance-weighted k-means leverages the attention scores generated during attention computation, and the channel permutation is generated offline using a calibration dataset and integrated into the projection weights during inference.

