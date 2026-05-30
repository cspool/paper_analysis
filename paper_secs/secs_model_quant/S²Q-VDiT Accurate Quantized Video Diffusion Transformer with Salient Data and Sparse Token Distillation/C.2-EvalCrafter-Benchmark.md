# C.2 EvalCrafter Benchmark

For EvalCrafter [\[34\]](#page-12-13) benchmark, consistent with prior work ViDiT-Q [\[62\]](#page-13-5), we select 5 low-level metrics to evaluate the generation performance.

CLIPSIM and CLIP-Temp: CLIPSIM computes the image-text CLIP similarity for all frames in the generated videos, and we report the averaged results. This quantifies the similarity between input text prompts and generated videos. CLIP-Temp computes the CLIP similarity of each two consecutive frames of the generated videos and then gets the averages for each two frames. This quantifies the semantic consistency of generated videos. We use the CLIP-VIT-B/32 [\[50\]](#page-13-13) model to compute CLIPSIM and CLIP-Temp. We use the implementation from EvalCrafter [\[34\]](#page-12-13) to compute these two metrics.

DOVER's VQA: VQA-Technical measures common distortions like noise, blur, and over-exposure. VQA-Aesthetic reflects aesthetic aspects such as the layout, the richness and harmony of colors, the photo-realism, naturalness, and artistic quality of the frames. We use the Dover [\[53\]](#page-13-14) method to compute these two metrics.

FLOW Score: Flow score was proposed in [\[34\]](#page-12-13) to measure the general motion information of the video. We use RAFT [\[48\]](#page-12-16) to extract the dense flows of the video in every two frames, and we calculate the average flow on these frames to obtain the average flow score of each generated video.

We use the prompt sets provided by the official github repository of ViDiT-Q [\[62\]](#page-13-5) to generate 10 videos for evaluation. We also attached the prompt sets in the supplementary material.

