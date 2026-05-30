# <span id="page-13-1"></span>H.7. Qualitative Results

Animal Identification. Figure [11a](#page-14-2) demonstrates our model's superior performance in animal identification compared to MovieChat. In this example, MovieChat incorrectly identifies a leopard as a cheetah, despite no cheetah being present in the video. This misidentification underscores the importance of accurate visual feature extraction and semantic understanding in long-form video analysis.

Animal Counting. Figure [11b](#page-14-2) showcases our model's ability to perform complex counting tasks, even with limited information. The task involves counting baby bears, which appear infrequently in the video. Despite analyzing only 100 frames

compared to MovieChat's 2048 frames, our model accurately locates and counts the baby bears. This demonstrates the efficiency of our ECO and SeTR modules in capturing and retaining crucial information from sparse appearances.

Determining People's Relationships. In Figure [11c,](#page-14-2) we compare our model's performance against MA-LMM in determining relationships between people over extended video sequences. Both models were trained on the LVU dataset. Our model's superior performance in this task can be attributed to the episodic memory compression technique, which allows for better retention and analysis of interactions across thousands of frames.

## <span id="page-13-0"></span>H.7.1. Visualization of ECO and SeTR

Figure [12](#page-15-0) demonstrates the inner-workings of ECO and SeTR. The top row illustrates a curated summary of the video content, highlighting diverse scenes, such as landscapes, wildlife, and environmental features.

SeTR is responsible for extracting high-level semantic features and grouping frames with similar themes, as shown in the mid row. For instance, the module effectively captures thematic clusters such as "Landscape," "Various Birds," and "Reptiles," providing a concise overview of the video.

Meanwhile, ECO processes the video at a more granular level, segmenting it into coherent episodes that reflect the narrative flow. The bottom row showcases this segmentation, organizing the content into episodic units like "Arid Landscape," "Lake and Aquatic Bird," and "Flies." This twotiered approach ensures both thematic abstraction and temporal coherence, enabling a comprehensive understanding of the video.

