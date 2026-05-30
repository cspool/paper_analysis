# <span id="page-2-1"></span>3. Problem Statement

Given a long video V = {f1, f2, . . . , f<sup>N</sup> }, where f<sup>i</sup> represents the i-th frame and N is the total number of frames, our objective is to develop a model M that can efficiently process V and construct an internal understanding U of its content. This understanding should enable the model to answer queries Q or follow instructions I related to the video content. Formally, we aim to find an optimal function:

$$\mathbf{M}: (V, I) \to U \tag{1}$$

such that:

- U captures episodic and semantic information from V .
- U can be used to maximize the probability P(A|Q, U) of generating correct answers A to queries Q about the video.

The key challenges in this formulation are:

- Temporal Complexity: Efficiently processing N frames, where N can be very large.
- Semantic Understanding: Extracting high-level concepts and narrative structure from video content.
- Memory Constraints: Developing a method to maintain relevant information without exhausting computational resources.

Addressing these challenges requires an approach that can effectively compress temporal information while preserving both detailed episodic content and high-level semantic understanding. In the following section, we propose a cognitively inspired framework to tackle these challenges.

