# **B** Qualitative Model Generation Samples

We further validate our converted models and our inference implementation by testing the FarSkip-Collective models qualitatively in chat-based generation for which we use the same generation

parameters as the original model, for example we observe the following sensible answer when prompting the FarSkip-Collective DeepSeek-V2 Lite in Fig. [9.](#page-17-0)

<span id="page-17-0"></span>*DeepSeek-V2 Lite Instruct (original)*

Prompt What are the main topics of Linear Algebra?

Generation *Linear Algebra is a branch of mathematics dealing with vectors, vector spaces (also called linear spaces), linear transformations, and systems of linear equations. It has a wide range of applications in physics, engineering, computer science, economics, and social sciences. The main topics of Linear Algebra include:*

*DeepSeek-V2 Lite Instruct (FarSkip-Collective)*

Prompt What are the main topics of Linear Algebra?

Generation *Linear Algebra is a branch of mathematics that deals with the study of vectors, vector spaces, linear transformations, and systems of linear equations. The main topics of Linear Algebra include: 1. Vectors: Understanding the concept of vectors and their properties, such as addition, subtraction, and scalar multiplication.*

Figure 9: Model generations of a fully converted FarSkip-Collective DeepSeek-V2-Lite model (bottom) and the original model checkpoint (top). Both generations use the default generation hyperparameters (τ = 0.3, p = 0.95).

