# **B.2 Pseudocode to Code**

This task involves translating pseudocode into C++ code, with a one-to-one correspondence between pseudocode and C++ lines. For examples of this translation, see Figure [F.2,](#page-26-0) and for the evaluation prompt, see Prompt [H.2.](#page-43-0)

We use the dataset from the original SPOC paper [\(Kulal et al.,](#page-12-5) [2019\)](#page-12-5), where each example includes line-by-line pseudocode annotations and associated test cases. A C++ translation is considered correct only if it passes all test cases. To create our test sets, we randomly sample programs from the combined training, development, and test splits of the original dataset. We formed two test sets: one containing 200 randomly sampled programs with outputs less than 0.5K tokens, and another 200 randomly sampled programs with outputs greater than 0.5K tokens.

