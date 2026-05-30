# <span id="page-19-0"></span>D INTRODUCTION TO BASELINES

LLaVA-Mini is an image and video LMM, so we compare it with several advanced image-based and video-based LMMs.

#### D.1 IMAGE-BASED LMMS

We compare LLaVA-Mini with LLaVA-v1.5 [\(Liu et al., 2023b\)](#page-13-0) and other advanced LMMs of similar data and model scales, including BLIP-2 [\(Li et al., 2023a\)](#page-13-3), InstructBLIP [\(Liu et al., 2024a\)](#page-14-3), IDEFICS [\(Laurenc¸on et al., 2023\)](#page-13-11), Qwen-VL [\(Bai et al., 2023\)](#page-10-4), Qwen-VL-Chat [\(Bai et al., 2023\)](#page-10-4), SPHINX [\(Lin et al., 2023b\)](#page-13-12), mPLUG-Owl2 [\(Ye et al., 2024c\)](#page-16-9).

LMMs with Fewer Vision Tokens Additionally, we assess LLaVA-Mini against various efficient LMMs that utilize fewer vision tokens, showing advantages in compression rate and performance. Most of these models share the same architecture and training data as LLaVA, primarily focusing on the merging of vision tokens in the vision encoder. These efficient LMMs are introduced as follows.

MQT-LLaVA [\(Hu et al., 2024\)](#page-12-0) introduces a flexible query transformer that allows encoding an image into a variable number of visual tokens (up to a predefined maximum) to adapt to different tasks and computational resources.

PruMerge [\(Shang et al., 2024\)](#page-14-7) reduces visual tokens in LMMs by identifying and merging important tokens based on the attention sparsity in vision encoder. PruMerge has a variant, named PruMerge++, which enhances the original PruMerge method by evenly adding more vision tokens (about 144 vision tokens) to further improve performance.

LLaMA-VID [\(Li et al., 2023b\)](#page-13-5) LLaMA-VID compresses the instruction and image into one token respectively, with a total of two tokens representing each image, thus facilitating the understanding of longer videos.

VoCo-LLaMA [\(Ye et al., 2024d\)](#page-16-4) compresses all vision tokens using language models, significantly improving computational efficiency.

TokenPacker [\(Li et al., 2024e\)](#page-13-2) is a visual projector that efficiently reduces visual tokens by 80% using a coarse-to-fine approach.

Previous methods have often focused on reducing the number of vision tokens output by the vision encoder. LLaVA-Mini takes this a step further by shifting attention to how vision tokens and text tokens interact within the LLM backbone. Based on this insight, we propose modality pre-fusion, which enables better performance even under the extreme compression of reducing vision tokens to just one token.

