# <span id="page-23-0"></span>F Architectural details

## <span id="page-23-2"></span>F.1 Embedder methods

Lookup For our main embedder design, we simply retrieve token-level embeddings from a lookup table and concatenate them to obtain the input block embeddings. The token-level embedding dimension is set to be 1/L<sup>B</sup> of the main model dimension.

Encoder To ablate the effect of adding encoding capability to the embedder, we encode the input tokens of a block with a small RoBERTa-based encoder. We use a fixed sized encoder with dimension size of 256 and 3 hidden layers. We concatenate the output hidden states and apply linear projection to obtain the input block embedding.

CLS token To investigate the feasibility of an embedder that can accept various input block lengths, we use CLS tokens previously used to extract sentence embeddings [\[27\]](#page-12-11). We use the same model size as the RoBERTa model and encode information in 3 CLS tokens, to increase the embedding dimension while minimizing the model dimension of the embedder. Similar to the RoBERTa embedder, we concatenate the output hidden states of the CLS tokens and apply linear projection to obtain the input block embedding.

## <span id="page-23-3"></span>F.2 Token decoder methods

Prefix For the main token decoder design, we incorporate the context embeddings from the block decoder by projecting them as prefix token embeddings. The token decoder can retrieve the context information from the prefix tokens via attention, and also further encode the context information. We can use multiple prefix tokens, i.e., increase the prefix length, to increase the computational width [\[34\]](#page-12-3) of the token decoder to increase performance with addtional FLOPs, are relatively cheap in terms of inference time in the token decoder.

Summation We also consider the summation method used in previous work [\[84\]](#page-16-0). Here, the context embeddings are projected to L<sup>B</sup> embeddings of dimension D and added to the token embeddings at each input position of the token decoder. This does not benefit from additional computation of the context information in the token decoder.

Cross-attention Finally, we consider an approach that uses cross-attention, treating the output context embedding as the output hidden states of an encoder in an encoder-decoder transformer [\[63\]](#page-14-10). Specifically, we project the the context embedding into L<sup>B</sup> hidden states each with dimension D and apply cross-attention between self-attention and feedforward operations at each transformer layer in the token decoder. This also does not benefit from additional computation of the context information in the token decoder.

