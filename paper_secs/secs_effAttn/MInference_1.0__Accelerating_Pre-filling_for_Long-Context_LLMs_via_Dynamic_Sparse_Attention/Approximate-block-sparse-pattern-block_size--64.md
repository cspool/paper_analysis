# Approximate block-sparse pattern (block\_size = 64)
 $\widehat{Q} \leftarrow \text{MeanPooling}(Q, block_size)$ 
 $\widehat{K} \leftarrow \text{MeanPooling}(K, block_size)$ 
 $\widehat{A} \leftarrow \text{softmax}\left(\widehat{Q}\widehat{K}^{\top}/\sqrt{d} + m_{\text{casual}}\right)$ 

