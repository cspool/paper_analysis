# **6 Conclusion**

We have introduced a hybrid RNN-Attention model architecture (GoldFinch) and trained models that demonstrate its performance up to 1.45B. The resulting hybrid RNN-Attention models combine the efficiency of RNNs with the capabilities of attention-based models. Having RNNs for the initial layers allows for fast pre-fill and removes the need for positional encoding on the RNN layers, while the attention layers improve associative recall. The combination with a highly compressed global KV-Cache unlocks a memory reduction in inference while maintaining enhanced performance. We release the trained weights and training code under the Apache 2.0 license.

## **7 Acknowledgements**

Special thanks to Bo Peng for his tireless dedication to the RWKV architecture and community. The main GoldFinch code herein was based on a modified version of his public Linear Attention Arena code repository, and upgraded models were based on his pre-trained Finch model releases.

