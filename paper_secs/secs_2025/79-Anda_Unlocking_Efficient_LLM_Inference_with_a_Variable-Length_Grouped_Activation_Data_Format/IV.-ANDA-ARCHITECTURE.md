# IV. ANDA ARCHITECTURE

In this section, we first present the three key components of the Anda architecture: (a) A variable-length activation data layout in on-chip memory storage, (b) an Anda-enhanced bit-serial processing unit, and (c) a runtime bit-plane compressor for output activations. These components collectively enhance storage efficiency, computational performance, and energy

![](_page_6_Figure_0.jpeg)

Fig. 10. The proposed bit-plane data layout scheme in memory for efficient variable-length activation data storage.

conservation. Finally, we present how these components integrate to form the overall Anda architecture, a computing system optimizing LLM inference using the Anda format.

