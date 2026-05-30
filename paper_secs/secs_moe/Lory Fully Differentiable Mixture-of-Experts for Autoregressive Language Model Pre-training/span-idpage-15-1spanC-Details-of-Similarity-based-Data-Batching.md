# <span id="page-15-1"></span>C Details of Similarity-based Data Batching

We adapt the pipeline of in-context pre-training (Shi et al., 2024) in our approach. Given a set of documents  $\mathcal{D}$ , for each document  $d \in \mathcal{D}$ , we first use Contriever (Izacard et al., 2022) to retrieve top-k most similar documents N(d). The similarity between the document  $d_i$  and  $d_j$  is defined as the cosine similarity of their Contriever embeddings, i.e.,  $\sin(d_i, d_j) = \cos(C(d_i), C(d_j))$ , where C denotes the Contriever encoder model. We implement an efficient approximate nearest-neighbors search based on the FAISS library (Johnson et al., 2019). Then, we sort all the documents according to the similarity and construct training instances by batch consecutive documents. We use the same greedy algorithm as Shi et al. (2024). We start from a single document and repeatedly add the document that has the highest similarity value and has not been added to the list; we restart the process with a new document if all documents that are connected to the last document of the list are selected. We repeat this process until there are no documents left.

<span id="page-15-0"></span>

| Model                                   | $n_{\mathrm{params}}$          | N  | D    | $n_{\rm head}$ |
|-----------------------------------------|--------------------------------|----|------|----------------|
| 0.3B<br>0.3B/8E<br>0.3B/16E<br>0.3B/32E | 0.3B<br>1.8B<br>3.5B<br>6.8B   | 24 | 1024 | 16             |
| 1.5B<br>1.5B/8E<br>1.5B/16E<br>1.5B/32E | 1.5B<br>7.8B<br>15.0B<br>29.5B | 48 | 1536 | 24             |

**Table 4:** Model architectures and sizes used in our experiments. For MoE models, we replace each FFN layers with a MoE layer. kE (e.g., "16E" in "0.3B/16E") represents the architecture in which each FFN layer is replaced with a MoE layer of k experts. N: number of layers; D: hidden dimension of the model;  $n_{\text{head}}$ : number of attention heads.

