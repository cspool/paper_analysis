# <span id="page-6-1"></span>3.4. Thought-Specific Positional Embedding

Merging multiple reasoning paths poses challenges due to positional ambiguity. LLMs distinguish tokens based on their content and positional encoding. When multiple reasoning paths are generated in parallel, tokens at the same relative position (e.g., the t-th token in each  $r^{(i)}$ ) share identical positional encodings. This causes confusion during summarization, as the model cannot differentiate which reasoning stream a token originated from.

**Flattened Encoding:** A naive solution assigns unique absolute positions across all paths:  $m = l_x + i \cdot l_{\text{max}} + t$ , where  $l_x$  is the input length, i indexes the reasoning path, and t indexes the token position within that path. While this resolves positional collisions, it results in large positional indices as P increases. Typical positional encoding mechanisms such as Rotary Position Embedding (RoPE) (Su et al., 2024) encodes relative positions via rotations, and large index differences |m-n| cause attention scores to decay. As a result, tokens from earlier paths (*i.e.*, lower i of  $r^{(i)}$ ) contribute less when generating the final answer, introducing imbalance across paths.

**Sequence-Aware Positional Embedding:** To address positional ambiguity in multi-response generation tasks, ParaThinker separates different reasoning paths by augmenting the RoPE mechanism with learnable thought embeddings  $\{T^{(j)}\}_{j=0}^{p}$ . Specifically, we add the  $T^{(j)}$  to the key and value embeddings of all tokens within the i-th reasoning path, which distinguishes each reasoning path at the summarizing phase. The thought embedding is added to the key before the RoPE rotation is applied. Let  $\tilde{k}_{t}^{(j)}$ ,  $\tilde{v}_{t}^{(j)}$  denote the cached key and value for token t at path j, respectively, from which the key and value vectors are formed as:

$$\tilde{k}_t^{(j)} = R_t(k_t^{(j)} + T^{(j)}) \tag{4}$$

<span id="page-6-0"></span>
$$\tilde{v}_t^{(j)} = v_t^{(j)} + T^{(j)} \tag{5}$$

Here,  $l_{max}$  denotes the maximum token number for each reasoning path, and  $R_t$  is the corresponding RoPE rotation matrix. The dot product attention score between a query  $q_n$  from the summary (at local position n) and a key  $\tilde{k}_t^{(j)}$  from path j (at position m) is:

$$score(n, m) = (R_n q_n^{(i)})^T \tilde{k}_m^{(j)} = (R_n q_n^{(i)})^T [R_m (k_m^{(j)} + T^{(j)})]$$
(6)

Using the RoPE property  $(R_n)^T R_m = R_{m-n}$ , Eq. 6 can be simplified into two distinct components:

$$score(n, m) = \underbrace{q_n^T R_{m-n} k_m^{(j)}}_{Content-to-Content} + \underbrace{q_n^T R_{m-n} T^{(j)}}_{Content-to-Segment}$$
 (7)

The Content-to-Content term is the standard RoPE attention score, which calculates the relevance between the query's content  $(q_n)$  and the key's content  $(k_m^{(j)})$ . This term is not related to the reasoning path number j and thus does not change when scaling parallel reasoning paths. Content-to-Segment term calculates the relevance between the query's content  $(q_n)$  and the learnable identity of the key's entire reasoning path  $(T^{(j)})$ . This allows the query to directly probe for the origin of the information. Because each reasoning path has a unique, learned thought embedding, this term provides an unambiguous signal for the model to differentiate between parallel streams of text, solving the positional ambiguity.

