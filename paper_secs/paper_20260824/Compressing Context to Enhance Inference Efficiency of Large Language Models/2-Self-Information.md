# 2 Self-Information

Self-information, also known as *surprisal* or *information content*, is a fundamental concept in information theory that quantifies the amount of information conveyed by an event given a distribution [\(Shannon,](#page-8-6) [1948\)](#page-8-6). In the context of language modelling, the event can be regarded as one step of generation (i.e., a token) and the distribution

corresponds to its output distribution. So the selfinformation of a token can be defined as the negative log likelihood:

$$I(x) = -\log_2 P(x_t|x_0, x_1, ..., x_{t-1})$$
 (1)

where I(x) represents the self-information of token x and P(x) denotes its output probability.

In information theory, self-information measures the level of surprise or uncertainty associated with an event; rare events convey more information and thus have higher self-information, while common events convey less information and have lower self-information. In the context of language modelling, self-information can be used to assess the informativeness of lexical units, e.g., words, phrases, or sentences. Lexical units with lower self-information are less informative and thus are more likely to be inferred from the context. As a result, we may treat these parts of input as redundant during LLM inference.

In NLP, self-information has been used to measure surprise in creative language artefacts [\(Bunescu and Uduehi,](#page-8-7) [2022\)](#page-8-7). In addition, related concepts of self-information such as entropy and perplexity are widely used in language model optimisation and evaluation.

$$H(S) = \frac{1}{N} \Sigma_t I(x_t) \tag{2}$$

$$PP(S) = 2^{H(S)} \tag{3}$$

where the entropy H(S) of the sentence S = (x0, ..., xn) is the average self-information of words in the sentence, and perplexity P P(S) of the sentence can be calculated with entropy. The property of self-information that is especially relevant to our method is the additivity.

$$I(x_0, x_1) = -\log_2 P(x_0, x_1)$$

$$= -\log_2 P(x_0)P(x_1|x_0)$$

$$= -\log_2 P(x_0) - \log_2 P(x_1|x_0)$$

$$= I(x_0)I(x_1)$$
(4)

This means we can calculate the self-information of a lexical unit by simply summing the selfinformation of the tokens in it.

