# <span id="page-22-0"></span>Chain -of -Rubrics (CoR) Roll-out for Instruct Models (no rubrics)

Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.

You should choose the chatbot that follows the client's instructions and answers the client's question better. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the chatbots. Be as objective as possible. First, compare the chatbot responses and provide your evaluations. Then, conclude with your verdict using exactly this format: <answer>[[A]]</answer> if Chatbot A is better, <answer>[[B]</answer> if Chatbot B is better.

Figure 8: The system prompt of the ablation study on cold start RL without any rubrics.

Table 9: Complete questions and chatbot responses.

<span id="page-23-0"></span>Question: symptoms of sickle-cell disease

#### Chatbot A:

- 1. Painful red or yellow skin lesions
- 2. Frequent infections3. Fatigue and weakness
- 4. Anemia
- 5. Slower growth and development
- 6. Jaundice
- 7. Swelling of hands and feet
- 8. Vision loss
- 9. Tiredness and weakness after physical activity
- 10. Breathlessness
- 11. Fever, chills, and night sweats.

#### Chatbot B

Sickle cell disease is a genetic disorder that affects the shape of red blood cells. It is most common in African Americans, but can also affect people from other ethnic groups. The symptoms of sickle cell disease can vary widely from person to person, but some common symptoms include:

- 1. Fatigue: People with sickle cell disease often experience fatigue and weakness, even when they have not been
- 2. Pain: Sickle cell disease can cause pain in the body, particularly in the arms, legs, back, and abdomen. Pain can be mild or severe and can last for days or weeks.
- 3. Fever: People with sickle cell disease may be prone to infections and may experience frequent fevers.
- 4. Anemia: Sickle cell disease can cause anemia, a condition in which the body does not have enough red blood cells to carry oxygen throughout the body.
- 5. Delayed growth and development: Children with sickle cell disease may be smaller than their peers due to delayed growth and development.
- 6. Infections: People with sickle cell disease are at increased risk for infections, particularly in the bloodstream and urinary tract.
- 7. Pigmentation: Sickle cell disease can cause dark areas of pigmentation in the body, particularly on the hands

It's important to note that not everyone with sickle cell disease will experience all of these symptoms, and some people may not experience any symptoms at all. Therefore, it's important to consult with a healthcare provider if you suspect you or someone you know may have sickle cell disease.

<span id="page-24-0"></span>Table 10: Cold-Start RL Judges. The rubrics and evaluation mainly center around superficial shallow features (e.g., number of points listed) without actually taking their accuracy into account.

#### Cold-start RL evaluation

<type>Chat</type>

<rubric>

The evaluation criteria for the responses are as follows:

- <item>Relevance to the question</item> (40%): The response should directly address the symptoms of sickle
- <item>Comprehensiveness</item> (30%): The response should include a wide range of symptoms commonly associated with sickle cell disease.
- <item>Clarity and organization</item> (30%): The response should be easy to read and understand, with symptoms listed in a clear and organized manner.

<justify>

The question specifically asks for the symptoms of sickle cell disease, so the responses should be directly relevant to this topic. Comprehensiveness is important because sickle cell disease can have a variety of symptoms, and a comprehensive list will provide a more complete picture of the condition. Clarity and organization are important because the symptoms can be complex, and a well-organized response will make it easier for the client to understand the information.

</justify>

<eval> <quote\_A>

- 1. Painful red or yellow skin lesions
- 2. Frequent infections
- 3. Fatigue and weakness
- 4. Anemia
- 5. Slower growth and development
- 6. Jaundice
- 7. Swelling of hands and feet
- 8. Vision loss
- 9. Tiredness and weakness after physical activity
- 10. Breathlessness
- 11. Fever, chills, and night sweats.

</quote\_A>

<summary A>

Chatbot A provided a list of symptoms that are directly relevant to sickle cell disease. The response is comprehensive and includes a wide range of symptoms commonly associated with the condition. The symptoms are listed in a clear and organized manner, making it easy for the client to understand the information.

</summary A>

<quote\_B>

Sickle cell disease is a genetic disorder that affects the shape of red blood cells. It is most common in African Americans, but can also affect people from other ethnic groups. The symptoms of sickle cell disease can vary widely from person to person, but some common symptoms include:

- 1. Fatigue: People with sickle cell disease often experience fatigue and weakness, even when they have not been physically active.
- 2. Pain: Sickle cell disease can cause pain in the body, particularly in the arms, legs, back, and abdomen. Pain can be mild or severe and can last for days or weeks.
- 3. Fever: People with sickle cell disease may be prone to infections and may experience frequent fevers.
- 4. Anemia: Sickle cell disease can cause anemia, a condition in which the body does not have enough red blood cells to carry oxygen throughout the body.
- 5. Delayed growth and development: Children with sickle cell disease may be smaller than their peers due to delayed growth and development.
- 6. Infections: People with sickle cell disease are at increased risk for infections, particularly in the bloodstream and urinary tract.
- 7. Pigmentation: Sickle cell disease can cause dark areas of pigmentation in the body, particularly on the hands and feet.

</quote\_B>

Chatbot B provided a detailed explanation of sickle cell disease and a list of symptoms. The response is relevant to the question and includes a wide range of symptoms commonly associated with the condition. However, the response is less organized than Chatbot A's response, and the explanation of sickle cell disease may be unnecessary for the client's question.

</summary\_B>

<answer>[[A]]</answer>

<span id="page-25-0"></span>Table 11: RM-R1 RL Judges. The rubrics and evaluation are well-designed and reasonable, assessing the effectiveness of the model response.

#### Warm-start RL (RM-R1) evaluation

<type>Chat</type>

<rubric>

- 1. Accuracy of information (40 %): How correctly does the response describe the symptoms of sickle-cell disease?
- 2. Comprehensiveness (30 %): How thoroughly does the response cover the range of symptoms associated with sickle-cell disease?
- 3. Clarity and organization (20 %): How well-structured and easy to understand is the information presented?
- 4. Helpfulness for the user (10 %): How useful is the information for someone seeking to understand sickle-cell disease symptoms?

<justify>

These criteria focus on evaluating how well each response addresses the user's question about sickle-cell disease symptoms. Accuracy is weighted highest because providing correct information about medical conditions is crucial. Comprehensiveness is important, as sickle-cell disease has multiple symptoms that vary in severity and presentation. Clarity and organization affect how easily the user can understand the information. Helpfulness considers how practical and actionable the information is for the user.

</rubric>

</rubric: <eval>

#### Comparing the two responses:

Accuracy of information:

<quote\_A>1. Painful red or yellow skin lesions

- 2. Frequent infections
- 3. Fatigue and weakness
- 4. Anemia
- 5. Slower growth and development
- 6. Jaundice
- 7. Swelling of hands and feet
- 8. Vision loss
- 9. Tiredness and weakness after physical activity
- 10. Breathlessness
- 11. Fever, chills, and night sweats.</quote\_A>

<quote\_B>1. Fatigue: People with sickle-cell disease often experience fatigue and weakness, even when they have not been physically active.

- 2. Pain: Sickle-cell disease can cause pain in the body, particularly in the arms, legs, back, and abdomen. Pain can be mild or severe and can last for days or weeks.
- 3. Fever: People with sickle-cell disease may be prone to infections and may experience frequent fevers.
- 4. Anemia: Sickle-cell disease can cause anemia, a condition in which the body does not have enough red blood cells to carry oxygen throughout the body.
- 5. Delayed growth and development: Children with sickle-cell disease may be smaller than their peers due to delayed growth and development.
- 6. Infections: People with sickle-cell disease are at increased risk for infections, particularly in the bloodstream and urinary tract.
- 7. Pigmentation: Sickle-cell disease can cause dark areas of pigmentation in the body, particularly on the hands and feet.

Chatbot A provides a list of symptoms without explanation, but some listed items (e.g. "painful red or yellow skin lesions," "vision loss") are not typical of sickle-cell disease. Chatbot B offers more accurate information, explaining each symptom's nature and severity.

Comprehensiveness:

Chatbot A lists 11 symptoms, several of which are inaccurate or uncommon. Chatbot B lists 7 symptoms that are more representative and supplies contextual detail about variability and medical advice. *Clarity and organization:* 

Chatbot A's plain numbered list lacks context, making severity unclear. Chatbot B structures the list with explanations, improving readability and understanding.

Chatbot A is less helpful due to inaccuracies and missing context. Chatbot B is more useful, providing accurate descriptions, context, and guidance.

</eval>

<answer>[[B]]</answer>

## <span id="page-26-0"></span>K ANALYSIS FOR THE IMPORTANCE OF RL

<span id="page-26-1"></span>Lemma 1 (High-reward filtering induces εtrain < δ). *Let* X *be a finite input (or trajectory) space and let* µ *be a probability distribution on* X *(the underlying environment distribution).*

*Let* ϕrob, ϕtriv : X → {0, 1} *be a robust and a trivial feature, and define the* disagreement event

$$D := \{x \in \mathcal{X} : \phi_{\text{rob}}(x) \neq \phi_{\text{triv}}(x)\}.$$

*Let* R : X → R *be a reward (or quality) function such that larger* R(x) *indicates better performance. Fix a threshold* τ ∈ R *and define the* high-reward *and* low-reward *events*

$$H := \{x : R(x) \ge \tau\}, \qquad L := \{x : R(x) < \tau\}.$$

*Assume:*

*(i) The SFT training distribution is the environment distribution conditioned on high reward:*

$$p_{\text{train}}(\cdot) := \mu(\cdot \mid H), \qquad p_{\text{env}}(\cdot) := \mu(\cdot).$$

*(ii) The filter is nontrivial:*

$$0 < \mu(H) < 1$$
, equivalently  $0 < \mu(L) < 1$ .

*(iii) Disagreement is more prevalent among low-reward trajectories than among high-reward ones:*

$$\Pr_{\mu}[D \mid L] > \Pr_{\mu}[D \mid H].$$

*Define*

$$\varepsilon_{\mathrm{train}} \; \coloneqq \; \Pr_{x \sim p_{\mathrm{train}}}[D] = \Pr_{\mu}[D \mid H], \qquad \delta \; \coloneqq \; \Pr_{x \sim p_{\mathrm{env}}}[D] = \Pr_{\mu}[D].$$

*Then* εtrain < δ*.*

Remark 1 (Connection to SFT distillation and RL exploration). *Lemma [1](#page-26-1) formalizes a realistic pipeline:*

- *The SFT (or distillation) dataset is collected by sampling trajectories from some base policy or environment, and then* filtering *to keep only those with sufficiently high reward (e.g., judged as "good" by humans or a reward model). This corresponds to* ptrain = µ(· | H)*.*
- *RL, in contrast, optimizes expected reward under the environment distribution* penv = µ*, and through on-policy rollouts can visit both high- and low-reward regions* H *and* L*.*

*If disagreement between trivial and robust features tends to* hurt *reward (i.e., is more common in* L *than in* H*), then assumption* (iii) *is natural. The lemma then shows that the disagreement probability* εtrain *seen in the filtered SFT data is strictly smaller than the true environment disagreement* δ*.*

*In practice, the gap can be amplified further by sampling effects. Let* N *be the number of (independent) trajectories in the SFT dataset drawn from* ptrain *and* M *the number of trajectories seen during RL rollouts drawn from* penv*. If* Pr<sup>p</sup>train [D] = εtrain *is very small, the probability that the SFT dataset contains* no *disagreement example is*

$$\Pr[no \ D \ in \ SFT \ data] = (1 - \varepsilon_{\text{train}})^N,$$

*which can be close to* 1 *for moderate* N*. In contrast, the probability that RL ever encounters a disagreement trajectory is*

$$\Pr[\text{at least one } D \text{ in RL rollouts}] = 1 - (1 - \delta)^M,$$

<span id="page-26-2"></span>*which quickly approaches* 1 *when* M *is large and* δ > 0*. Since the number of possible trajectories typically grows exponentially with horizon, a finite distilled SFT dataset can easily miss rare but systematically harmful disagreement patterns, while RL exploration eventually uncovers them. This provides a concrete justification for the core assumption* εtrain < δ *that we use in the next proposition.* Proposition 1 (Approximate shortcut agreement on SFT vs. RL). *Let* X *be a finite input space and let* ptrain *and* penv *be two probability distributions on* X *.*

*Let*

$$\phi_{\rm rob}, \phi_{\rm triv}: \mathcal{X} \to \{0, 1\}$$

*denote a* robust *and a* trivial *feature, respectively, and define the ground-truth label as*

$$y(x) \coloneqq \phi_{\text{rob}}(x), \qquad x \in \mathcal{X}.$$

*Define the disagreement probabilities*

$$\varepsilon_{\text{train}} \; \coloneqq \; \Pr_{x \sim p_{\text{train}}} \left[ \phi_{\text{triv}}(x) \neq \phi_{\text{rob}}(x) \right], \qquad \delta \; \coloneqq \; \Pr_{x \sim p_{\text{env}}} \left[ \phi_{\text{triv}}(x) \neq \phi_{\text{rob}}(x) \right].$$

*Assume the gap*

<span id="page-27-0"></span>
$$0 \le \varepsilon_{\text{train}} < \delta \le 1. \tag{12}$$

*(For example, this is guaranteed if* ptrain *and* penv *arise via high-reward filtering as in Lemma [1.](#page-26-1)) Consider deterministic policies* π : X → {0, 1} *and define:*

• *The SFT (imitation) objective*

$$\mathcal{L}_{SFT}(\pi) := \mathbb{E}_{x \sim p_{train}} [\mathbf{1} \{ \pi(x) \neq y(x) \}].$$

• *The RL objective (expected reward in the environment) with reward* r(x, a) = 1{a = y(x)}*:*

$$J_{\mathrm{RL}}(\pi) := \mathbb{E}_{x \sim p_{\mathrm{env}}} [r(x, \pi(x))] = \mathbb{E}_{x \sim p_{\mathrm{env}}} [\mathbf{1} \{\pi(x) = y(x)\}].$$

*Define the* trivial *and* robust *policies by*

$$\pi_{\text{triv}}(x) \coloneqq \phi_{\text{triv}}(x), \qquad \pi_{\text{rob}}(x) \coloneqq \phi_{\text{rob}}(x) = y(x).$$

*Then:*

*(a) The SFT objective distinguishes the two policies only up to the small margin* εtrain*:*

$$\mathcal{L}_{SFT}(\pi_{rob}) = 0, \qquad \mathcal{L}_{SFT}(\pi_{triv}) = \varepsilon_{train}.$$

*(b) The RL objective distinguishes them by the larger gap* δ*:*

$$J_{\rm RL}(\pi_{\rm rob}) = 1, \qquad J_{\rm RL}(\pi_{\rm triv}) = 1 - \delta.$$

*In particular, the improvement in SFT loss when going from* πtriv *to* πrob *is* εtrain*, whereas the improvement in RL reward is* δ > εtrain*.*

*(c) For any policy* π*, we have*

$$J_{\rm RL}(\pi) \leq 1 = J_{\rm RL}(\pi_{\rm rob}),$$

*with equality if and only if* π(x) = y(x) *for* penv*-almost every* x*. Thus* πrob *is the unique optimal policy for the RL objective (up to* penv*-null sets), while* πtriv *is strictly suboptimal whenever* δ > 0*.*

Remark 2 (Interpretation for SFT vs. RL and distillation). *In the regime where the SFT teacher or data-generating process is strong but imperfect, it is natural to have* εtrain *very small (rare disagreements between trivial and robust features on the observed SFT data), while the environment disagreement* δ *is larger, because harder or more diverse test prompts break the trivial shortcut more frequently.*

*One concrete interpretation is:*

• *The* trivial *feature* ϕtriv *encodes the behavior of a teacher policy* π<sup>T</sup> *, obtained for example by supervised finetuning or heuristics, with*

$$\pi_T(x) := \phi_{\text{triv}}(x), \qquad \Pr_{x \sim p_{\text{env}}}[\pi_T(x) = y(x)] = 1 - \delta \quad \text{with } 0 < \delta \ll 1.$$

• *The SFT dataset is produced by distilling this teacher: sample* x ∼ ptrain *and label it with* π<sup>T</sup> (x)*. Under high-reward filtering (Lemma [1\)](#page-26-1), the teacher almost never disagrees with* y(x) *on* ptrain*, so* εtrain *is small.*

*Proposition [1](#page-26-2) then says:*

- *From the SFT objective's perspective, moving from* πtriv *(the shortcut / teacher policy) to* πrob *only improves the loss by* εtrain ≪ 1*, which can be negligible relative to optimization noise, model capacity, or inductive bias. This makes it easy for SFT to "overfit" to the trivial pattern.*
- *From the RL objective's perspective, the same move improves the reward by* δ*, which can be substantially larger. Thus RL has a much stronger structural incentive to abandon the trivial shortcut and adopt the robust strategy* πrob*, which is in fact the unique policy that maximizes expected reward.*

*The exact case* εtrain = 0 *corresponds to an idealized limit in which trivial and robust features agree on all SFT training examples; the general case* εtrain < δ *is what one expects in realistic pipelines with high-reward filtering and broader RL exploration.*

## <span id="page-28-0"></span>K.1 PROOF OF LEMMA [1](#page-26-1)

*Proof.* Write

$$\alpha := \mu(H) \in (0,1), \qquad 1 - \alpha = \mu(L).$$

By the law of total probability,

$$\begin{split} \delta &= \Pr_{\mu}[D] \\ &= \Pr_{\mu}[D \mid H] \, \mu(H) \, + \, \Pr_{\mu}[D \mid L] \, \mu(L) \\ &= \varepsilon_{\mathrm{train}} \, \alpha \, + \, \Pr_{\mu}[D \mid L] \, (1 - \alpha). \end{split}$$

We want to compare δ with εtrain. Subtracting, we obtain

$$\begin{split} \delta - \varepsilon_{\text{train}} &= \varepsilon_{\text{train}} \, \alpha \; + \; \Pr_{\mu}[D \mid L] \, (1 - \alpha) \; - \; \varepsilon_{\text{train}} \\ &= (1 - \alpha) \, \Pr_{\mu}[D \mid L] \; - \; (1 - \alpha) \, \varepsilon_{\text{train}} \\ &= (1 - \alpha) \, \left( \Pr_{\mu}[D \mid L] - \varepsilon_{\text{train}} \right). \end{split}$$

By assumption, 0 < 1 − α < 1 and

$$\Pr_{\mu}[D \mid L] > \Pr_{\mu}[D \mid H] = \varepsilon_{\text{train}},$$

so the product (1 − α) Prµ[D | L] − εtrain is strictly positive. Therefore

$$\delta - \varepsilon_{\rm train} > 0 \quad \Longrightarrow \quad \varepsilon_{\rm train} < \delta,$$

as claimed.

## <span id="page-28-1"></span>K.2 PROOF OF PROPOSITION [1](#page-26-2)

*Proof.* We prove (a), (b), and (c) in turn.

(a) SFT objective. By definition of y and πrob, we have πrob(x) = y(x) for all x ∈ X , so

$$\mathbf{1}\{\pi_{\text{rob}}(x) \neq y(x)\} = 0$$
 for all  $x$ .

Therefore

$$\mathcal{L}_{SFT}(\pi_{rob}) = \mathbb{E}_{x \sim p_{train}} [\mathbf{1}\{\pi_{rob}(x) \neq y(x)\}] = 0.$$

For πtriv, observe that

$$\pi_{\text{triv}}(x) \neq y(x) \iff \phi_{\text{triv}}(x) \neq \phi_{\text{rob}}(x),$$

so

$$\mathcal{L}_{\mathrm{SFT}}(\pi_{\mathrm{triv}}) = \mathbb{E}_{x \sim p_{\mathrm{train}}} \big[ \mathbf{1}\{\phi_{\mathrm{triv}}(x) \neq \phi_{\mathrm{rob}}(x)\} \big] = \Pr_{x \sim p_{\mathrm{train}}} \big[ \phi_{\mathrm{triv}}(x) \neq \phi_{\mathrm{rob}}(x) \big] = \varepsilon_{\mathrm{train}}.$$

This proves (a).

(b) RL objective. We similarly have

$$1{\pi_{\text{rob}}(x) = y(x)} = 1$$
 for all  $x$ ,

so

$$J_{\mathrm{RL}}(\pi_{\mathrm{rob}}) = \mathbb{E}_{x \sim p_{\mathrm{env}}} \left[ \mathbf{1} \{ \pi_{\mathrm{rob}}(x) = y(x) \} \right] = 1.$$

For πtriv, we use

$$\pi_{\text{triv}}(x) = y(x) \iff \phi_{\text{triv}}(x) = \phi_{\text{rob}}(x),$$

to obtain

$$J_{\rm RL}(\pi_{\rm triv}) = \mathbb{E}_{x \sim p_{\rm env}} \left[ \mathbf{1} \{ \phi_{\rm triv}(x) = \phi_{\rm rob}(x) \} \right]$$
$$= 1 - \Pr_{x \sim p_{\rm env}} \left[ \phi_{\rm triv}(x) \neq \phi_{\rm rob}(x) \right] = 1 - \delta.$$

This proves (b).

(c) Optimality of the robust policy for RL. Let π be any deterministic policy. For each x ∈ X ,

$$\mathbf{1}\{\pi(x) = y(x)\} \le 1,$$

with equality if and only if π(x) = y(x). Taking expectations under penv yields

$$J_{\rm RL}(\pi) = \mathbb{E}_{x \sim p_{\rm env}} [\mathbf{1} \{ \pi(x) = y(x) \}] \le \mathbb{E}_{x \sim p_{\rm env}} [1] = 1 = J_{\rm RL}(\pi_{\rm rob}).$$

Hence JRL(π) ≤ JRL(πrob) for all π, so πrob is an optimal policy.

Moreover, if JRL(π) = 1, then the above inequality must hold with equality, which is only possible if 1{π(x) = y(x)} = 1 for penv-almost every x. Equivalently,

$$\pi(x) = y(x)$$
 for  $p_{\text{env}}$ -almost every  $x$ .

Thus any maximizer coincides with πrob almost surely under penv, and πrob is the unique optimal policy up to penv-null sets.

Finally, since δ > 0 by equation [12,](#page-27-0) we have

$$J_{\rm RL}(\pi_{\rm triv}) = 1 - \delta < 1 = J_{\rm RL}(\pi_{\rm rob}),$$

so πtriv is strictly suboptimal. This proves (c) and completes the proof.