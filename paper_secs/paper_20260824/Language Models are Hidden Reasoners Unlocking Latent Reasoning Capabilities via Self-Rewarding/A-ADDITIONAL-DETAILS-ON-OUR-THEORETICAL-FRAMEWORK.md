# A ADDITIONAL DETAILS ON OUR THEORETICAL FRAMEWORK

#### <span id="page-15-0"></span>A.1 PROOF OF PROPOSITION [2](#page-4-2)

*Proof.* We restate the objective as follows:

$$J(\theta) := \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y}) \sim \mathcal{D}_{Gold}} \left[ \mathbb{E}_{\boldsymbol{z} \sim \pi_{\theta}(\cdot|\boldsymbol{x})} \left[ \log \pi_{\theta}(\boldsymbol{y}|\boldsymbol{x} \oplus \boldsymbol{z}) \right] - \beta D_{KL} [\pi_{\theta}(\boldsymbol{z}|\boldsymbol{x}) || \pi_{0}(\boldsymbol{z}|\boldsymbol{x})] \right],$$

$$= \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y}) \sim \mathcal{D}_{Gold}} \left[ \mathbb{E}_{\pi_{\theta}(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \pi_{\theta}(\boldsymbol{y}|\boldsymbol{x} \oplus \boldsymbol{z}) - \beta \log \pi_{\theta}(\boldsymbol{z}|\boldsymbol{x}) + \beta \log \pi_{0}(\boldsymbol{z}|\boldsymbol{x}) \right] \right],$$

where β > 0 is a positive coefficient to control the regularization strength. We take the gradient *w.r.t* θ at each sample pair (x,y), and we get

$$\nabla_{\theta} J(\theta; \boldsymbol{x}, \boldsymbol{y}) := \nabla_{\theta} \int (\pi_{\theta}(\boldsymbol{z}|\boldsymbol{x})) (\log \pi_{\theta}(\boldsymbol{y}|\boldsymbol{x} \oplus \boldsymbol{z}) - \beta \log \pi_{\theta}(\boldsymbol{z}|\boldsymbol{x}) + \beta \log \pi_{0}(\boldsymbol{z}|\boldsymbol{x})) d\boldsymbol{z}$$

$$= \mathbb{E}_{\pi_{\theta}(\boldsymbol{z}|\boldsymbol{x})} \left[ \nabla_{\theta} \log \pi_{\theta}(\boldsymbol{z}|\boldsymbol{x}) \left( \log \pi_{\theta}(\boldsymbol{y}|\boldsymbol{x} \oplus \boldsymbol{z}) - \beta \log \frac{\pi_{\theta}(\boldsymbol{z}|\boldsymbol{x})}{\pi_{0}(\boldsymbol{z}|\boldsymbol{x})} \right) \right]$$

$$+ \mathbb{E}_{\pi_{\theta}(\boldsymbol{z}|\boldsymbol{x})} \left[ \nabla_{\theta} \log \pi_{\theta}(\boldsymbol{y}|\boldsymbol{x} \oplus \boldsymbol{z}) - \beta \nabla_{\theta} \log \pi_{\theta}(\boldsymbol{z}|\boldsymbol{x}) \right].$$

We further define r(z) := log πθ(y|x ⊕ z) − β log <sup>π</sup>θ(z|x) π0(z|x) , and use the fact that Eπθ(z|x) [∇<sup>θ</sup> log πθ(z|x)] = R πθ(z|x) ∇θπθ(z|x) πθ(z|x) dz = ∇<sup>θ</sup> R πθ(z|x)dz = 0. we obtain the final gradient as

$$\nabla_{\theta} J(\theta; \bm{x}, \bm{y}) = \mathbb{E}_{\pi(\bm{z}|\bm{x})} \left[ \nabla_{\theta} \log \pi_{\theta}(\bm{z}|\bm{x}) \cdot r(\bm{z}) + \nabla_{\theta} \log \pi_{\theta}(\bm{y}|\bm{x} \oplus \bm{z}) \right] \,.$$

And when we use RLOO estimator with empirical samples, we can replace above gradient estimation with empirical samples, which gives us the following result:

$$\nabla_{\theta} \widehat{J}(\theta) := \frac{1}{NK} \sum_{i=1}^{N} \sum_{k=1}^{K} \left( \nabla_{\theta} \log \pi_{\theta}(\boldsymbol{z}_{k}^{(i)} \mid \boldsymbol{x}_{i}) \cdot A_{k}^{(i)} + \nabla_{\theta} \log \pi_{\theta}(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{i} \oplus \boldsymbol{z}_{k}^{(i)}) \right),$$
with  $A_{k}^{(i)} = r(\boldsymbol{z}_{k}^{(i)}) - \frac{1}{K-1} \sum_{j\neq k}^{K} r(\boldsymbol{z}_{j}^{(i)}), r(\boldsymbol{z}_{k}^{(i)}) := \log \pi_{\theta}(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{i} \oplus \boldsymbol{z}_{k}^{(i)}) - \beta \log \frac{\pi_{\theta}(\boldsymbol{z}_{k}^{(i)} \mid \boldsymbol{x}_{i})}{\pi_{0}(\boldsymbol{z}_{k}^{(i)} \mid \boldsymbol{x}_{i})},$ 

which completes the proof.

### <span id="page-15-1"></span>B PROMPT TEMPLATES

In this section, we list all prompt templates used in our experiments. For the GSM8K dataset, we used a minimum prompt template. For the ARC-Challenge dataset, we use a CoT template for the base model and LaTRO fine-tuning, and another template direct answer generation in SFT.

{question} Let's think step by step.

Figure 5: CoT template for GSM8K

Question: {question} Options: {choices["text"]} Let's think step by step first then print your answer like 'The answer is (one of the options)'.""",

Figure 6: CoT template for ARC-Challenge

The output MUST strictly follow the structure and format described below:

- 1. The answer should be strictly one of the options given in the question
- 2. Do not include any text after you choose an option.

\*\*Example:\*\*

Question: (question here)

Options: [option1, option2, option3] The answer is: (one of the options).

Question: {question}

Options: {choices["text"]}

Figure 7: Direct answer generation template for ARC-Challenge

### <span id="page-16-0"></span>C SAMPLE RESPONSES

Question: Kylar went to the store to buy glasses for his new apartment. One glass costs \$5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them? Let's think step by step.

#### Response (Base model):

Step 1: Determine the cost of one regular glass and one discounted glass. One regular glass costs \$5. Every second glass costs 60% of the price of a regular glass, so: 60% of 5 = 0.60∗5 = \$3

Step 2: Calculate the total number of regular and discounted glasses. Kylar wants to buy 16 glasses in total. Since every second glass is discounted, we can divide the total number of glasses by 2 to find the number of discounted glasses: 16 glasses / 2 = 8 discounted glasses Now, subtract the number of discounted glasses from the total number of glasses to find the number of regular glasses: 16 glasses - 8 discounted glasses = 8 regular glasses Step (did not finish)

Response (LaTRO200): Let's break it down:

- 1. Kylar needs to buy 16 glasses.
- 2. Every second glass costs 60% of the original price (\$5), which is \$3.
- 3. So, for every pair of glasses (one full price and one discounted), Kylar pays \$5 + \$3 = \$8.
- 4. Since he needs 16 glasses, he buys 8 pairs. 5. Therefore, Kylar needs to pay 8 pairs \* \$8 per pair = \$64. So, Kylar needs to pay \$64 for the 16 glasses. The answer is 64.

Figure 8: Sample responses of a GSM8K question, from Phi-3.5 models, maximum generation length L = 200. The base model does not finish the generation, while the LaTRO model generates a short and correct answer.

Question: Water, ice, and steam all have different temperatures. What is the order from coldest to hottest?

Options: ['ice, water, steam', 'ice, steam, water','steam, ice, water','steam, water, ice'] Response (Base model): To determine the order from coldest to hottest, let's evaluate each option.

- Ice: It is the solid form of water and is typically around 0°C (32°F).
- Water: It is the liquid form of water and is usually around 100°C (212°F) or 0°C (32°F) at standard pressure and temperature conditions, respectively.
- Steam: It is the gaseous form of water and is typically at a temperature equal to or greater than 100°C (212°F) at standard pressure conditions, depending on the pressure and other factors.

Option 1: 'ice, water, steam' - This order seems incorrect since steam is typically hotter than water at standard conditions.

Option 2: 'ice, steam, water' - This order is incorrect because ice is the coldest, but the relative order of water and steam is opposite of what we know.

Option 3:'steam, ice, water' - This order is incorrect since ice is colder than steam.

Option 4:'steam, water, ice' - This order seems the most plausible, considering that steam is hotter than water and water is hotter than ice.

The answer is 'steam, water, ice'.

Response (SFT): water, ice, steam

Response (LaTRO): To determine the order from coldest to hottest, let's analyze each option:

- 1. Ice has a solid form and a fixed temperature, which is typically around 0°C (32°F) at standard pressure.
- 2. Water is liquid and exists at a temperature of 0°C (32°F) at standard pressure, melting point of ice.
- 3. Steam is the gaseous state of water and is typically around 100°C (212°F) at standard pressure.

Based on this information, we can now order them from coldest to hottest:

First, we have ice, as it is the coldest. Then, we have water at 0°C, which is the middle temperature. Lastly, we have steam, which is the hottest.

So, the order from coldest to hottest is 'ice, water, steam'.

The answer is ['ice, water, steam'].

Figure 9: Sample responses of an ARC-Challenge question, from Llama-3.1 models, the errors are highlighted in red. The base model shows knowledge about basic physics but makes a logical error on the order.