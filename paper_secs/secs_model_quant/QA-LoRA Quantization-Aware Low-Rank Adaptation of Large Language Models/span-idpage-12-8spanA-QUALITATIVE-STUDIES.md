# <span id="page-12-8"></span>A QUALITATIVE STUDIES

In this section, we show a few examples of dialog. We compare our method, QA-LoRA, to the direct competitor, QLoRA, under different quantization bit widths. All QLoRA models are postprocessed with GPTQ as described in the main text. We highlight inaccurate answers in blue and totally unacceptable answers in red.

Overall, QLoRA with GPTQ shows unstable behaviors across different cases. For example, in the second case, it crashes in 4-bit and 2-bit quantization but works well in 3-bit quantization. This mainly owes to the uncontrollable quantization loss in post-processing and such loss cannot be amended by any of the subsequent stages.

Instruction: Suggest five ways to budget your expenses.

QLoRA (4-bit): *1. Write down all of your current expenses. 2. Estimate your future expenses. 3. Cut down on non-essential spending. 4. Rate your expenses according to their importance. 5. Categorize your expenses to make it easier to.*

Ours (4-bit): *1. Plan a monthly budget and stick to it. 2. Track your spending. 3. Narrow down your wants and priorities. 4. Cut expenses by shopping around. 5. Eat at home as much as possible.*

———— QLoRA (3-bit): *1. Gather and document your financial information (income, expenses, and assets) 2. Estimate your cash flow (income, expenses, and assets) 3. Allocate your cash flow to the needed/wanted category. 4. Monitor your spending*

Ours (3-bit): *To budget expenses, one should: (1) keep track of spending; (2) have a spending plan; (3) set weekly goals; (4) only use cash; and (5) try to overspend one area to cover another.*

———— QLoRA (2-bit): *the maximum number of times 2026 #03289-012-052112. 03289- 0222500032815-0451 2291-253222.*

Ours (2-bit): *1. Plan a monthly budget and stick to it. 2. Track your spending. 3. Narrow down your wants and priorities. 4. Cut expenses by shopping around. 5. Eat at home as much as possible.*

Instruction: How do you respond to a negative comment on social media?

QLoRA (4-bit): *I'm sorry you feel that way.*

Ours (4-bit): *Negative comments on social media can be hard to respond to, but I would first try to understand where the person is coming from and why they are saying what they are saying. I would then strive to reply in a way that does not further escalate the discussion, but instead responds with a mature.*

———— QLoRA (3-bit): *I would first thank the person for taking the time to share their thoughts. I would then say that I value their opinion and that I am here to listen and learn. I would then go on to explain why I feel the way I do and invite them to an open discussion.*

Ours (3-bit): *I response to a negative comment on social media by thanking the person for their feedback, clarifying if they are basing their comment off of a bad experience, and inviting them to contact me directly so we can work through any issues or concerns they may have.*

———— QLoRA (2-bit): *Demostimeta for the Parts of the Their work of for the parts of Their work of For the parts of3.4.2004 G Mach Rack 2002-2004 G 2.*

Ours (2-bit): *Negative comments on social media can be hard to respond to, but I would first try to understand where the person is coming from and why they are saying what they are saying. I would then strive to reply in a way that does not further escalate the discussion, but instead responds with a mature.*

Instruction: Categorize the following fruits according to color. Input: Apple, Banana, Avocado, Peach.

QLoRA (4-bit): *of the same color: Red (Apple, Banana), Green (Avocado, Peach) Other: Yellow (Peach).*

Ours (4-bit): *Apple - Red. Banana - Yellow. Avocado - Green. Peach - Pink.*

QLoRA (3-bit): *is yellow is green is black is white.*

Ours (3-bit): *- Apple (Red)- Banana (Yellow)- Avocado (Green)- Peach (Pink).*

———— QLoRA (2-bit): *Apple, mash up, and avocado.*

————

Ours (2-bit): *Apple - Red. Banana - Yellow. Avocado - Green. Peach - Pink.*

Instruction: Generate an HTML page that contains the text \"Welcome to my page\".

QLoRA (4-bit): \*n*\*n*<*html*>\*n* <*head*>\*n* <*title*>*Welcome to my page*<*/title*>\*n* <*/head*>\*n* <*body*>\*n* <*h1*>*Welcome to my page*<*/h1*>\*n* <*/body*>\*n*<*/html*>\*n.*

Ours (4-bit): *Welcome to my page.*\*n*<*html*>\*n*<*body*>\*n*<*p*>*Welcome to my page.*<*/p*>\*n*<*/body*>\*n*<*/html*>\*n.* ————

QLoRA (3-bit): <*html*>\*n* <*head*>\*n* <*title*>*My Page*<*/title*>\*n* <*/head*>\*n* <*body*>\*n* <*p*>*Welcome to my page!*<*/p*>\*n* <*/body*>\*n*<*/html*> \*n package org.michael.*

Ours (3-bit): <*HTML*>\*n* <*P*>*Welcome to my page*<*/P*>\*n* <*P*>*Lorem Ipsum is simply placeholder text.*<*/P*>\*n*<*/HTML*>\*n*\*n Note that this response may also contain other elements such as stylesheets and script, if they were defined in the task.*

———— QLoRA (2-bit): *The page has the text* \*"Welcome to my page*\*" as it.*\*n*\*n28. A reint: a 28!313473-020-520. The*\*n313473-020/2*

Ours (2-bit): *Welcome to my page.*\*n*<*html*>\*n*<*body*>\*n*<*p*>*Welcome to my page.*<*/p*>\*n*<*/body*>\*n*<*/html*>\*n.*

**Instruction**: Create a dialogue between two people having a conversation about going to the beach.

**QLoRA** (4-bit): of course, I'm in! what time were you thinking of meeting up? I was thinking of going around noon. sounds good, I'll see you then!

**Ours** (4-bit): .... (beach conversations A and B)\n.... B: Going to the beach sounds like a blast! A: It does, isn't it? B: Yep! A: so what are we doing? B: Planning a full itinerary.

**QLoRA** (3-bit): "Person 1: "I'm so excited to go to the beach! The sun, sand, and water are all so soothing. I can't wait to get there and dive in!" Person 2: "Me too! I can't wait to experience that feeling of being".

**Ours** (3-bit): "So which beach should we go to, Fiji or Bora Bora?"\n, "Fiji has more coral reefs, Bora Bora has more beaches, right?"\n, "Yeah, Bora Bora also has the world's most"

**QLoRA** (2-bit): Such as a user 31 is 4 0-5 Gigs Sector 4 (22.12.2018) - [P] / f / Beyond there is a lot of and.

**Ours** (2-bit): .... (beach conversations A and B)\n.... B: Going to the beach sounds like a blast! A: It does, isn't it? B: Yep! A: so what are we doing? B: Planning a full itinerary.

#### B SIMPLE PROOF

In this section, we give a simple proof to explain that why the proposed approach QA-LoRA can be merged with quantized base models. Some of the letters used in this section are conflicted with that in the main script.

#### **B.1** QUANTIZATION

The pre-trained LLMs weights are quantized by the state-of-the-art PTQ methods (e.g. GPTQ). Each column of weights W is divided with groups with group size g. The de-quantization process is:

$$\tilde{\mathbf{W}} = \mathbf{A} \odot (\hat{\mathbf{W}} - \mathbf{B}) \tag{2}$$

where  ${\bf A}$  is the scaling matrix and  ${\bf B}$  is the zero matrix.  ${\bf A}_{i,j}=\alpha_{\lfloor\frac{i}{q}\rfloor,j}$  and  ${\bf B}_{i,j}=\beta_{\lfloor\frac{i}{q}\rfloor,j}$ .

#### B.2 QA-LORA TRAINING

We introduce an aggregation operation before the input is fed into the LoRA adapters.

$$\mathbf{H}_{I}^{a} = \mathcal{A}(\mathbf{H}_{I}) = [h_{i,k}^{a}] = [\sum_{r=1}^{g} h_{i,(k-1)g+r}], \tag{3}$$

where g is the aggregation window size which equals to the quantization group-size,  $\mathbf{H}_I \in \mathcal{R}^{b \times h}$ , aggregated input  $\mathbf{H}_I^a \in \mathcal{R}^{b \times (h/g)}$ . The latent features in each aggregation window are added and there is no overlap between windows. As shown in Equ. 4, the aggregated input passes through the low-rank adapters and augments on the output of the quantized weight.

<span id="page-14-1"></span><span id="page-14-0"></span>
$$\mathbf{H}_O = \mathbf{H}_I \tilde{\mathbf{W}} + s \mathcal{A}(\mathbf{H}_I) \mathbf{L}_1 \mathbf{L}_2, \tag{4}$$

As the input feature dimension is changed  $(h \to h/l)$ , the dimension of  $L_1$  is changed, accordingly.

## B.3 MERGE FOR INFERENCE

In this section, we discuss the feasibility of merging the learned LoRA adapters with the quantized weights with the quantization characteristic kept. In this way, the merged weights can be deployed with low-bit inference directly.

The left term of Equ. 4 can be calculated as:

$$\mathbf{H}_{I}\tilde{\mathbf{W}} = \mathbf{H}_{I}(\mathbf{A} \odot (\hat{\mathbf{W}} - \mathbf{B})) = \mathbf{H}_{I}(\mathbf{A} \odot \hat{\mathbf{W}}) - \left[\sum_{m=1}^{d} h_{i,m} (\alpha_{\lfloor \frac{m}{g} \rfloor, j} \beta_{\lfloor \frac{m}{g} \rfloor, j})\right], \tag{5}$$

Considering the right term of Equ. [4,](#page-14-0) for simplicity, we use P = [pi,j ] to substitute L1L<sup>2</sup> and P ∈ R(d/l)×<sup>o</sup> :

<span id="page-15-0"></span>
$$s\mathcal{A}(\mathbf{H}_I)\mathbf{L} = s\mathbf{H}_I^a\mathbf{L} = s\left[\sum_{k=1}^{d/g} h_{i,k}^a p_{k,j}\right]$$
$$= s\left[\sum_{k=1}^{d/g} \sum_{r=1}^g h_{i,(k-1)g+r} p_{k,j}\right] = s\left[\sum_{m=1}^d h_{i,m} p_{\lfloor \frac{m}{g} \rfloor,j}\right]$$
(6)

Substitute Equ. [5](#page-14-1) and Equ. [6](#page-15-0) into Equ. [4:](#page-14-0)

<span id="page-15-1"></span>
$$\mathbf{H}_{O} = \mathbf{H}_{I}(\mathbf{A} \odot \hat{\mathbf{W}}) - \left[\sum_{m=1}^{d} h_{i,m} (\alpha_{\lfloor \frac{m}{g} \rfloor, j} \beta_{\lfloor \frac{m}{g} \rfloor, j})\right] - s \left[\sum_{m=1}^{d} h_{i,m} p_{\lfloor \frac{m}{g} \rfloor, j}\right]$$

$$= \mathbf{H}_{I}(\mathbf{A} \odot \hat{\mathbf{W}}) - \left[\sum_{m=1}^{d} h_{i,m} (\alpha_{\lfloor \frac{m}{g} \rfloor, j} \beta_{\lfloor \frac{m}{g} \rfloor, j} - s p_{\lfloor \frac{m}{g} \rfloor, j})\right]$$

$$= \mathbf{H}_{I}(\mathbf{A} \odot \hat{\mathbf{W}}) - \mathbf{H}_{I} \left[\alpha_{\lfloor \frac{m}{g} \rfloor, j} (\beta_{\lfloor \frac{m}{g} \rfloor, j} - s \frac{p_{\lfloor \frac{m}{g} \rfloor, j}}{\alpha_{\lfloor \frac{m}{g} \rfloor, j}})\right]$$

$$= \mathbf{H}_{I}(\mathbf{A} \odot \hat{\mathbf{W}}) - \mathbf{H}_{I}(\mathbf{A} \odot (\mathbf{B} - s \mathbf{L}_{1} \mathbf{L} \mathbf{2} \otimes \mathbf{A})), \tag{7}$$

where ⊘ is the element-wise divide operation. From Equ. [7,](#page-15-1) we can conclude that the adapter weights of QA-LoRA can be perfectly merged into the quantized weights by merely updating the zero-point matrix B into B − sL1L2 ⊘ A.