# B Case Study

To better understand the behavioral differences between baseline and TLDR strategies, we conduct a qualitative analysis using the DeepSeek-R1-Distill-Qwen-7B model. Case studies are drawn from three representative math datasets: GSM8K, AIME, and MATH500. As shown in Figures [6–](#page-20-0)[8,](#page-22-0) the baseline model tends to generate verbose reasoning paths with redundant or speculative content. In contrast, TLDR produces significantly more concise outputs while maintaining correctness and logical structure. These examples demonstrate TLDR's ability to suppress unnecessary reasoning tokens—such as exploratory or reflective phrases—leading to more efficient and focused reasoning processes.

### <span id="page-19-0"></span>Evaluation Prompt on Dataset

### === EVALUATION PROMPT FOR GSM8K ===

<|begin\_of\_sentence|>Please reason step by step, and put your final answer within 'boxed'.

<|User|>query<|Assistant|>Given the following problem, reason and give a final answer to the problem. Problem: {question}

Your response should end with The final answer is [answer] ¨ where [answer] is the response to the ¨ problem. <think>

### === EVALUATION PROMPT FOR MATH500 ===

<|begin\_of\_sentence|>Please reason step by step, and put your final answer within '

boxed'. <|User|>{query}<|Assistant|>Return your final response within 'boxed'. {problem}. <think>

### === EVALUATION PROMPT FOR AIME24 ===

<|begin\_of\_sentence|>Please reason step by step, and put your final answer within '

boxed'. <|User|>query<|Assistant|>Return your final response within '

boxed'. {problem}. <think>

### === EVALUATION PROMPT FOR AMC ===

<|begin\_of\_sentence|>Please reason step by step, and put your final answer within 'boxed'. <|User|>query<|Assistant|>Return your final response within 'boxed'. {problem}. <think>

### === EVALUATION PROMPT FOR MINERVAMATH ===

<|begin\_of\_sentence|>Please reason step by step, and put your final answer within 'boxed'. <|User|>query<|Assistant|>Return your final response within 'boxed'. problem. <think>

Figure 5: Evaluation Prompt for GSM8K, MATH500, AIME24, AMC, MinervaMath

### Generated Answer For GSM8K

### <span id="page-20-0"></span>Question:

Michael is traveling on the interstate at an average speed of 50km/hr (taking into account all the necessary stops for gas, relaxation, etc.). If he goes on like this for 3 days, and the distance between Alaska and Texas is 6000 km, what percentage of this distance has he covered?

