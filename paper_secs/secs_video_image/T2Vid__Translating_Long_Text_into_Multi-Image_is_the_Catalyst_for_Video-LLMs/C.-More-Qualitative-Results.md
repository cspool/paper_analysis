# C. More Qualitative Results

In this section, we present more qualitative results, as illustrated in Fig. [11](#page-16-0) and Fig. [12.](#page-16-1)

## <span id="page-14-0"></span>System message

You are an AI assistant who will help me match an answer with several options of a single-choice question.

#### Prompt

You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer.

If the meaning of all options is significantly different from the answer, output Z. You should directly output a single uppercase character, such as A, B, C, D (if they are valid options), and Z, and nothing else. Here are two examples.

Example 1:

Question: What is the main object in the image?

Options: A. teddy bear.

B. rabbit.

C. cat.

D. dog.

Answer: a cute teddy bear

Output: A

Example 2:

Question: What is the main object in the image?

Options: A. teddy bear.

B. rabbit.

C. cat.

D. dog.

Answer: Spider Output: Z

Now here are the question, options, and the answer, you should match and give me the option letter:

Question: {Question} Options: {Options}

Answer: {Model Answer}

Output:

Table 4. Template for prompting LLM to perform option matching. {Question} is the specific question of a benchmark sample, and {Options} are corresponding choices of the question. {Model Answer} is the raw prediction of MLLMs.

#### <span id="page-15-0"></span>User:

You will be provided with some information about a video, including a global caption for the whole video, global captions for each video frame, and descriptions of key objects in each frame. Answer the questions using the information below.

#### Video information:

Caption: This video shows the carpentry process. At first, the person sits on a workbench and measures the length of the plank. Then, he uses a saw to cut the wooden plank into multiple pieces. Then, he uses a hammer to nail two pieces of wood together. Finally, he takes a break and drinks water, and leaves the workshop.

## Frame information:

## Frame 1:

Caption: the person sits on a workbench and holds a hammer in his right hand.

the person: the person is sitting on a workbench in the center of the frame. He is looking at a wooden plank in front of him.

hammer: He has a hammer in his right hand.

workbench: He is sitting on a workbench in the center of the frame.

