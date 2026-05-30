# <span id="page-13-0"></span>A. Answer Judging

We notice that MiniCPM-8B [\[1\]](#page-10-2) often fails to follow instructions properly when we explicitly ask the model to "Answer with the option's letter from the given choices directly", making simple exact matching inaccurate. Specifically, the model often prepends or appends additional text other than the option letters, *e.g*. "Answer: B. Pink.", or gives additional explanations apart from the answer.

To cope with these issues, we adopt a combination of exact matching and LLM matching for assessment. Specifically, we strip the prefixes such as "Answer:" from the prediction and try to use regular expression matching to find the option letter. When the exact matching scheme fails, we use an LLM (Llama-3.1-8B-Instruct [\[64\]](#page-12-24)) to find an option closest to the model prediction. When the LLM matching fails, a placeholder outside of the available options (such as "Z") is returned to denote a wrong answer. Our judging prompt for the LLM is modified from VLMEvalKit [\[65\]](#page-12-25), as shown in Table [4.](#page-14-0)

