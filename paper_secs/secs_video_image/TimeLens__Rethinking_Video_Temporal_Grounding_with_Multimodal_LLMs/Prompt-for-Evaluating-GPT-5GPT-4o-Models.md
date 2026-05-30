# **Prompt for Evaluating GPT-5/GPT-4o Models**

<span id="page-18-0"></span>*Prompt for videos shorter than 80 seconds:*

You are given multiple frames from a video. Each frame is preceded by its timestamp (e.g., 'Frame at 2.5s:'). Please find the visual event described by the sentence '**{query}**' , determining its starting and ending times. Answer in the format 'The event happens in x - y seconds' , where x is the start time and y is the end time (x < y).

### *Prompt for videos longer than 80 seconds:*

You are given multiple frames from a video. Every four adjacent frames are stacked into a 2x2 grid (topleft, top-right, bottom-left, bottom-right in timestamp order). Each stacked frame is preceded by its timestamps (e.g., 'Stacked frames at 1.0s, 2.0s, 3.0s, 4.0s:'). Please find the visual event described by the sentence '**{query}**', determining its starting and ending times. Answer in the format 'The event happens in x - y seconds', where x is the start time and y is the end time (x < y).

Figure 11. Prompts for evaluating GPT-5 and GPT-4o.

