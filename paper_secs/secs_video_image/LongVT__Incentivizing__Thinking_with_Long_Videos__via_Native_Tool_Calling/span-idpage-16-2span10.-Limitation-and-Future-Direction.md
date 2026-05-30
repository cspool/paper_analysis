# <span id="page-16-2"></span>10. Limitation and Future Direction

While our efficiency analysis in Section [7](#page-15-0) confirms that multi-turn tool interactions do not impose significant latency penalties, the memory footprint of such recursive reasoning remains a bottleneck. The single-agent architecture of LongVT is constrained by the inherent context window of the underlying LMM: as the number of interaction turns increases—driven by the need for multiple crop video calls to inspect ultra-long or infinite video streams—the accumulation of history tokens (including dense visual features returned by tools) can rapidly exhaust the context budget. This accumulation poses a risk of Out-of-Memory errors during training and imposing performance degradation due to truncation.

A promising future direction to resolve this limitation lies in multi-agent collaboration. Inspired by recent advancements in multi-agent reinforcement learning such as MATPO [\[33\]](#page-9-24), we envision a hierarchical framework where context management is decoupled from reasoning. In this future paradigm, a "Manager Agent" could orchestrate highlevel planning and dispatch sub-tasks to specialized "Worker Agents," each responsible for inspecting distinct temporal segments or executing specific tool calls. By enabling workers to summarize their observations into concise natural language updates for the manager, such a system could theoretically support infinite-horizon reasoning loops without succumbing to context overflow. We leave the exploration of this scalable, divide-and-conquer architecture to future work.

## <span id="page-16-3"></span>11. Broader Impact

LongVT advances the field of long-video understanding by introducing an agentic framework capable of proactive evidence seeking and self-correction. By enabling LMMs to dynamically inspect and re-examine video segments, this work addresses critical reliability issues—such as hallucinations and temporal misalignment that hinder the deployment of AI in high-stakes domains. As video-based AI systems

become integral to applications ranging from automated surveillance and content moderation to educational analytics and assistive technologies for the visually impaired, the improved factual grounding and transparency offered by LongVT support safer and more trustworthy interactions.

## <span id="page-17-0"></span>12. Ethical Considerations

Advancing Reliability and Safety. LongVT is explicitly designed to enhance the reliability of video LMMs by mitigating hallucinations through on-demand visual verification. By grounding answers in retrieved video evidence, the system reduces the likelihood of fabricating events or misinterpreting context, thereby fostering more trustworthy predictions in complex, long-form video scenarios.

Transparency and Interpretability. By decomposing the reasoning process into observable steps—global skimming, tool invocation, evidence retrieval, and self-reflection—LongVT inherently supports transparent decision-making. This explicit chain of tool-augmented thought facilitates auditing and debugging, allowing users to trace *why* a model arrived at a specific conclusion and *which* video segments informed that decision.

Responsible Use of Data. The system does not access private or surveillance feeds, and no additional personally identifiable information is introduced. We advocate for the strict adherence to privacy standards and ethical guidelines when deploying such long-video analysis tools in real-world settings.

## Prompt Template for RL

## <span id="page-18-0"></span>**System**

You are a helpful assistant.

Tools

You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags:

<tools>{\"type\": \"function\", \"function\": {\"name\": \"crop\_video\", \"description\": \"Crop a video to a specified duration.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"video\_path\": {\"type\": \"string\", \"description\": \"Path to the video file\", \"enum\": null}, \"start\_time\": {\"type\": \"number\", \"description\": \"Start time in seconds\", \"enum\": null}, \"end\_time\": {\"type\": \"number\", \"description\": \"End time in seconds, must be > start\_time\", \"enum\": null}}, \"required\": []}, \"strict\": false}}</tools>

For each function call, return a json object with function name and arguments within <tool\_call></tool\_call> XML tags:\n<tool\_call>{\"name\": <function-name>, \"arguments\": <args-jsonobject>}</tool\_call>

## **User**

(Question here) Think first, call \*\*crop\_video\*\* if needed, then answer. Format strictly as: <think>...</think> <tool\_call>...</tool\_call> (if tools needed) <answer>...</answer>. The Video path for this video is:

Figure 5. Prompt Template Utilized for RL. This template outlines the structural guidelines and system instructions provided to the model during the RL training phase.

