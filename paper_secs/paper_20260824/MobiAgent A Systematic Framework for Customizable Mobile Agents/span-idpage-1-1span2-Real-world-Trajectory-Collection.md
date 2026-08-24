# <span id="page-1-1"></span>2 Real-world Trajectory Collection

In this section, we present a framework for collecting real-world GUI task trajectories with rich contextual information and high data quality, while requiring minimal human efforts. The collected trajectories are further refined and used to train our agent models, enabling them to learn from diverse and realistic user interactions.

## <span id="page-1-0"></span>2.1 Agent Trajectory Collection

To ensure that the collected trajectories accurately reflect real-world user behavior, we developed a lightweight action recording tool for smartphones. During data collection, annotators interact with this tool, which captures and logs every user action before forwarding it to the device. The tool renders all bounding boxes (bbox) of interactive elements on the screen based on XML files that describe UI hierarchies. If a bbox is missing due to incomplete XML data, we employ OmniParser [\[17\]](#page-14-13) to regenerate the corresponding bbox.

The recorded action space includes:

- Click(bbox: List[int]): Click at the center point of bbox using absolute coordinates.
- Input(text: str): Input text to a currently activated input area.
- Swipe(direction: Literal["UP","DOWN","LEFT","RIGHT"]): Swipe to direction.
- Done(): Signify the completion of the task.

For rare but important cases that are difficult to capture with the recording tool efficiently, such as closing a pop-up ad or waiting for it to disappear, we create a separate single-step dataset containing these cases and add an new wait action to the action space:

• Wait(sec: int): Wait for sec seconds before the next action.

For simpler tasks which a powerful pretrained VLM (e.g. Gemini-2.5) can handle, we directly prompt the VLM to execute these tasks and record the trajectories with a uniform format, which further improves the efficiency of data collection.

