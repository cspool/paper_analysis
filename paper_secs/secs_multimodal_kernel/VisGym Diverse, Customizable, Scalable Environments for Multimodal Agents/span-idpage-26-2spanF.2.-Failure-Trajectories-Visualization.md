# <span id="page-26-2"></span>F.2. Failure Trajectories Visualization

Using StringSight, we visualize the trajectory for each failure type. In each trajectory, we show the prompt, the image, the models' raw output, and the action parsed from the raw output. We also show the output from StringSight for each trajectory, tagged "Reason" and "Evidence" at the top, where "Reason" stands for StringSight's reason for classifying this trajectory into a specific failure category, and "Evidence" stands for the evidence in the trajectory that leads to the conclusion.

- (1) Restricted action space and action looping: As in Sec. F.2.1, we show a case of action looping of GPT-5 on the Jigsaw task. The model repeatedly takes the same action "("swap", (0,0), (0,1))", resulting in looping behaviors without making any progress.
- (2) State mismanagement: As in Sec. F.2.2, we show a case of Claude Sonnet 4 on Maze 2D. In Observation 7, the model takes action "("move", 2)", which leads to the environment feedback "Cannot move into a wall." However, at Observation 16, the model is in the exact same state, disregards the previous feedback, and takes the same action "("move", 2)" again.
- (3) Early termination: We show in Sec. F.2.3 a case of Gemma 3 27B Instruct on Matchstick Equation, where the model decides to give up and terminate at step 13, while the model is allowed to take 30 steps in total.
- (4) Failure to use visual or spatial information: As in Sec. F.2.4, we show a case of Gemini 2.5 Pro on Mental Rotation 3D (Cube). In the last three steps, after rotating in the wrong direction, the model does not take the visual information into account and continues to rotate in the same direction, which moves the object even farther away from the target position.

