# MuJoCo Fetch (Pick-and-Place)

- The model issues repetitive movement commands without adapting based on task progress or environment feedback, resulting in oscillatory behaviors like moving up and down, left and right, or in a single direction with no meaningful progress toward grasping or placing the cube. For example, it may move forward repeatedly even after overshooting the target or oscillate without ever attempting a grasp.
- Ignores visual feedback from images and does not adjust its actions in response to clear cues about the state or position of the cube, gripper, or target marker. This results in the model issuing irrelevant or counterproductive actions, such as continuing to move when the cube has not been grasped.
- Issues the "stop" command prematurely before the end-effector is even close to the target marker, ending the task early without justification or clear success criteria.

