# <span id="page-46-0"></span>F.2.4 Sample trajectory for "Visual/Spatial Perception" (Gemini 2.5 Pro) (back to Sec. F.2)

### Reason

The model frequently misjudges the effect of its rotations, such as applying large yaw or roll corrections that result in the object being further from the target orientation, indicating it is not fully utilizing the visual feedback provided after each step.

### **Evidence**

After '('rotate', [-45.0, 0.0, 0.0])', the model observes the object is now vertical and then applies '('rotate', [90.0, 0.0, 0.0])', which overcorrects. Later, it issues '('rotate', [-90.0, 0.0, 0.0])' and then again '('rotate', [-90.0, -30.0, 0.0])', showing it is not accurately interpreting the visual state changes. The model also states, 'This is a mess,' and attempts to 'reset and reconsider,' indicating confusion about the spatial state.

### Instruction I

```
You see two images side-by-side:

- Left: the original 3D object (target orientation)

- Right: the same object under an unknown rotation

Your goal is to rotate the right object back to the original target orientation.

Available actions:

1. 'rotate': Rotate the 3D object by Euler angles. Format: '('rotate', [dyaw, dpitch, droll])' where each of dyaw, dpitch, droll is in degrees between -180 and 180.

2. 'stop': Submit your final rotation. Format: '('stop', 'stop')'

Success: You succeed if you end up within 10.0 degrees of target orientation.

Please respond with exactly one action and its arguments in the specified format. For example:

- To rotate in yaw: '('rotate', [15.0, 0.0, 0.0])'

- To rotate in roll: '('rotate', [0.0, 0.0, -10.0])'

- To submit: '('stop', 'stop')'

This is step 1. You are allowed to take 19 more steps.
```

