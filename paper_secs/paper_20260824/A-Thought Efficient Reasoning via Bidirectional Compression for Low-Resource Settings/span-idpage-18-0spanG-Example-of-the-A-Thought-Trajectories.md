# <span id="page-18-0"></span>G Example of the A\*-Thought Trajectories

This section provides two comparative examples that illustrate the CoT produced by A\*-Thought versus the original CoT of QwQ-32B and s1.1-32B. Figures [11](#page-19-0) and [12](#page-20-0) demonstrate that A\*-Thought successfully reduces redundant thought trajectories while preserving reasoning ability.

Okay, so I need to convert the rectangular coordinates (0, 3) to polar coordinates. Hmm, let me remember how to do this. Polar coordinates are given as  $(r,\theta)$ , where r is the distance from the origin to the point, and  $\theta$  is the angle made with the positive x-axis.

First, to find r, the formula is  $r=\sqrt{x^2+y^2}$ . In this case, x is 0 and y is 3. Plugging those into the formula, I get  $r=\sqrt{0^2+3^2}=\sqrt{0+9}=\sqrt{9}=3$ . Okay, so r is 3. That seems straightforward.

Now, for the angle  $\theta$ . The formula for  $\theta$  is usually  $\theta$  $\arctan\left(\frac{y}{x}\right)$ . But wait, here x is 0. That would make the denominator zero, which would make the tangent undefined. So, nominator zero, which would make the tangent undefined. So arctan  $(\frac{y}{x})$  would be arctan  $(\frac{3}{0})$ , which is undefined. But arctangent of infinity is  $\frac{\pi}{2}$ , right? Because as x approaches 0 from the positive side,  $\frac{y}{x}$  approaches positive infinity, so arctan approaches  $\frac{\pi}{2}$ . But since x is exactly 0, we have to consider the direction. If x is 0 and y is positive, the point is on the positive y-axis, which is  $\theta = \frac{\pi}{2}$ . If x is 0 and y is negative, it's  $\theta = \frac{3\pi}{2}$ . In this case, y is 3, which is positive, so  $\theta$  should be  $\frac{\pi}{2}$ 

But let me double-check. Sometimes when converting coordinates, especially when x is 0, the angle can be tricky. Let me visualize the point (0,3). In rectangular coordinates, that's 3 units up along the y-axis. So if I imagine the polar coordinate system, the angle  $\theta$  is measured from the positive x-axis. If I rotate  $\frac{\pi}{2}$  radians (which is 90 degrees) from the positive x-axis, I end up pointing straight up along the positive y-axis. That's exactly where the point (0,3) is. So  $\theta = \frac{\pi}{2}$  makes sense here.

Therefore, combining r and  $\theta$ , the polar coordinates should be  $(3, \frac{\pi}{2})$ . But let me make sure there's not another way to represent this. Since polar coordinates can have negative rvalues and angles adjusted by  $\pi$ , but the problem specifies that r > 0 and  $0 \le \theta < 2\pi$ . So we need r positive and  $\theta$ within that interval. Since  $\theta$  is already  $\frac{\pi}{2}$ , which is between 0 and  $2\pi$ , and r is 3, which is positive, this should be the correct answer.

Wait, but just to confirm, what if I tried to use another **angle?** For example, could  $\theta$  be  $\frac{5\pi}{2}$ ? But  $\frac{5\pi}{2}$  is equivalent to  $\frac{\pi}{2}$  when you subtract  $2\pi$ , so it's the same direction. But since  $\theta$  is required to be less than  $2\pi$ ,  $\frac{5\pi}{2}$  is out of range. So  $\frac{\pi}{2}$  is the correct angle here. Also, if I tried to use a negative r, like r=-3, then  $\theta$  would be  $\frac{\pi}{2}+\pi=\frac{3\pi}{2}$ . But r must be greater than 0, so that's not allowed. Therefore,  $(3, \frac{\pi}{2})$  is the only valid polar coordinate representation for the point  $\left(0,3\right)$ under the given constraints.

I think that's all. Let me recap: the point is on the positive y-axis, so the radius is 3, and the angle is  $\frac{\pi}{2}$ . Yep, that seems right. I can't think of any mistakes here. The formulas for r and  $\theta$  both lead to these values, and considering the position of the point confirms it. So the final answer should be  $(3, \frac{\pi}{2})$ . Final Answer: The polar coordinates of the point (0,3) are

$$\left(3,\frac{\pi}{2}\right)$$

To convert the rectangular coordinates (0, 3) to polar coordinates, we start by calculating the radius r using the formula  $r = \sqrt{x^2 + y^2}$ . Here, x = 0 and y = 3:

$$r = \sqrt{0^2 + 3^2} = \sqrt{9} = 3$$

Next, we determine the angle  $\theta$ . Since x = 0 and y = 3 is positive, the point lies on the positive y-axis. The angle  $\theta$  for a point on the positive y-axis is  $\frac{\pi}{2}$ . We verify that  $\theta = \frac{\pi}{2}$  is within the required interval  $0 \le \theta <$ 

 $2\pi$  and that r=3 is positive. Thus, the polar coordinates are:

$$\left(3, \frac{\pi}{2}\right)$$

#### A\*-Thought-QwQ-32B

So, in this case, r is 3. Then  $\theta$  is the angle made with the positive x-axis. The point is (0,3), which is on the positive y-axis. So, the angle  $\theta$  here is  $\frac{\pi}{2}$  radians, because that's 90 degrees, pointing straight up along the y-axis.

But let me verify. The formulas for converting from rectangular to polar coordinates are  $r = \sqrt{x^2 + y^2}$ , which gives  $r = \sqrt{0^2 + 3^2} = \sqrt{9} = 3$ . Then  $\theta$  is  $\arctan\left(\frac{y}{x}\right)$ . But wait, x is 0 here. So,  $\arctan\left(\frac{y}{x}\right)$  would be  $\arctan\left(\frac{3}{0}\right)$ . However, division by zero is undefined, but in the context of polar coordinates the context of polar coordinates and the context of polar coordinates are the context of polar coordinates. dinates, when x is 0,  $\theta$  is either  $\frac{\pi}{2}$  or  $-\frac{\pi}{2}$ , depending on the sign of y. Since y is positive (3),  $\theta$  is  $\frac{\pi}{2}$ . So, that checks out But let me make sure there isn't another angle that could represent the same point. For example, sometimes adding  $2\pi$  to  $\theta$  gives the same direction, but since the problem requires  $\theta$  to be between 0 and  $2\pi$ ,  $\frac{\pi}{2}$  is already within that range. So, the answer should be  $(3, \frac{\pi}{2})$ .

Final Answer: The polar coordinates are  $\left(3, \frac{\pi}{2}\right)$ 

To convert the point (0,3) from rectangular coordinates to polar coordinates, we use the formulas  $r=\sqrt{x^2+y^2}$  and  $\theta = \arctan\left(\frac{y}{x}\right)$ .

1. Calculate r:

$$r = \sqrt{0^2 + 3^2} = \sqrt{9} = 3$$

- 2. Determine  $\theta$ :
  - Since x = 0 and y = 3, the point lies on the positive y-axis.
  - The angle  $\theta$  for the positive y-axis is  $\frac{\pi}{2}$ .

Thus, the polar coordinates are  $\left(3, \frac{\pi}{2}\right)$ 

<span id="page-19-0"></span>Figure 11: A specific example comparing QwQ-32B and A\*-Thought-QwQ-32B (Ours), which red box represents the question, purple box represents the thought path, blue box represents the solution.

#### s1.1-32B

