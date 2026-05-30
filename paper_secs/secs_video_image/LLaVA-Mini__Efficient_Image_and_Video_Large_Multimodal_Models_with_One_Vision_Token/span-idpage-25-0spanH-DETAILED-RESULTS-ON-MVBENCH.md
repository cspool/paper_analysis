# <span id="page-25-0"></span>H DETAILED RESULTS ON MVBENCH

<span id="page-25-1"></span>Table [15](#page-25-1) reports the detailed results on each subset of MVBench, corresponding to Table [3.](#page-7-0)

Table 15: Detailed results on 20 subsets of MVBench.

| Spatial   | Temporal                 | mPLUG<br>Owl | Video<br>ChatGPT | Video<br>LLaMA | VideoChat LLaMA | VID  | Video<br>LLaVA | LLaVA<br>Mini |
|-----------|--------------------------|--------------|------------------|----------------|-----------------|------|----------------|---------------|
| Average   |                          | 29.7         | 32.7             | 34.1           | 35.5            | 41.4 | 43.1           | 44.5          |
| Action    | Action Sequence          | 22.0         | 23.5             | 27.5           | 33.5            | 63.5 | 44.5           | 44.5          |
|           | Action Prediction        | 28.0         | 26.0             | 25.5           | 26.5            | 42.0 | 50.0           | 44.5          |
|           | Action Antonym           | 34.0         | 62.0             | 51.0           | 56.0            | 26.5 | 49.0           | 76.0          |
|           | Fine-grained Action      | 29.0         | 22.5             | 29.0           | 33.5            | 43.0 | 42.0           | 37.0          |
|           | Unexpected Action        | 29.0         | 26.5             | 39.0           | 40.5            | 42.0 | 54.5           | 58.5          |
| Object    | Object Existence         | 40.5         | 54.0             | 48.0           | 53.0            | 39.0 | 52.5           | 50.0          |
|           | Object Interaction       | 27.0         | 28.0             | 40.5           | 40.5            | 34.5 | 46.5           | 50.0          |
|           | Object Shuffle           | 31.5         | 40.0             | 38.0           | 30.0            | 36.5 | 40.5           | 29.5          |
| Position  | Moving Direction         | 27.0         | 23.0             | 22.5           | 25.5            | 44.0 | 27.0           | 31.0          |
|           | Action Localization      | 23.0         | 20.0             | 22.5           | 27.0            | 35.5 | 28.5           | 32.5          |
| Scene     | Scene Transition         | 29.0         | 31.0             | 43.0           | 48.5            | 22.0 | 84.5           | 85.5          |
| Count     | Action Count             | 31.5         | 30.5             | 34.0           | 35.0            | 44.5 | 44.5           | 35.0          |
|           | Moving Count             | 27.0         | 25.5             | 22.5           | 20.5            | 28.5 | 26.5           | 40.0          |
| Attribute | Moving Attribute         | 40.0         | 39.5             | 32.5           | 42.5            | 19.0 | 53.0           | 48.0          |
| Pose      | State Change             | 44.0         | 48.5             | 45.5           | 46.0            | 55.6 | 38.5           | 41.0          |
|           | Fine-grained Pose        | 24.0         | 29.0             | 32.5           | 26.5            | 37.5 | 34.0           | 29.5          |
| Character | Character Order          | 31.0         | 33.0             | 40.0           | 41.0            | 34.0 | 42.5           | 52.0          |
| Cognition | Egocentric Navigation    | 26.0         | 29.5             | 30.0           | 23.5            | 84.5 | 32.5           | 31.0          |
|           | Episodic Reasoning       | 20.5         | 26.0             | 21.0           | 23.5            | 40.5 | 38.0           | 38.0          |
|           | Counterfactual Inference | 29.5         | 35.5             | 37.0           | 36.0            | 56.5 | 32.0           | 36.0          |