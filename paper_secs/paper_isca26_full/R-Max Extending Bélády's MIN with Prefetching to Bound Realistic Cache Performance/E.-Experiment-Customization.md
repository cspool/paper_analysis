# *E. Experiment Customization*

A list of configurations can be found in ./sim\_configs, including:

- R-Max in various cache levels.
- MIN cache replacement policy in various cache levels.
- Different existing prefetchers in various cache levels.
- SPP-Max and Berti-Max. Note that one must run the simulator using normal SPP or Berti configurations to record the virtual to physical page translations and the list of prefetches issued by the prefetcher. Then change the simulator to use R-Max as the prefetcher in the configuration file, use pre-recorded page translations and the list of prefetches generated. Also please configure R-Max to only issue prefetches found in the generated list. Those configurations are done by the compilation scripts and dividing the simulation workflow into two phases.

