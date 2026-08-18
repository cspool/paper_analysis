# *E. Experiment workflow*

Inside the container, we provide a shell script ttp\_simulations.sh that has all the simulation commands needed to generate results. However, due to the large number of simulations, we do not recommend running the shell script directly, as it runs the simulations sequentially. Instead, depending on the resources, simulations should be run in parallel. The shell script simply serves as a reference for simulation commands. Workflow for launching parallel jobs depends on the system being used, therefore we cannot provide a one-for-all script to launch parallel simulations. To launch a simulation in the container,

\$ cd /home/root/vulkan-sim-root \$ source init\_vars.sh \$ cd RayTracingInVulkan/build/linux/bin \$ ./RayTracer --scene 20 --width 32 \ --height 32 --samples 1 > ship\_pt.log

We also provide all of the raw simulation logs and the Python scripts that we used to plot the figures in this paper. To generate the figures, following command can be used inside the container,

\$ python3 figure1.py

This will generate fig1.png using the simulation logs under ttp\_raw\_simulation\_logs. Other figures can be generated in a similar fashion.

