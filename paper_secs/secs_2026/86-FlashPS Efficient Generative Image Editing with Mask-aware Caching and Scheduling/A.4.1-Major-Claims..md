# A.4.1 Major Claims.

- (C1): FlashPS achieves better serving performance across various diffusion models compared to other baselines, as shown in Fig. [12.](#page-10-1)
- (C2): FlashPS's efficient serving approach does not compromise image quality.

#### A.4.2 Experiments.

Experiment (E1): [End-to-end Serving Performance of SDXL] [30 human-minutes + 1 compute-hour]: Evaluate the serving performance of each baseline method (TeaCache and Diffusers) to reproduce the serving performance of SDXL in the paper.

```
1 cd / app /image - inpainting / scheduler /
2 # Ensure the repo up -to - date
3 git pull
4 # Run the server to test TeaCache and diffusers baseline .
5 # It may take two minutes to start the server .
6 # When the server successfully starts , it will print
7 # " INFO : Uvicorn running on http ://0.0.0.0:8005 (
        Press CTRL +C to quit )" on the console .
8 bash run_server_ootd_no_cb . sh
9 # Send requests to the server .
10 # Note that the first 10 requests are for warm -up.
11 # For each baseline , we send requests with different RPS.
12 # Send requests to evaluate the baseline TeaCache .
13 bash / app /image - inpainting / scheduler / test_ootd_teacache .
        sh
14 # Send requests to evaluate the baseline diffusers .
15 bash / app /image - inpainting / scheduler / test_ootd_diffusers .
        sh
16 # Remember to kill the server .
17 bash kill_server . sh
18 # Run the server to test FlashPS baseline .
19 # It may take two minutes to start the server .
20 # When the server successfully starts , it will print
21 # " INFO : Uvicorn running on http ://0.0.0.0:8005 (
        Press CTRL +C to quit )" on the console .
22 bash run_server_ootd . sh
23 # Send requests to evaluate FlashPS .
24 bash / app /image - inpainting / scheduler / test_ootd_flashps . sh
25 # Remember to kill the server .
26 bash kill_server . sh
27 # Analyze and plot the results .
28 # The script will print out the path to the figure .
29 python scripts / parse_end2end . py
```

You may compare the relative serving performance of each method in the output figure with those in Fig. [12.](#page-10-1)

Experiment (E2): [End-to-end Performance of SD2][30 human-minutes + 1 compute-hour]: Evaluate the serving performance of each baseline method (FISEdit and Diffusers) to reproduce the serving performance of SD2 in the paper.

We configured the FISEdit environment following this link: https://github.com/Hankpipi/diffusers-hetu, which is not included in the docker container.

```
1 # Initialize the environment
2 source activate pytorch
3 # Go to the project directory
4 cd / home / ubuntu /image - inpainting / scheduler
5 # Run the server to evaluate FlashPS .
6 # It may take two minutes to start the server .
7 # When the server successfully starts , it will print
8 # " INFO : Uvicorn running on http ://0.0.0.0:8005 (
        Press CTRL +C to quit )" on the console .
9 bash run_server_sd2_cb . sh
10 # Send requests to evaluate FlashPS .
11 bash scripts / test_cb_sd2 . sh
12 # kill the server
13 bash scripts / kill_gpu_processes . sh
14 # Run the server to evaluate Diffusers .
15 # It may take two minutes to start the server .
16 # When the server successfully starts , it will print
17 # " INFO : Uvicorn running on http ://0.0.0.0:8005 (
        Press CTRL +C to quit )" on the console .
18 bash run_server_sd2_no_cb . sh
19 # Send requests to evaluate Diffusers .
20 bash scripts / test_no_cb_sd2 . sh
21 # kill the server
22 bash scripts / kill_gpu_processes . sh
23 # activate fisedit environment
24 conda activate fisedit
25 source ~/ Hetu / hetu .exp
26 # Run the server to evaluate FisEdit .
27 # It may take two minutes to start the server .
28 # When the server successfully starts , it will print
29 # " INFO : Uvicorn running on http ://0.0.0.0:8005 (
        Press CTRL +C to quit )" on the console .
30 bash run_server_fisedit_no_cb . sh
31 # Send requests to evaluate FisEdit .
32 bash scripts / test_fisedit_e2e . sh
33 # kill the server
34 bash scripts / kill_gpu_processes . sh
35 # Analyze and plot the result .
36 # The script will print out the path to the figure .
37 python scripts / parse_end2end . py
```

You may compare the relative serving performance of each method in the output figure with those in Fig. [12.](#page-10-1)

Experiment (E3): [Image Quality Assessment] [10 humanminutes + 30 compute-minutes]: Evaluate the quality of generated images of each baseline quantitatively.

```
1 # Pull the Docker image .
2 docker pull jiangxiaoxiao / flashps : latest
3 # Clear the stopped container , if it exists
4 docker kill flashps - ae
5 docker rm flashps - ae
6 # Spin up the container . This may take minutes .
7 docker run -d -- name flashps - ae -- runtime = nvidia -- gpus
        all -- shm - size =16 g -e NVIDIA_VISIBLE_DEVICES =all -e
        CUDA_VISIBLE_DEVICES =0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 -e
        CONDA_DEFAULT_ENV ="" -e CONDA_AUTO_ACTIVATE_BASE =
        false jiangxiaoxiao / flashps sleep infinity
8 # Enter the container
9 docker exec - it flashps - ae zsh
10 # Activate the environment
11 conda activate flashps
12 # Go to the directory .
13 cd / app /image - inpainting /
14 # Run the script .
15 bash scripts / test_quality . sh
```

The results will be printed on the console. You may compare them with those in Table [2.](#page-11-3)

Experiment (E4): [Distribution of Mask Ratios] [10 humanminutes + 10 compute-minutes]: Illustrate the distribution of mask ratio. You can run this experiment on your local machine.

```
1 # clone the repository
2 git clone https :// github . com / Sylvia -16/ FlashPS . git
3 # Go to directory
4 cd FlashPS / mask_ratio_distribution
5 # Run the plot script
6 python plot_mask_ratio_2traces . py
```

You may compare the output figure mask\_ratio\_2traces.pdf with Fig. [3.](#page-2-3)