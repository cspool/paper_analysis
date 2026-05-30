# *D. Installation*

For artifact evaluation, begin by downloading the top-level repository from Zenodo:

\$ wget −O BitMoD.zip https://zenodo.org/records/14252531/ files/BitMoD−HPCA−25.zip \$ unzip BitMoD.zip

The artifact is inside the unzipped BitMoD-HPCA-25 repository, which contains five sub-folders, each targeting one set of experiments:

- 1) bitmod-quant, which runs the baseline weight-only quantization with different data types. This can reproduce the results in Table VI and Table VIII.
- 2) bitmod-sim, contains a custom simulator that calculates the latency and energy of the *BitMoD* accelerator. This can reproduce the results in Fig. 7 and Fig. 8.
- 3) AWQ-BitMoD, which runs AWQ [30] with integer and *BitMoD* data types. This can reproduce the AWQ results in Table XI.

- 4) OmniQuant-BitMoD, which runs OmniQuant [42] with integer and *BitMoD* data types. This can reproduce the OmniQuant results in Table XI.
- 5) SmoothQuant-BitMoD, which runs SmoothQuant [52] with integer and *BitMoD* data types for weight quantization. This can reproduce the results in Table XII.

Please go to every sub-folder and refer to the corresponding 'README.md' for detailed setup instructions. Note that AWQ, OmniQuant, and SmoothQuant will require different conda environments. Hence, please change back to the base environment after installing a conda environment before creating the next.

For example, the AWQ environment can be created by running:

- \$ cd AWQ−BitMoD
- \$ conda create −n awq−bitmod python=3.10 −y
- \$ conda activate awq−bitmod
- # Follow 'README.md' inside AWQ−BitMoD folder to set up other dependencies.
- \$ conda deactivate # change back to the base environment

The OmniQuant and SmoothQuant environments can be created in a similar way by repeating the above step and following their 'README.md' to set up other dependencies.

### *E. Experiment workflow*

Once the environment is set up, we will conduct five sets of experiments, each corresponding to one of the five sub-folders within the BitMoD-HPCA-25 repository.

*1) BitMoD Weight-only Quantization:* Run the basic LLM weight-only quantization experiments to reproduce the results in Table VI and Table VIII.

\$ cd bitmod quant

\$ conda activate awq−bitmod

In 'run\_exp.sh', modify the 'export' command by specifying the HuggingFace home directory, 'HF\_HOME', on your computer. By default, this can be set to your home directory.

\$ export HF HOME="your/HF HOME/directory"

Then run the following:

\$ bash run exp.sh

The perplexity result will be saved in the folder called 'results\_quant'.

*2) BitMoD Hardware Simulation:* Before running the simulator, go to 'bitmod\_sim' of the repository:

\$ cd bitmod sim

\$ conda activate awq−bitmod

In 'run\_shape\_profile.sh', modify the 'export' command by specifying the HuggingFace home directory, 'HF\_HOME', on your computer:

```
$ export HF HOME="your/HF HOME/directory"
```

Then generate the model shape information that can be passed to the accelerator simulator:

```
$ bash run shape profile.sh
```

Next, run different simulators for the baseline FP16 accelerator, ANT, OliVe, and *BitMoD*:

```
$ python test baseline.py −−is generation
```

- \$ python test ant.py −−is generation
- \$ python test olive.py −−is generation
- \$ python test bitmod.py −−is generation −−is lossless

The flag *--is generation* is optional. When enabled / disabled, it will evaluate the hardware performance of generative / discriminative tasks. The flag *--is lossless* is optional for *BitMoD*. When enabled / disabled, it will evaluate the hardware performance of lossless / lossy *BitMoD* quantization.

Finally, to generate Fig. 7 and Fig. 8 of the paper, go to 'bitmod\_sim/plot' directory and run the Jupyter notebooks inside. Note that the cycle and energy numbers are the same as those output by the simulators.

*3) AWQ:* Go to the 'AWQ-BitMoD' directory:

```
$ cd AWQ−BitMoD
```

\$ conda activate awq−bitmod

In 'run\_awq.sh' and 'run\_eval\_ppl.sh', modify the first 'export' command by specifying the HuggingFace home directory, 'HF\_HOME', on your computer:

```
$ export HF HOME="your/HF HOME/directory"
```

Then, run the following two commands separately:

```
$ bash run awq.sh # will take several hours
```

\$ bash run eval ppl.sh

The perplexity results will be saved in the folder called 'results'. You can compare these with the AWQ results in Table XI.

*4) OmniQuant:* Go to the 'OmniQuant-BitMoD' directory:

```
$ cd OmniQuant−BitMoD
```

\$ conda activate omniquant−bitmod

The comprehensive scripts to reproduce the Table XI OmniQuant results are available in the 'scripts' directory. Before running any command in the scripts, execute the following 'export' command and specify the HuggingFace home directory, 'HF\_HOME', on your computer:

#### \$ export HF HOME="your/HF HOME/directory"

In every shell script, you need to change the parameter of *--model* flag to the LLM path in your computer. By default, this can be set to the LLM directory from the official HuggingFace website. For example, inside the script 'llama-2-13b-int.sh', the Llama-2-7B model path can be specified with:

```
−−model meta−llama/Llama−2−7b−hf
```

After changing the *--model* flag to the correct model path, copy and execute every script's python command under the 'OmniQuant-BitMoD' directory. You may check the perplexity results at the end of the log file specified by the *--output dir* flag in every command, and compare those with the OmniQuant results in Table XI.

*5) SmoothQuant:* Go to the 'SmoothQuant-BitMoD' directory:

```
$ cd SmoothQuant−BitMoD
```

\$ conda activate smoothquant−bitmod

In 'run\_experiments.sh', modify the 'export' command by specifying the HuggingFace home directory, 'HF\_HOME', on your computer:

```
$ export HF HOME="your/HF HOME/directory"
```

Then, run the following command:

```
$ bash run experiments.sh
```

The perplexity results will be saved in the folder called 'results\_mod'. You can compare these results with the SmoothQuant results in Table XII.

*F. Evaluation and expected results*

There are five result folders after running the above experiments:

- 1) bitmod-quant/results\_quant, contains the perplexity results in Table VI and Table VIII.
- 2) bitmod-sim/plot, contains two Jupyter notebooks to reproduce Fig. 7 and Fig. 8, respectively.
- 3) AWQ-BitMoD/results, contains the AWQ results in Table XI.
- 4) OmniQuant-BitMoD/log, contains the OmniQuant results in Table XI.
- 5) SmoothQuant-BitMoD/results\_mod, contains the SmoothQuant results in Table XII.

## *G. Methodology*

Submission, reviewing and badging methodology:

- https://www.acm.org/publications/policies/artifactreview-and-badging-current
- https://cTuning.org/ae