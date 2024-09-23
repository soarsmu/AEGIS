# Synthesizing Efficient and Permissive Programmatic Runtime Shields for Neural Policies

This repository contains the code for the paper "Synthesizing Efficient and Permissive Programmatic Runtime Shields for Neural Policies" co-authored by [Jieke Shi](https://jiekeshi.github.io), [Junda He](https://jundahe.notion.site/Junda-He-ab59db5a4604450dadd66591c1c357da), [Zhou Yang](https://yangzhou6666.github.io), [Ðorđe Žikelić](https://djordjezikelic.github.io), and [David Lo](http://www.mysmu.edu/faculty/davidlo/).

## Abstract

With the increasing use of neural policies in control systems, ensuring their safety and reliability has become a critical software engineering task. One prevalent approach to ensuring the safety of neural policies is to deploy programmatic runtime shields alongside them to correct their unsafe commands. However, the programmatic runtime shields synthesized by existing methods are either computationally expensive or insufficiently permissive, resulting in high overhead and unnecessary interventions on the system. To address these challenges, we propose Aegis, a novel framework that synthesizes lightweight and permissive programmatic runtime shields for neural policies. Aegis achieves this by formulating the seeking of a runtime shield as a sketch-based program synthesis problem and proposing a novel method that leverages counterexample-guided inductive synthesis and Bayesian optimization to solve it. To evaluate Aegis and its synthesized shields, we use four representative control systems and compare Aegis with the current state-of-the-art. Our results show that the programmatic runtime shields synthesized by Aegis can correct all unsafe commands from neural policies, ensuring that the systems do not violate any desired safety properties at all times. Compared to the current state-of-the-art, Aegis’s shields exhibit a 2.1× reduction in time overhead and a 4.4× reduction in memory usage, suggesting that they are much more lightweight. Moreover, Aegis’s shields incur an average of 1.6× fewer interventions than other shields, showing better permissiveness.

## Environment Setup

We provide a Dockerfile to set up the environment for running the experiments. To build the Docker image, run the following command:

```bash
docker build -t aegis .
```

To run the Docker container, execute the following command:

```bash
docker run -it -v /path/to/aegis:/aegis aegis
```

## Running Experiments

We provide the scripts to run the experiments of Aegis and the baseline VRL, which can be found in their respective directories, `aegis` and `baseline`, so that you can reproduce the results of our paper easily.

To run the experiments of Aegis, execute the following command:

```bash
cd aegis

bash RQ1.sh # Run the experiments for RQ1
bash RQ23.sh # Run the experiments for RQ2 and RQ3
```
The results of the experiments will be stored in the `results` directory.

To run the experiments of VRL, execute the following command:

```bash
cd baseline

bash run_test_1000ep_table_1.sh # Run the experiments for RQ1
bash RQ23.sh # Run the experiments for RQ2 and RQ3
```

The results of the experiments will be stored in the `results` directory.

We hope the above instructions are helpful for you to reproduce the results of our paper. If you have any questions, please feel free to contact us.