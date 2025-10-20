# Synthesizing Efficient and Permissive Programmatic Runtime Shields for Neural Policies

This repository contains the code for our ACM TOSEM paper "Synthesizing Efficient and Permissive Programmatic Runtime Shields for Neural Policies" co-authored by [Jieke Shi](https://jiekeshi.notion.site), [Junda He](https://jundahe.github.io), [Zhou Yang](https://yangzhou6666.github.io), [Ðorđe Žikelić](https://djordjezikelic.github.io), and [David Lo](http://www.mysmu.edu/faculty/davidlo/).

## Abstract

With the increasing use of neural policies in control systems, ensuring their safety and reliability has become a critical software engineering task. One prevalent approach to ensuring the safety of neural policies is to deploy programmatic runtime shields alongside them to correct their unsafe commands. However, the programmatic runtime shields synthesized by existing methods are either computationally expensive or insufficiently permissive, resulting in high overhead and unnecessary interventions on the system. To address these challenges, we propose Aegis, a novel framework that synthesizes lightweight and permissive programmatic runtime shields for neural policies. Aegis achieves this by formulating the seeking of a runtime shield as a sketch-based program synthesis problem and proposing a novel method that leverages counterexample-guided inductive synthesis and Bayesian optimization to solve it. To evaluate Aegis and its synthesized shields, we use eight representative control systems and compare Aegis with the current state-of-the-art. Our results show that the programmatic runtime shields synthesized by Aegis can correct all unsafe commands from neural policies, ensuring that the systems do not violate any desired safety properties at all times. Compared to the current state-of-the-art, Aegis’s shields exhibit a 2.2× reduction in time overhead and a 3.9× reduction in memory usage, suggesting that they are much more lightweight. Moreover, Aegis’s shields incur an average of 1.5× fewer interventions than other shields, showing better permissiveness.

## Environment Setup

We provide a Dockerfile to set up the environment for running the experiments. To build the Docker image, run the following command:

```bash
docker build -t aegis .
```

⚠️ **Note:** We build the Dockerfile on our Linux machine with AMD64 architecture. It should also work on Intel x86_64 architecture. However, we are not sure if it works on other architectures like Apple Silicon (ARM64). We cannot guarantee the compatibility so please be careful.

To run the Docker container, execute the following command:

```bash
docker run -it -v /path/to/AEGIS:/root/AEGIS aegis
```

## Running Experiments

We provide the scripts to run the experiments of Aegis and the baseline VRL, which can be found in their respective directories, `aegis` and `baseline`, so that you can reproduce the results of our paper easily.

Note that the checkpoint files of the neural policies can be found in the repository of VRL, which can be downloaded from [here](https://github.com/RU-Automated-Reasoning-Group/VRL_CodeReview/tree/master/ddpg_chkp).

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

The results of the experiments will be also stored in the `results` directory.

We hope the above instructions are helpful for you to reproduce the results of our paper. If you have any questions, please feel free to contact us.

## Citation
If you use our code, please kindly cite our paper. The paper has been accepted by ACM Transactions on Software Engineering and Methodology (TOSEM). The ArXiv version and BibTeX entry are provided below:
```bibtex
@misc{shi2025synthesizing,
      title={Synthesizing Efficient and Permissive Programmatic Runtime Shields for Neural Policies}, 
      author={Jieke Shi and Junda He and Zhou Yang and Đorđe Žikelić and David Lo},
      year={2025},
      eprint={2410.05641},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2410.05641}, 
}
```
We will update this entry once the final version is available in the ACM Digital Library.
