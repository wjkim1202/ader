# An Adaptive Entropy-Regularization Framework for Multi-Agent Reinforcement Learning (ADER), accepted at ICML 2023



This codebase is built on top of the [PyMARL](https://github.com/oxwhirl/pymarl) framework for multi-agent reinforcement learning algorithms.
This codebase is implemented based on [PyMARL](https://github.com/oxwhirl/pymarl), [FOP](https://github.com/liyheng/FOP) and [FACMAC](https://github.com/oxwhirl/facmac) githubs.
Please refer to that repo for more documentation.

## Setup instructions

Set up StarCraft II and SMAC:
```
bash install_sc2.sh
```
You can install the modified SMAC environment (we only modify a single file: /smac/smac/env/starcraft2/starcraft2.py)
```
pip install -e smac/

```


## Run an experiment 

To run the ADER algorithm on some SMAC map (say '1c3s5z') for 2mil timesteps:
```
python src/main.py --config=ader_smac --env-config=sc2 with env_args.map_name=1c3s5z t_max=2000000
```


## Cite the paper

To cite our paper, use the following bib

```
@InProceedings{pmlr-v202-kim23v,
  title = 	 {An Adaptive Entropy-Regularization Framework for Multi-Agent Reinforcement Learning},
  author =       {Kim, Woojun and Sung, Youngchul},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {16829--16852},
  year = 	 {2023},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR}
}
```
