# The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective

This is the official repo of the paper [The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective](https://arxiv.org/abs/2405.16918).  

## Dependencies
We provide the list of requirements in the file `requirements.txt`, which can be installed with 
```
pip install -r requirements.txt
```
## Demo
We created a jupyter notebook `demo.ipynb` which shows how to run experiments and replicate the results from the paper. To give an easier start, we make the models used in the paper publically available under this [link](https://dl.cispa.de/s/LdFw5ZJMHMgq75e). The models must be placed in a directory called `demo-models`. Besides the PGD-attack presented  in the paper, we also provide code for C&W attack, both can be easily selected in the notebook.

## Reference
In case you find our work useful, please consider citing
```
@misc{walter2024uncanny,
  title={The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective},
  author={Walter, Nils Philipp and Adilova, Linara and Vreeken, Jilles and Kamp, Michael},
  archivePrefix={arXiv},
  eprint={2405.16918},
  year={2024}
}
```

### Other code ressources
This code base is mostly build on the code of the paper [Hydra](https://github.com/inspire-group/hydra). We also  include the code of the repository [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch).

## License 
<p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>
