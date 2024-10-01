# The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective

This is the official repo of the paper The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective.

We use code from several resources, which we disclose here. First, the basis for training and 
attacking the CNNs stems from [Sehwag et al., 2020]. We modified the code according to our needs. The
code for DenseNet121 stems from the official PyTorch library. To attack and evaluate the LLMs
we use the official implementation of the attack [Zou et al., 2023]. CIFAR-10 and CIFAR-100 were
also downloaded from PyTorch. The LLMs stem from HuggingFace.


We provide in the folder `demo` a jupyter notebook, that gives an easy start to run the experiments with your own models.

## Dependencies
We provide the list of requirements in the file `requirements.txt`, which can be installed with 
```
pip install -r requirements.txt
```

### Other code ressources
This code base is mostly build on the code of the paper [Hydra](https://github.com/inspire-group/hydra). We also  include the code of the repository [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch).

### Models
The models to evaluated must be placed in `./demo-models/` and trained using the code in `../Final-code`
## License 
<p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>



