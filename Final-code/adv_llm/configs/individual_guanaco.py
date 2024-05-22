import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_guanaco'

    config.tokenizer_paths=["guanaco/models--TheBloke--guanaco-7B-HF/snapshots/293c24105fa15afa127a2ec3905fdc2a0a3a6dac"]
    config.model_paths=["adv_llm/models/guanaco/models--TheBloke--guanaco-7B-HF/snapshots/293c24105fa15afa127a2ec3905fdc2a0a3a6dac"]
    config.conversation_templates=['vicuna']
    config.filter_cand = False

    return config