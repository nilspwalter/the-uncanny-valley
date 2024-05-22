import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_falcon'

    config.tokenizer_paths=["falcon/models--tiiuae--falcon-7b-instruct/snapshots/cf4b3c42ce2fdfe24f753f0f0d179202fea59c99"]
    config.model_paths=["models--tiiuae--falcon-7b-instruct/snapshots/cf4b3c42ce2fdfe24f753f0f0d179202fea59c99"]
    config.conversation_templates=['vicuna']

    return config