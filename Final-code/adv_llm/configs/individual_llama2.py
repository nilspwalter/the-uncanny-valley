import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_llama2'

    config.tokenizer_paths=["llama/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"]
    config.model_paths=["llama/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"]
    config.conversation_templates=['llama-2']
    config.filter_cand = False
    
    return config