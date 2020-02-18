import torch
from transformers import *

def get_pretrained_bert():
    model_class, tokenizer_class, config_class = BertForPreTraining, BertTokenizer, BertConfig
    WEIGHTS_BASE_PATH = './pre_trained_weights/clinical_bert/bert_pretrain_output_disch_100000/'
    bert_config_file = 'bert_config.json'
    pytorch_model_file = 'pytorch_model.bin'
    vocab_file = 'vocab.txt'
    config_obj = config_class.from_json_file(WEIGHTS_BASE_PATH + bert_config_file)
    config_obj.output_hidden_states = True
    config_obj.output_attentions = True

    # Create model object
    model = model_class(config_obj)
    state_dict = torch.load(WEIGHTS_BASE_PATH + pytorch_model_file)
    model.load_state_dict(state_dict)

    # Create tokenizer object
    tokenizer = tokenizer_class(WEIGHTS_BASE_PATH + vocab_file, do_lower_case=True)
    return model, tokenizer, config_obj

