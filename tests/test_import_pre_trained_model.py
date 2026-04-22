import os
from pathlib import Path
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def change_environment_name(destiny_dir_string: str, os_enviroment_name: str):
    try:
        base_path = Path(__file__).parent.parent #todo retirar um parent fora de test
        destiny_dir = base_path / Path(destiny_dir_string)
        destiny_dir.mkdir(parents=True, exist_ok=True)
        os.environ[os_enviroment_name] = destiny_dir_string
    except Exception as e:
        print(f"> Exceção: {e}")
    
    if (os.environ.get(os_enviroment_name) == destiny_dir_string):
        return True
    return False


def download_model(personalized_cache_dir: bool, pretrained_model: str, model_destiny_dir: str):
    try:
        if(personalized_cache_dir):
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model,
                cache_dir = model_destiny_dir)
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model,
                cache_dir = model_destiny_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            model = AutoModelForCausalLM.from_pretrained(pretrained_model)

        #todo retirar anotacoes fora de test
        print("> Encontrando melhor dispositivo para rodar o modelo...")
        best_device = find_best_device()
        print(f"> {best_device} selecionado como dispositivo!")

        model = model.to(best_device)

        return model, tokenizer, best_device
    except Exception as e:
        print(f"> Exceção: {e}")


def find_best_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device


# def import_pre_trained_model():
def import_model():
    models_dir_name = "AI_models"
    hf_enviroment_name = "HF_HOME"
    models = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-3B"]
    model_name = models[1]
    ## inserir token do HF
    model_personalized_dir_sucess = False


    print(f"\n> Configurando variável de Ambiente: {hf_enviroment_name}...")
    if(change_environment_name(models_dir_name, hf_enviroment_name)):
        print(f"> Variável {hf_enviroment_name} configurada para o diretório: {models_dir_name} com sucesso!")
        model_personalized_dir_sucess = True
    else:
        print(f"> Ocorreu um erro na definição da Variável de Ambiente {hf_enviroment_name}. O modelo de IA será baixado no diretório padrão")


    print("\n> Importando Modelo de IA...")
    model, tokenizer, best_device = download_model(model_personalized_dir_sucess, model_name, models_dir_name)
    print("> Modelo importado com sucesso!\n")

    return model, tokenizer, best_device