print("\n> Importando bibliotecas...")
import gradio as gr
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import pandas as pd
from pathlib import Path
import json
print("> Bibliotecas importadas com sucesso!")

def load():
    return