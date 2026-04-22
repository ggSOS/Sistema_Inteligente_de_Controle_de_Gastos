from test_loader import load
from test_import_pre_trained_model import import_model
import gradio as gr
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import pandas as pd
from pathlib import Path
import json

def sys_prompt_dir_check():
    system_prompt_path = chat_input_dir_check()[1] / Path("system_prompt.jsonl")
    with open(system_prompt_path, 'w', encoding='utf-8') as arquivo:
        arquivo.write(json.dumps({"type": "system_prompt",
                                  "role": "system",
                                  "content": "Você é um assistente útil e amigável."
                                  "Sua base de conhecimento são as ciências econômicas, com foco em economia no dia a dia, e três grupos de informações a seguir: {Resumo da Planilha de Gastos Mensal - gastos menores não serão nomeados/destacados}(opcional) - {Histórico de Conversa}(opcional) - {Pedido de usuário}(obrigatório)."
                                  "Sempre se atente ao historico da conversa, caso exista, e ao resumo de gastos do usuário, caso exista. Tente sempre assumir que o usuário esteja tirando dúvidas sobre a sua planilha de gastos, caso exista um resumo da planilha."
                                  "Caso o usuário pergunte sobre algum gasto que não consta na planilha ele provavelmente é muito baixo, instrua-o a se atentar aos maiores gastos(outliers_numericos no resumo)"
                                  "Se planeje para construir uma resposta sempre com menos de 100 tokens, em Português do Brasil, sem informações/temas sensíveis e se possível em tópicos."}
                                  , ensure_ascii=False))


def chat_input_dir_check():
    input_dir_name = "chat_input"
    input_components_dir_name = "input_components"

    base_path = Path(__file__).parent.parent #todo retirar um parent fora de test
    input_dir = base_path / Path(input_dir_name)
    input_dir.mkdir(parents=True, exist_ok=True)
    input_components_dir = input_dir / Path(input_components_dir_name)
    input_components_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, input_components_dir


def salvar_interacao(
        user_msg,
        bot_msg,
        filename="chat_log.jsonl"
        ):
    ## checagem das pasta de destino
    
    saida = chat_input_dir_check()[1] / Path(filename)

    with open(saida, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "chat_message",
                            "role": "user",
                            "content": user_msg},
                            ensure_ascii=False) + "\n")
        f.write(json.dumps({"type": "chat_message",
                            "role": "assistant",
                            "content": bot_msg},
                            ensure_ascii=False) + "\n")


def extrair_texto(content):
    if isinstance(content, list):
        return content[0]["text"]
    return content


def upload_para_jsonl(arquivo):

    if arquivo is None:
        return "Nenhum arquivo enviado"

    ## extensão 
    nome = arquivo.name.lower()
    if nome.endswith(".csv"):
        df = pd.read_csv(arquivo.name)
    elif nome.endswith(".xlsx"):
        df = pd.read_excel(arquivo.name)
    else:
        return "Formato não suportado."

    ## checagem das pasta de destino
    saida = chat_input_dir_check()[1] / Path("transacoes.jsonl")

    ## para jsonl
    with open(saida, "w", encoding="utf-8") as f:
        for _, linha in df.iterrows():
            f.write(
                json.dumps({"type": "transaction",
                            "role": "context",
                            "content": linha.to_dict()},
                    ensure_ascii=False
                ) + "\n"
            )

    return "Arquivo recebido com sucesso! Faça perguntas sobre a planilha no chat a seguir:"


def resumir_transacoes(transacoes):

    registros = []

    for t in transacoes:
        registros.append(
            t["content"]
        )

    if not registros:
        return {}

    df = pd.DataFrame(registros)

    summary = {}

    ## nomes das colunas
    summary["colunas"] = list(df.columns)

    ## quantidade de transações
    summary["total_registros"] = len(df)

    ## tipos das colunas
    summary["tipos"] = {
        c:str(df[c].dtype)
        for c in df.columns
    }

    ## gastos relevantes
    summary["outliers_numericos"] = {}

    ## perfil por tipo de coluna
    summary["resumo_colunas"] = {}
    for col in df.columns:
        # numéricas
        if pd.api.types.is_numeric_dtype(
            df[col]
        ):
            top_rows = (
                df.nlargest(
                    10,
                    col
                )
                .to_dict(
                    orient="records"
                )
            )
            summary["resumo_colunas"][col] = {
             "soma":
             float(df[col].sum()),
             "media":
             float(df[col].mean()),
             "min":
             float(df[col].min()),
             "max":
             float(df[col].max())
            }
            summary["outliers_numericos"][col] = top_rows
        # texto/categórica
        else:
            summary["resumo_colunas"][col] = {
              "valores_unicos":
              int(
               df[col].nunique()
              ),
              "exemplos":
              list(
               df[col]
               .dropna()
               .astype(str)
               .head(5)
              )
            }

    
    return summary


def montar_prompt_final(
        nova_pergunta=None,
        arquivo_saida_string="prompt_final.jsonl"
        ):
    sys_prompt_dir_check()
    input_dir, input_components_dir = chat_input_dir_check()
    system_prompt_path = input_components_dir / Path("system_prompt.jsonl")
    transacoes_path = input_components_dir / Path("transacoes.jsonl")
    chat_log_path = input_components_dir / Path("chat_log.jsonl")
    prompt_final_path = input_dir / Path(arquivo_saida_string)
    registros = []

    ## system_prompt
    with open(
        system_prompt_path,
        "r",
        encoding="utf-8"
    ) as f:
        for linha in f:
            registros.append(
                json.loads(linha)
            )

    ## transacoes
    if transacoes_path.is_file():
        transacoes = []
        with open(
            transacoes_path,
            "r",
            encoding="utf-8" ) as f:
            for linha in f:
                transacoes.append(
                    json.loads(linha)
                )
        summary = resumir_transacoes(
            transacoes
        )
        registros.append({
        "type":"financial_summary",
        "role":"context",
        "content":summary
        })

    ## últimas 10 mensagens do chat_log
    if chat_log_path.is_file():
        with open(
            chat_log_path,
            "r",
            encoding="utf-8"
        ) as f:
            linhas_chat = f.readlines()
        ultimos_5_pares = linhas_chat[-10:]
        for linha in ultimos_5_pares:
            registros.append(
                json.loads(linha)
            )

    ##  nova_pergunta
    if nova_pergunta:
        registros.append({
            "type":"chat_message",
            "role":"user",
            "content":nova_pergunta
        })

    ## prompt final
    with open(
        prompt_final_path,
        "w",
        encoding="utf-8"
    ) as f:
        for registro in registros:
            f.write(
                json.dumps(
                    registro,
                    ensure_ascii=False
                ) + "\n"
            )
    
    return prompt_final_path


def rodar_ia():
    model, tokenizer, best_device = import_model()

    # Defining a custom stopping criteria class for the model's text generation.
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = [2]  # IDs of tokens where the generation should stop.
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
                    return True
            return False


    # Function to generate model predictions.
    def predict(message, history):
        MAX_HISTORY = 5
        # history sendo o histórico completo
        history = history[-MAX_HISTORY:]
        stop = StopOnTokens()
        prompt_final_path = montar_prompt_final(nova_pergunta=message)
        messages = []
        with open(
            prompt_final_path,
            "r",
            encoding="utf-8"
        ) as f:
            for linha in f:
                registro = json.loads(linha)
                ## roles válidas para o chat template
                if registro["role"] in [
                    "system",
                    "user",
                    "assistant"
                ]:
                    messages.append({
                        "role": registro["role"],
                        "content": registro["content"]
                    })
                # transações/contexto
                elif registro["role"] == "context":
                    messages.append({
                    "role":"system",
                    "content":
                    f"Dado financeiro: {registro['content']}"
                    })
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([prompt], return_tensors="pt").to(best_device)
        streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop])
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()  # Starting the generation in a separate thread.
        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
                break
            yield partial_message
            
        salvar_interacao(message, partial_message)
        print(f"> tokens da instrução + contexto: {len(tokenizer(prompt)["input_ids"])}") ## todo retirar fora de test

    with gr.Blocks() as demo:
        gr.Markdown("# Assistente Financeiro")
        with gr.Column():
            arquivo = gr.File(
                label='Insira aqui uma planilha em CSV ou XLSX para ser analisada'
            )
            status = gr.Textbox(
                label='Status'
            )
            # AUTO dispara quando arquivo muda
            arquivo.change(
                fn=upload_para_jsonl,
                inputs=arquivo,
                outputs=status
            )

        gr.Markdown('---')

        # Chat embaixo
        gr.ChatInterface(
            fn=predict,
            title='Agente de Gastos',
            description="Pergunte algo para o Agente de Gastos",
            examples=['Poderia me dar dicas do dia a dia sobre economia?', 'Onde começar a economizar nos meus gastos?']
        )
    demo.launch()
    
if __name__ == '__main__':
    load()
    rodar_ia()
