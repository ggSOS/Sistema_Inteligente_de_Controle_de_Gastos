from test_import_pre_trained_model import import_model
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread


import json

def salvar_interacao(user_msg, bot_msg, filename="chat_log.jsonl"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps({"role": "user", "content": user_msg}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"role": "assistant", "content": bot_msg}, ensure_ascii=False) + "\n")


def extrair_texto(content):
    if isinstance(content, list):
        return content[0]["text"]
    return content


def rodar_ia():
    system_prompt = "Você é um assistente útil e amigável. Sua base de conhecimento são as ciências econômicas. Crie uma resposta sempre com menos de 250 tokens e em Português do Brasil."

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
        history = history[-MAX_HISTORY:]
        stop = StopOnTokens()

        # Formatting the input for the model.
        chat_messages = [
            {"role": "system", "content": system_prompt}
        ]
        # adicionar histórico
        for item in history:
            chat_messages.append({
                "role": item["role"],
                "content": extrair_texto(item["content"])
            })
        # adicionar mensagem atual
        chat_messages.append({
            "role": "user",
            "content": message
        })
        # context = f"{system_prompt}\n{messages}"
        # salvar_conversa(messages)
        # print(f"tokens do contexto: {len(tokenizer(messages)["input_ids"])}")

        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"tokens do contexto: {len(tokenizer(prompt)["input_ids"])}")
        model_inputs = tokenizer([prompt], return_tensors="pt").to(best_device)
        streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=256,
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


    # Setting up the Gradio chat interface.
    gr.ChatInterface(predict,
                    title="Tinyllama_chatBot",
                    description="Pergunte algo para Tiny llama",
                    examples=['Poderia me dar dicas do dia a dia sobre economia?', 'Não entendo nada de economia']
                    ).launch()  # Launching the web interface.
    
if __name__ == '__main__':
    rodar_ia()
