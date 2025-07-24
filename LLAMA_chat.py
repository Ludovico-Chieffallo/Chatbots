from llama_cpp import Llama

# Carica il modello
llm = Llama(
    model_path="/Users/ludovico_chieffallo/Downloads/llama-2-7b-chat.Q2_K.gguf",
    n_ctx=2048,
    n_threads=4,
    n_batch=512,
    verbose=False
)

# Prompt iniziale con istruzioni
chat_history = [
    {
    "role": "system",
    "content": (
        "Sei un assistente esperto nella creazione di chatbot in Python, specializzato nell'uso dei modelli LLaMA "
        "in formato GGUF. Parli sempre in italiano, fornisci spiegazioni chiare e dettagliate di almeno cinque frasi, "
        "e scrivi codice Python ben commentato ogni volta che serve. Il tuo tono è professionale ma amichevole."
    )
}
]

# Funzione per costruire il prompt completo
def build_prompt(chat_history):
    prompt = ""
    if chat_history and chat_history[0]["role"] == "system":
        system_prompt = chat_history[0]["content"]
        prompt += f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        history = chat_history[1:]
    else:
        prompt += "[INST] "
        history = chat_history

    for i in range(0, len(history), 2):
        user_turn = history[i]["content"]
        prompt += f"{user_turn} [/INST]\n"
        if i + 1 < len(history):
            assistant_turn = history[i + 1]["content"]
            prompt += f"{assistant_turn}\n[INST] "

    return prompt.strip()


# Chat loop
print("🤖 Chatbot attivo! Scrivi 'esci' per terminare.\n")

while True:
    user_input = input("Tu: ")
    if user_input.lower() in ["esci", "quit", "exit"]:
        print("Fine della chat. A presto!")
        break

    # Aggiungi input dell'utente alla storia
    chat_history.append({"role": "user", "content": user_input})
    full_prompt = build_prompt(chat_history)

    # Genera la risposta con parametri ottimizzati
    output = llm(
        full_prompt,
        max_tokens=600,
        temperature=0.8,
        top_p=0.95,
        repeat_penalty=1.1
        # Nota: non mettiamo stop token perché con modelli base possono tagliare male
    )

    # Estrai il testo generato
    response = output["choices"][0]["text"].strip()

    # Mostra e salva la risposta
    print("AI:", response)
    chat_history.append({"role": "assistant", "content": response})
