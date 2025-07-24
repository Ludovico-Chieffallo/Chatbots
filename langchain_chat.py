from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

model_path = "/Users/ludovico_chieffallo/Documents/chatbot_llama/llama-2-7b-chat.Q2_K.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    verbose=True
)

# Prompt template corretto per LLaMA 2
template = "[INST] {input} [/INST]"

prompt = PromptTemplate(
    input_variables=["input"],
    template=template
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

print("Chatbot pronto! Digita 'esci' per terminare.\n")

while True:
    user_input = input("Tu: ")
    if user_input.lower() == "esci":
        break

    response = llm_chain.invoke({"input": user_input})
    print("Bot:", response["text"].strip())
