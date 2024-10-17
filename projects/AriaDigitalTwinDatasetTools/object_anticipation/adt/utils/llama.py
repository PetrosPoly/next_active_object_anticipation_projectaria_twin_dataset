from llama_cpp import Llama

path_to_the_model = "path/to/the/model.gguf"

llm  = Llama(model_path = "./Llama3-gguf-unsloth.Q4_K_M.gguf", n_gpu_layers=-1)   # n_gpu_layers=-1) to load the model to GPU 

output = llm (
    "Q: Name 5 species of llamas? A:",
    max_tokens = 32, # limit the number of tokens
    stop = ["Q; ", "\n"], # stops generation when these tokens are generated so the model doesn't rample w
)

print(output['choices'][0]['text'].split('A:')[1])

# llm  = Llama(model_path = path_to_the_model)
# output = llm("Where do Llama lives?")
# print(output)
# output_text = output['choices'][0]['text']
# print(output_text)
