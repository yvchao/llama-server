# A simple python wrapper of the llama.cpp server

## Installation
- [Llama server ] Execute `bash ./install-llama.sh` to install the [llama.cpp](https://github.com/ggerganov/llama.cpp) server.
- [Dependencies ] Install python dependencies via `pip install -r requirements.txt`.

## Usage

### Create a llama server
```python
config = {
    "alias": "LLaVA 1.5",
    "model": "path to model.gguf", # model needs to be converted to formats that are compatible with llama.cpp
    "multimodal_projector": "path to projector.gguf",
    "server_exe": "path to llama_server",
    "system_prompt": "path to system prompt file",
    "prefix": "User:",
    "suffix": "\nAssistant:",
}
slots = 5 # max number of parallel inference tasks
llm = LlamaServer(config, context_size=4096, slots=slots)
```

### Start and stop the server
```python
# start server
proc = await llm.start_server()
# stop server
llm.stop_server()
```

### Perform batch inference
```python
prompts = [
    "How to build a website?",
    "Teach me how to calculate $\pi$?"
]
results, _ = await llm.batch_query(prompts)
for prompt, result in zip(prompts, results):
    answer = result["answer"].strip()
    print(f"Q: {prompt}\nA: {answer}\n")
```

### Perform multimodal inference
```python
prompt = "Please describe concisely the content in the above figure."
result, image = await llm.query(prompt, "./figures/earth.jpg")
print(result["answer"].strip())
```
