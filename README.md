# auto-jailbreak-llm

Exploring methods to automatically jailbreak LLMs.

To get started, install requirements from `requirements.txt`

## Training

To fine-tune an open-source HuggingFace model to be a jailbreak prompt generator, use the `finetune-script.py` file. A demo of the fine-tuning process can be found in the jupyter notebook.

## Inference

To run inference locally, first install llama-cpp.

```
brew install llama-cpp
```

Then, start an OpenAI compatible server. Replace the --hf-repo and --hf-file with directories/GGUF files of your choice.

```
llama-server --hf-repo jasonhwan/phi3-redteamer --hf-file phi3-3.8B-redteamer.gguf
```

Once the server is running, you can run inference on localhost or via the bash script provided in `inference.sh`. Just run:

```
./inference.sh "Your vanilla prompt" 128 // replace 128 with any integer representing the max tokens you want to generate
```
