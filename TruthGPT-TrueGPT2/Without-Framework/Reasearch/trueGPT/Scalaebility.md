# Scale

# Description

Create tranformers input and produce a high-dimensional vector as output.
```
import torch
import transformers

model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

prompts = [
    "I need some inspiration to write a poem about love.",
    "Can you help me come up with a creative name for my new business?",
    "I'm feeling down and could use some words of encouragement.",
    "I want to plan a fun surprise date for my partner. Any ideas?",
]

labels = [
    "Love poem:",
    "Business name:",
    "Words of encouragement:",
    "Surprise date ideas:",
]

for prompt, label in zip(prompts, labels):
    # Encode prompt and generate text
    encoded_prompt = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    generated = model.generate(encoded_prompt, max_length=100, do_sample=True)
    decoded_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    # Print output
    print(f"{label} {decoded_text}")

```
The generate method is called on the model to generate text based on the given input prompt. The max_length parameter limits the maximum number of tokens in the generated text, and do_sample=True enables random sampling of the model's output to add variety to the generated text.
