import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_gpt_model(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def generate_text(tokenizer, model, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    model_name = 'gpt2'  # Replace with the desired GPT model name
    prompt = 'Once upon a time'  # Replace with the desired text prompt

    tokenizer, model = load_gpt_model(model_name)
    generated_text = generate_text(tokenizer, model, prompt)
    print(generated_text)

if __name__ == '__main__':
    main()
