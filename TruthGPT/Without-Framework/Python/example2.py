from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the model and tokenizer with gpt-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# User input query
user_query = "What is AI?"

# Encode user input and add end of string token
input_ids = tokenizer.encode(user_query + tokenizer.eos_token, return_tensors='pt')

# Generate a response
output = model.generate(input_ids, max_length=1000,
                        pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

print(response)
