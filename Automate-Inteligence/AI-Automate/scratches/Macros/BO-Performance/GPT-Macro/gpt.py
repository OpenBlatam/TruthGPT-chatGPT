from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Prepare the dataset
train_dataset = TextDataset(
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
    file_path="path_to_train_dataset",
    block_size=128
)
eval_dataset = TextDataset(
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
    file_path="path_to_eval_dataset",
    block_size=128
)

# Prepare the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2'), mlm=False,
)

# Load the GPT model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

# Create the Trainer and fine-tune
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()