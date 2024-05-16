import openai

openai.api_key = "sk-proj-w2cneIgr3XmsgiPs5P5ET3BlbkFJZJ3q9njclxn8PIXiIv2G"

def tokenize(text):
    prompt = f"import re\n\ndef tokenize(text):\n    # Replace all non-alphanumeric characters with a space\n    text = re.sub(r'\W+', ' ', text)\n\n    # Convert to lowercase and split on whitespace\n    tokens = text.lower().split()\n\n    return tokens\n\ntokenize(\"{text}\")"
    completions = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    tokens = completions.choices[0].text.strip().split("\n")
    tokens = [token.strip() for token in tokens if token != '']
    return tokens
