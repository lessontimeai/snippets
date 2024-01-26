from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM
import torch
import tqdm
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained("gpt2")
text = "def fibonacci(n=10):"

for i in tqdm.tqdm(range(100)):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    likelihoods = output['logits'][0,-1]
    top = torch.argsort(-likelihoods)[:10]
    likelihoods = likelihoods[top] - torch.max(likelihoods[top])
    likelihoods = torch.exp(likelihoods)**1
    likelihoods = likelihoods/torch.sum(likelihoods)
    text += tokenizer.decode(top[likelihoods.multinomial(1)])
print(text)