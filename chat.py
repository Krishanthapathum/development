from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

print("Start chatting with the bot (type 'exit' to stop)!")
chat_history_ids = None

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Encode the input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append to chat history or start fresh
    if chat_history_ids is None:
        bot_input_ids = new_input_ids
    else:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        top_k=50
    )

    # Decode and print the response
    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {reply}")
