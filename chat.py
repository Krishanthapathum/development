from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create a ChatBot instance
chatbot = ChatBot('MySimpleBot')

# Set up the trainer
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot with the English corpus
trainer.train("chatterbot.corpus.english")

# Chat loop to interact with the bot
print("Start chatting with the bot (type 'exit' to stop)!")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = chatbot.get_response(user_input)
    print("Bot:", response)
