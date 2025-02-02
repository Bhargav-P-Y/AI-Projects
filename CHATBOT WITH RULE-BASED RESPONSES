# Import necessary modules
import nltk
import re
import wikipediaapi
from nltk.chat.util import Chat, reflections

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# Initialize Wikipedia API with a proper User-Agent
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent="MyChatbot/1.0 (contact: yellambalse.bhargav@gmail.com)",
    language='en'
)

# Function to fetch Wikipedia summary
def get_wikipedia_summary(topic):
    page = wiki_wiki.page(topic)
    if page.exists():
        return page.summary[:500] + "..."  # Limiting summary to 500 characters
    else:
        return f"Sorry, I couldn't find information on '{topic}'."

# Define patterns and responses
pairs = [
    [r"my name is (.*)", ["Hello %1, how can I assist you today?", ]],
    [r"hi|hey|hello",
     ["Hello, how can I help you?", "Hey there! What can I do for you?", "Hi! How can I assist you today?"]],
    [r"what is your name?", ["I am a chatbot created to assist you. You can call me Chatbot.", ]],
    [r"how are you?", ["I'm a bot, so I don't have feelings, but I'm here to help you!", ]],
    [r"can you help me with (.*)", ["Sure, I can help you with %1. Please provide more details.", ]],
    [r"sorry (.*)", ["It's okay. How can I assist you?", ]],
    [r"thank you|thanks", ["You're welcome!", "No problem!", "Happy to help!"]],
    [r"quit", ["Bye! Have a great day!", "Goodbye!"]],
    [r"tell me about (.*)|explain about (.*)", [lambda match: get_wikipedia_summary(match.group(1))]],  # Dynamic response using Wikipedia
    [r"(.*)", ["I'm sorry, I don't understand that. Can you rephrase?", "Could you please elaborate on that?"]]
]

# Define the chatbot class
class RuleBasedChatbot:
    def __init__(self, pairs):
        self.chat = Chat(pairs, reflections)

    def respond(self, user_input):
        for pattern, responses in pairs:
            match = re.match(pattern, user_input, re.IGNORECASE)
            if match:
                if callable(responses[0]):  # Check if response is a function (for Wikipedia queries)
                    return responses[0](match)
                else:
                    return self.chat.respond(user_input)
        return "I'm sorry, I don't understand that."


# Initialize the chatbot
chatbot = RuleBasedChatbot(pairs)


# Function to chat with the bot
def chat_with_bot():
    print("Hi, I'm your chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Bye! Have a great day!")
            break
        response = chatbot.respond(user_input)
        print(f"Chatbot: {response}")


# Start chatting with the bot
chat_with_bot()
