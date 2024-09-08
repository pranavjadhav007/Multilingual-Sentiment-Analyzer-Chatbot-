import pickle
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

store = {}

sentiment_model = load_model('sentiment_LSTM_model.h5')

with open('sentiment_LSTM_tokenizer.pickle', 'rb') as handle:
    sentiment_tokenizer = pickle.load(handle)

class Sentiment_chatbot:
    model = ChatGroq(model="llama3-8b-8192")

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a human friendly assistant and answer in less words. You answer in a friendly manner. You are like a friend to human and understand the sentiment or the emotion of the chat. This is the sentiment of the user chat {sentiment}. So based on this sentiment, answer the user so it seems like a human interaction",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | model

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )
    config = {"configurable": {"session_id": "mychat"}}

    def final_bot_ans(self, tex):
        response = self.with_message_history.invoke(
            {"messages": [HumanMessage(content=tex)], "sentiment": "negative"},
            config=self.config,
        )
        return response.content

    def preprocess(self, tex):
        stem = PorterStemmer()
        stopword = stopwords.words("english")
        tex = re.sub('<.*?>', "", tex)
        tex = re.sub(r'\S*http\S+', '', tex)
        tex = re.sub(r'[^a-zA-Z\s]', "", tex)
        tex = word_tokenize(tex)
        tex = [stem.stem(word) for word in tex if word not in stopword]
        return " ".join(tex)

    def sentiment_input_predict(self, processed_input):
        input_sequences = sentiment_tokenizer.texts_to_sequences([processed_input])
        padded_input_sequences = pad_sequences(input_sequences)
        predictions = sentiment_model.predict(padded_input_sequences)
        predicted_labels = np.argmax(predictions, axis=1)
        if predicted_labels[0] == 0:
            return "NEGATIVE"
        elif predicted_labels[0] == 1:
            return "NEUTRAL"
        else:
            return "POSITIVE"

    def generate_final_answer(self, tex):
        processed_input = self.preprocess(tex)
        sentiment = self.sentiment_input_predict(processed_input)
        bot_reply = self.final_bot_ans(processed_input)
        return bot_reply, sentiment

# Example usage
chatbot_instance = Sentiment_chatbot()
output_answer, sentiment = chatbot_instance.generate_final_answer("I don't feel good")
print(output_answer, sentiment)

        
    # user_inp=input("Whats in ur mind: ")
    # while(user_inp != 'bye'):
    # print(sentiment)
    # print("\n Bot: ",give_ans(processed_input))
    # user_inp = input("\nUser: ")




