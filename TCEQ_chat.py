import gradio as gr
from langchain.schema import AIMessage
import numpy as np
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from langchain.chat_models import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from config import ANTHROPIC_API_KEY

global rule_path
rule_path = './rule_pkls/217.pkl'

class ChatbotApp:
    def __init__(self):
        self.history = []
        self.embedding_model = "text-embedding-ada-002"
        self.df = pd.read_pickle(rule_path)
        self.chat = ChatAnthropic(model='claude-v1.3', max_tokens_to_sample=512, anthropic_api_key=ANTHROPIC_API_KEY)
        text = '''As a specialist in reviewing rules set by the Texas Commission on Environmental Quality (TCEQ), your role involves addressing user inquiries regarding these regulations. You'll use the information contained in the relevant rules message to offer comprehensive responses to each question, ensuring that you specify the relevant rule number and quote the specific part of the rule in question. While previous chat history can offer context, your main goal is to deliver the most effective and accurate response to the most recent user query.'''
        self.system_message = SystemMessage(content=text) # Initialize with your system message

    def get_embedding(self, message):
        from openai.embeddings_utils import get_embedding
        user_input = np.array(get_embedding(message, engine=self.embedding_model))
        user_input = user_input.reshape(1, -1)
        return user_input

    def predict(self, user_message):
        messages = [self.system_message]  # Initialize with your system message
        for message in self.history:
            if message is None:
                break
            ai_message = AIMessage(content=message[1])
            human_message = HumanMessage(content=message[0])
            messages.extend([human_message, ai_message])

        user_message = HumanMessage(content=user_message)
        user_input = self.get_embedding(user_message.content)
        self.df["cos_dist"] = cdist(np.stack(self.df.embeddings), user_input, metric="cosine")
        self.df.sort_values("cos_dist", inplace=True)

        res = 'Use the following relevant rules to answer the user\'s question:\n\n'
        for row in self.df.head(10).index:
            res += f"Rule: {row}\nText: {self.df.loc[row, 'rule']}\n---\n"
        context_message = AIMessage(content=res)

        messages.extend([context_message, user_message])

        response = self.chat(messages).content.strip()  # You need to define the 'chat' function or use the appropriate model
        self.history.append([user_message.content, response])
        return self.history
    def run(self):      
        with gr.Blocks() as demo: 

            # creates a new Chatbot instance and assigns it to the variable chatbot.
            chatbot = gr.Chatbot() 

            # creates a new Row component, which is a container for other components.
            with gr.Row(): 
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
            txt.submit(self.predict, txt, chatbot) 
            txt.submit(None, None, txt, _js="() => {''}")
            with gr.Row():
                save_chat_submit = gr.Button(value="Save Chat History").style(full_width=False)
                save_chat_submit.click(self.save_history)
                reset_chat = gr.Button(value="Reset Chat").style(full_width=False)
                reset_chat.click(self.restart)

        demo.launch() 

    def restart(self):
        self.__init__()
    def save_history(self):
        '''This methods takes the conversation history and saves in a pickled file with information about the time and date of the conversation.
        '''        
        import pickle
        import datetime
        now = datetime.datetime.now()
        filename = 'conversation_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.history, f)
        print('Saved conversation to ' + filename)
def main():
    chatbot_app = ChatbotApp()
    chatbot_app.run()
if __name__ == '__main__':
    main()   