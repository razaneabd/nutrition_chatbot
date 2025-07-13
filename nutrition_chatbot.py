import gradio as gr
import os
import json
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
# from langchain.prompts import PromptTemplate
import json
from dotenv import load_dotenv


# get prompt templates
with open("prompts.json", "r") as file:
    data = json.load(file)
    """
    The contextualize_q_system_prompt:
    Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    contextualize_q_system_prompt = data["contextualize_q_system_prompt"]

    """
    You are a nutrition guide  which means you chat with the user and answer questions about nutrition and lifestyle in a lovely way following the following rules:
    1. Only use the provided context and chat history to answer the question. Do not use any external sources.
    2. If you don't know the answer, don't try to make up an answer. 
    Just say "I can't find the final answer but you may want to check the following links".
    3.If you find the answer, write the answer in a concise way with five sentences maximum unless it is a recipe you can write longer but bullet
    4. If the question is not about the topic you provide your answer by: 
    I am a nutrition guide feel free to ask me about recipes, nutrition facts....
    Context:
    {context}

    Answer:"  
    """
    nutrition_system_prompt = data["nutrition_system_prompt"]


# load API-KEYS
load_dotenv()
os.environ["GROQ_API_KEY"] = "**"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "**"


huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # normalize embeddings to unit length
)

# load vector store through vector database
if os.path.exists("vector_database") and len(os.listdir("vector_database")):
    vector_store = FAISS.load_local(
        "vector_database",
        huggingface_embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    loader = PyPDFDirectoryLoader("nutrition-pdfs")#loads PDF from nutrition-pdfs 
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)#splits them into chunks
    #text_splitter = SentenceTextSplitter(chunk_size=5)  # 5 sentences per chunk
    splitted_pages = text_splitter.split_documents(pages)
    vector_store = FAISS.from_documents(splitted_pages, huggingface_embeddings)
    vector_store.save_local("vector_database")

# define retriever
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3, "score_threshold": 0.5}
)

# define llm
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
#from openai import OpenAI llama-3.3-70b-versatile
#llm = OpenAI(model_name="gpt-3", temperature=0.3)

"""
define the contextualize_q_prompt:
contextualize prompt tells the model that it should take both the chat history and the user's latest question, 
and if the question relies on the previous context, it should be rephrased into a standalone question. 
If the question doesn't need rephrasing, it is returned as-is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

"""
create_history_aware_retriever function works by combining the user’s current question 
with the prior chat history to improve the relevance of the retrieved context
"""
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

"""
create template for the question-answering prompt:
The system message sets the instructions for answering questions.
MessagesPlaceholder("chat_history") represents where the chat history will go in the prompt.
("{input}") is where the actual user question will be placed.
"""
nutrition_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", nutrition_system_prompt),
        MessagesPlaceholder("chat_history"),  #  chat history
        ("human", "{input}"),  # User-provided input
    ]
)

"""
create a chain that uses the language model (llm) and the question-answering prompt (qa_prompt) to generate answers.
It takes the context retrieved by the retriever and processes it using the qa_prompt to provide the answer.
"""
question_answer_chain = create_stuff_documents_chain(llm, nutrition_prompt)

"""
combine the history-aware retriever (history_aware_retriever) and the question-answer chain (question_answer_chain) into one unified chain.
This chain will handle both retrieving relevant context and then answering the user’s query using that context.
"""
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def chat_with_model(history, new_message, chat_history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_message, assistant_response in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_response})

    messages.append({"role": "user", "content": new_message})
    response = rag_chain.invoke({"input": new_message, "chat_history": chat_history})

    assistant_message = response["answer"]
    chat_history.extend([{"role": "user", "content": new_message}, {"role": "assistant", "content": assistant_message}])
    history.append({"role": "user", "content": new_message})
    history.append({"role": "assistant", "content": assistant_message})   
    
    return history, ""


def gradio_chat_app():
    with gr.Blocks() as app:
        chat_history = gr.State([])
        gr.Markdown("# The nutrition assistant Chat Bot")
        gr.Markdown(
            "Chat with me about nutrition,lifestyle, healthy recipes...."
        )

        chatbot = gr.Chatbot(label="Chat Interface", type='messages')
        user_input = gr.Textbox(
            label="your message", placeholder="Type something ...", lines=2
        )
        send_button = gr.Button("send")

        def clear_chat():

            return [], "", []

        clear_button = gr.Button("Clear chat")

        send_button.click(
            fn=chat_with_model,
            inputs=[chatbot, user_input, chat_history],
            outputs=[chatbot, user_input],
        )
        clear_button.click(
            fn=clear_chat, inputs=[], outputs=[chatbot, user_input, chat_history]
        )

    return app


if __name__ == "__main__":
    app = gradio_chat_app()
    app.launch(share=True)