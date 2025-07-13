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
from dotenv import load_dotenv


# prompts
with open("prompts.json", "r") as file:
    data = json.load(file)
    contextualize_q_system_prompt = data["contextualize_q_system_prompt"]
    nutrition_system_prompt = data["nutrition_system_prompt"]


load_dotenv()
os.environ["GROQ_API_KEY"] = "**"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "**"


huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # normalize to unit length
)

# load vector store
if os.path.exists("vector_database") and len(os.listdir("vector_database")):
    vector_store = FAISS.load_local(
        "vector_database",
        huggingface_embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    loader = PyPDFDirectoryLoader("nutrition-pdfs")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40)
    #text_splitter = SentenceTextSplitter(chunk_size=5)  # 5 sentences per chunk
    splitted_pages = text_splitter.split_documents(pages)
    vector_store = FAISS.from_documents(splitted_pages, huggingface_embeddings)
    vector_store.save_local("vector_database")

# retriever
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3, "score_threshold": 0.5}
)

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#combine (users current quest) with (prior chat history) 
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


#create template for the prompt:
nutrition_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", nutrition_system_prompt),
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}"),  # User input
    ]
)


#chain: llm + qa_prompt to generate answers.
question_answer_chain = create_stuff_documents_chain(llm, nutrition_prompt)

# combine (history aware retrieevr) + (qa prompt) into one chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def chat_with_model(history, new_message, chat_history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_message, assistant_response in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_response})

    messages.append({"role": "user", "content": new_message})
    response = rag_chain.invoke({"input": new_message, "chat_history": chat_history})

    assistant_message = response["answer"]
    chat_history.extend([HumanMessage(content=new_message), response["answer"]])
    history.append((new_message, assistant_message))
    return history, ""


def gradio_chat_app():
    with gr.Blocks() as app:
        chat_history = gr.State([])
        gr.Markdown("# The nutrition Chat Bot")
        gr.Markdown(
            "Chat with nutrition assistant about recipes health lifestyle..."
        )

        chatbot = gr.Chatbot(label="Chat Interface")
        user_input = gr.Textbox(
            label="your message", placeholder="Type something ...", lines=1
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
