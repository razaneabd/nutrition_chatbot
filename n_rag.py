files = [
    "https://aubmc.org.lb/Documents/publications/patient_info/Nutrition-tips-cancer.pdf",
    "https://www.nyc.gov/assets/doh/downloads/pdf/cdp/kot-simple-steps.pdf",
    "https://www.lambethtogether.net/wp-content/uploads/2023/12/Thrive-for-Life-Healthy-Eating-and-Living-Guide.pdf",
    "https://sunnybrook.ca/uploads/YNC_guidelines.pdf",
    "https://www.cancer.org/content/dam/cancer-org/cancer-control/en/booklets-flyers/nutrition-for-the-patient-with-cancer-during-treatment.pdf",
    "https://jbsnourishwell.com/wp-content/uploads/2020/08/Immune-Boosting-Recipes-Dorine-Lam.pdf"
]

os.makedirs('nutrition-pdfs',exist_ok=True)

import os
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

for url in files:
    try:
        file_path = os.path.join("nutrition-pdfs", url.rpartition("/")[2])
        urlretrieve(url, file_path)  # retrieves pdfs from the web
    except (URLError, HTTPError) as e:
        print(f"Failed to access {url}: {e}")


loader = PyPDFDirectoryLoader("nutrition-pdfs") #it loads the directory that contains the pdfs

listed_docs = loader.load() 
len(listed_docs[0].page_content)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)

chunked_docs = text_splitter.split_documents(listed_docs)
chunked_docs[0]
len(chunked_docs[0].page_content)
avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs])//len(docs)
avg_char_listed_docs = avg_doc_length(listed_docs)
avg_char_chunked_docs = avg_doc_length(chunked_docs)

avg_char_listed_docs,avg_char_chunked_docs


huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # one of the most popular embedding models in hugging face
    model_kwargs={"device": "cpu"},
    encode_kwargs={"noramlize_embeddings": True},  # normalize embeddings to unit length
)

huggingface_embeddings.embed_query("healthy lifestyle")

pinecone_api = "**"
pc = Pinecone(api_key=pinecone_api)


#an index is a data structure that enables efficient similarity search and retrieval of high-dimensional vectors
index_name = "langchain-test-index"  
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
pc.describe_index(index_name)
pine_vectorstore = PineconeVectorStore(index=index, embedding=huggingface_embeddings)

uuids = [str(uuid4()) for _ in range(len(chunked_docs))]

pine_vectorstore.add_documents(documents=chunked_docs, ids=uuids)

vectorstore = FAISS.from_documents(chunked_docs,huggingface_embeddings)

query = "give me Immune boosting recipes"
results = pine_vectorstore.similarity_search(query,k=5)
for res in results:
	print(f"{res.page_content} [{res.metadata}]")
	print("*****")
     
#apply similarity search to the query using vectorstore.similarity_search()
relevant_docs = vectorstore.similarity_search(query)
print(relevant_docs[0].page_content)

pinecone_retriever = pine_vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k":3,"score_threshold" : 0.6})
retriever = vectorstore.as_retriever(search_type="similarity" , search_kwargs={"k" : 3})

access_token = "**"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = access_token

hf = HuggingFaceHub(
    repo_id='mistralai/Mistral-Nemo-Instruct-2407',#'mistralai/Mistral-7B-v0.1'
    model_kwargs = {"temperature" : 0.1 , "max_length" : 900}
)

output = hf.invoke(query)
print(output)


prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
. If you find the answer, write the answer in a concise way with five sentences maximum.
. Only use the provided context to answer the question. Do not use any external sources. 
{context}

Question: {question}

Helpful Answer:
"""

PROMPT = PromptTemplate(
 template=prompt_template, input_variables=["context", "question"]
)

pinecone_retievalQA = RetrievalQA.from_chain_type(
    llm=hf,
    chain_type="stuff",
    retriever = pinecone_retriever,
    return_source_documents = True,
    chain_type_kwargs={"prompt":PROMPT}
)

#create the chain
retrievalQA = RetrievalQA.from_chain_type(
    llm=hf, 
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs = {"prompt" : PROMPT}
)

pinecone_result = pinecone_retievalQA.invoke({"query":query})
print(pinecone_result['result'])

result = retrievalQA.invoke({"query":query})
print(result)

print(result['result'])

relevant_docs = result['source_documents']
print(f'There are {len(relevant_docs)} documents retrieved which are relevant to the query.')
print("*" * 100)
for i, doc in enumerate(relevant_docs):
    print(f"Relevant Document #{i+1}:\nSource file: {doc.metadata['source']}, Page: {doc.metadata['page']}\nContent: {doc.page_content}")
    print("-"*100)
    print(f'There are {len(relevant_docs)} documents retrieved which are relevant to the query.')