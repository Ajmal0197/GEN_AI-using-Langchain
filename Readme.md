Follow:
https://www.youtube.com/watch?v=swCPic00c30

https://python.langchain.com/docs/how_to/


### Step to run:
1. Installation https://dev.to/ajmal_hasan/setting-up-a-conda-environment-for-your-python-projects-251d
2. For test we can check in LangSmith dashboard (https://smith.langchain.com/)

------------

Theory:

### **Ingestion: Preparing the Document**

1. **Loading the Document**: Initially, the document is loaded into the system.  
   *LangChain Concept Used*: **Document Loaders** (e.g., `WebBaseLoader`, `PDFLoader`, etc.)

2. **Splitting the Document**: Large documents are divided into smaller, manageable chunks.  
   *LangChain Concept Used*: **Text Splitters** (e.g., `RecursiveCharacterTextSplitter`)

3. **Creating Embeddings**: Each chunk is converted into vector embeddings representing the document's content in a high-dimensional space.  
   *LangChain Concept Used*: **Embeddings** (e.g., `OpenAIEmbeddings`, `HuggingFaceEmbeddings`)

4. **Storing in a Vector Store**: These embeddings are stored in a vector database, making the document searchable.  
   *LangChain Concept Used*: **Vector Stores** (e.g., `FAISS`, `Pinecone`, `Weaviate`)

---

### **Generation: Answering the Question**

1. **Accepts a User's Question**: The system takes in queries from users.  
   *LangChain Concept Used*: **Retrievers** (e.g., `VectorStoreRetriever`)

2. **Finds Relevant Content**: It identifies the most relevant section(s) of the document based on the query.  
   *LangChain Concept Used*: **Retrievers and Chains** (e.g., `RetrievalQAChain`)

3. **Generates an Answer**: Using the LLM, the system generates a precise answer to the user's question based on the identified document section.  
   *LangChain Concept Used*: **LLMs and Chains** (e.g., `OpenAI`, `ConversationalRetrievalChain`)  

LangChain provides seamless integration of these concepts to manage both ingestion and generation phases effectively.