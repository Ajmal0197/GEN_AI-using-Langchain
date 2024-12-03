# ---LangChain As Api For Deployment---

# ref: https://python.langchain.com/docs/langserve/

#  Run in terminal 1: `streamlit run app.py`
#  Open : http://localhost:8000/docs to open Swagger
#  Run in terminal 2: `streamlit run client.py`

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # Access GPT models
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model=ChatOpenAI()

##ollama llama
llm=Ollama(model="llama3.2")

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

add_routes(
    app,
    prompt1|model,
    path="/essay"
)

add_routes(
    app,
    prompt2|llm,
    path="/poem"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)


# ### **Key Functionalities**

# 1. **API Server with FastAPI**:
#    - Creates an API server using FastAPI to expose endpoints for AI tasks.

# 2. **Environment Variable Management**:
#    - Loads sensitive keys like `OPENAI_API_KEY` securely from a `.env` file using `dotenv`.

# 3. **Integration with LangChain**:
#    - Uses LangChain’s `ChatOpenAI` model for OpenAI-based AI tasks.
#    - Leverages LangChain prompts (`ChatPromptTemplate`) to structure queries.

# 4. **Ollama for Local LLaMA2 Model**:
#    - Integrates the **LLaMA2** model via `Ollama` for running local AI tasks like generating poems.

# 5. **Custom Endpoints**:
#    - **`/openai`**: Provides general-purpose access to OpenAI’s `ChatOpenAI` model.
#    - **`/essay`**: Generates a 100-word essay on a specified topic using OpenAI’s model.
#    - **`/poem`**: Creates a 100-word child-friendly poem on a specified topic using the LLaMA2 model.

# 6. **Prompt-Based Tasks**:
#    - **Prompt 1**: Generates essays (`Write me an essay about {topic} with 100 words`).
#    - **Prompt 2**: Generates poems (`Write me a poem about {topic} for a 5 years child with 100 words`).

# 7. **Dynamic Route Addition**:
#    - Uses `langserve.add_routes` to easily map prompts and models to specific API endpoints.

# 8. **Server Execution**:
#    - Runs the FastAPI server locally with Uvicorn at `localhost:8000`.
