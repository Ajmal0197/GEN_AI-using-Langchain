# --- Using LLM locally, here are running llama3.2 model locally ---

# ref: https://python.langchain.com/docs/integrations/llms/ollama/

# Step 1: Download ollama application. (https://github.com/ollama/ollama)
# Step 2: Download modal locally (https://github.com/ollama/ollama?tab=readme-ov-file#model-library)
# Step 3: Then in terminal run > `streamlit run ollamaLocalApp.py`

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title('Langchain Demo With LLAMA2 API')
input_text=st.text_input("Search the topic u want")

# ollama LLAma2 LLm 
llm=Ollama(model="llama3.2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
    
    
### **Key Explanation for Running LLaMA2 Locally**

# 1. **Local LLM Setup**:
#    - This script uses **Ollama** to run the **LLaMA2** model locally on your system.  
#    - Ollama allows efficient execution of large language models without needing external cloud services.

# 2. **Prompt Template**:
#    - Defines how the AI behaves (as a helpful assistant) and structures user queries using placeholders like `{question}`.

# 3. **Streamlit Web Interface**:
#    - Creates a simple, interactive webpage where users can input questions.

# 4. **Processing Pipeline**:
#    - Combines the prompt template, the locally running LLaMA2 model (via Ollama), and an output parser into a seamless chain.

# 5. **Real-Time Interaction**:
#    - Processes the user's input dynamically and displays AI-generated responses on the web interface.

# 6. **Running Locally**:
#    - The script does not require internet for querying the LLaMA2 model if properly configured, leveraging local hardware resources.
