# --- Chatbot application Using Langchain  ---

# ref: https://python.langchain.com/docs/integrations/chat/openai/

# Step 1: Get required api keys. (https://github.com/ollama/ollama)
# Step 2: Then in terminal run > `streamlit run app.py`

from langchain_openai import ChatOpenAI  # Access GPT models
from langchain_core.prompts import ChatPromptTemplate  # Define structured prompts
from langchain_core.output_parsers import StrOutputParser  # Parse model output

# Additional libraries
import streamlit as st  # Build interactive web apps
import os  # Manage environment variables
from dotenv import load_dotenv  # Load .env variables

# Load and set environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Set OpenAI API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable LangChain tracing
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # Set LangChain API key

#1. Define chatbot prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries"),  # AI behavior
    ("user", "Question:{question}")  # User input placeholder
])

# Streamlit app setup
st.title('Langchain Demo With OPENAI API')  # App title
input_text = st.text_input("Search the topic you want")  # User input box

#2. Initialize LLM and output parser
llm = ChatOpenAI(model="gpt-3.5-turbo")  # GPT-3.5 model
#3. 
output_parser = StrOutputParser()  # Parse responses

# Combine (1,2,3) prompt, LLM, and parser into a processing chain
# chain = prompt | llm | output_parser creates a pipeline where user input is formatted by the prompt, processed by the LLM, and cleaned by the output parser for display.
chain = prompt | llm | output_parser

# Process user input and display response
if input_text:
    st.write(chain.invoke({'question': input_text}))  # Generate and display response


# ### Key Points for Beginners:
# 1. **Environment Variables**: Sensitive information like API keys is fetched securely from a `.env` file using `os.getenv`.
# 2. **Prompt Template**: A structured way to create prompts, separating instructions (for the system) and user inputs.
# 3. **Streamlit**: A Python library to build web interfaces where users can interact with your application.
# 4. **Pipeline (`chain`)**: Combines prompt generation, model interaction, and output formatting into a single workflow.
# 5. **Interactive Web App**: The user inputs a query, and the app uses OpenAI’s model to generate and display a response dynamically.