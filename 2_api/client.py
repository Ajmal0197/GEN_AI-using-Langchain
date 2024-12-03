import requests
import streamlit as st

def get_openai_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']['content']

def get_ollama_response(input_text):
    response=requests.post(
    "http://localhost:8000/poem/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']

    ## streamlit framework

st.title('Langchain Demo With LLAMA2 API')
input_text=st.text_input("Write an essay on")
input_text1=st.text_input("Write a poem on")

if input_text:
    st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_ollama_response(input_text1))
    
#     ### **Key Functionalities of the Code**

# 1. **Integration with LangChain API Endpoints**:
#    - Makes HTTP POST requests to the FastAPI endpoints `/essay/invoke` and `/poem/invoke` to get responses for essays and poems.

# 2. **API Interaction**:
#    - **`get_openai_response(input_text)`**:
#      - Sends the `topic` as input to the `/essay/invoke` endpoint.
#      - Parses and returns the essay content from the API's JSON response.
#    - **`get_ollama_response(input_text)`**:
#      - Sends the `topic` as input to the `/poem/invoke` endpoint.
#      - Parses and returns the poem content from the API's JSON response.

# 3. **Streamlit UI**:
#    - Provides a simple user interface for the following tasks:
#      - **Essay Input**: Allows users to input a topic for an essay via a text box.
#      - **Poem Input**: Allows users to input a topic for a poem via another text box.

# 4. **Real-Time Interaction**:
#    - Displays the API responses dynamically:
#      - Generates an essay when a topic is entered in the essay input box.
#      - Generates a poem when a topic is entered in the poem input box.

# ---

# ### **How It Works**

# 1. **User Input**:
#    - Users input a topic in the **essay** and/or **poem** text boxes.

# 2. **API Calls**:
#    - The essay input triggers a POST request to `/essay/invoke`.
#    - The poem input triggers a POST request to `/poem/invoke`.

# 3. **Response Parsing**:
#    - Extracts and displays the generated content (essay or poem) from the API's JSON response.

# 4. **Streamlit Display**:
#    - The generated essay or poem is displayed immediately on the web page.

# ---

# ### **End-to-End Flow**
# 1. User inputs a topic into the Streamlit app.
# 2. The app sends the input as a JSON payload to the FastAPI endpoints.
# 3. The respective LangChain model generates a response based on the input and prompt.
# 4. The response is displayed on the Streamlit interface.

