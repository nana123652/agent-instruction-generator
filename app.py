import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import tiktoken

import pandas as pd
import numpy as np

import time
from datetime import datetime

load_dotenv()

st.set_page_config("Multi-LLM Tester", page_icon=":memo:")

# Custom styles for the author name and updated date
st.markdown("""
<style>
.title-style {
    font-size: 50px;
    font-weight: bold;
    color: #5D00FF;
    text-align: center;
    margin-top: 30px;
}
.author-style {
    font-size: 20px;
    font-weight: bold;
    color: #D4BCFF;
    text-align: center;
    margin-bottom: 5px;
}
.update-date-style {
    font-size: 16px;
    color: #555;
    text-align: center;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# Display the stylish title
st.markdown('<div class="title-style">Multi-LLM Tester</div>', unsafe_allow_html=True)

# Display author and the updated date
current_date = datetime.now().strftime("%m/%d/%Y")
st.markdown('<div class="author-style">By Mamoru</div>', unsafe_allow_html=True)
st.markdown(f'<div class="update-date-style">Last Updated: {current_date}</div>', unsafe_allow_html=True)

# Privacy Policy
privacy_policy = st.expander("Privacy Policy")
privacy_policy.write("""
According to OpenAI and Anthropic, information sent via their APIs is not used for training large language models (LLMs).
- [OpenAI Documentation](https://platform.openai.com/docs/introduction)
- [Anthropic Support](https://support.anthropic.com/en/articles/7996875-can-you-delete-data-that-i-sent-via-api)
""")


# Add a new section for how to use the Paper Summarizer
st.markdown("For instructions on how to use the Paper Summarizer, visit the following link:")
st.markdown("[How to Use Paper Summarizer](https://github.com/aisprint/paper_summarizer_v2/blob/master/README.md)", unsafe_allow_html=True)


# Initialize or update the session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar for API keys
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
st.sidebar.markdown(
    "[How to get OpenAI API key?](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)", 
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "[OpenAI API Pricing](https://openai.com/api/pricing)", 
    unsafe_allow_html=True
)

anthropic_api_key = st.sidebar.text_input('Anthropic API Key', type='password')
st.sidebar.markdown(
    "[How to get Anthropic API key?](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)", 
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "[Anthropic API Pricing](https://www.anthropic.com/api)", 
    unsafe_allow_html=True
)

# API Pricing
data = {
    "Model": [
        "claude-3-haiku-20240307", 
        "claude-3-sonnet-20240229", 
        "gpt-3.5-turbo", 
        "gpt-4-turbo", 
        "claude-3-opus-20240229"
    ],
    "Prompt Cost": [
        "$0.0025", 
        "$0.03", 
        "$0.005", 
        "$0.1", 
        "$0.15"
    ],
    "Completion Cost": [
        "$0.0025", 
        "$0.03", 
        "$0.003", 
        "$0.06", 
        "$0.15"
    ],
    "Total Cost": [
        "$0.005", 
        "$0.06", 
        "$0.008", 
        "$0.16", 
        "$0.3"
    ]
}

# dataframe for api pricing
df_costs = pd.DataFrame(data)
st.sidebar.title("API cost estimation (Prompt 10,000 tokens, Completion 2,000 Tokens)")
st.sidebar.table(df_costs)


# Function to add conversation to history with formatted title
def add_to_history(user_input, response, file_name):
    now = datetime.now()
    formatted_date = now.strftime("%m/%d/%Y %H:%M:%S")
    title = f"{formatted_date}"
    st.session_state.history.append({"title": title, "user_input": user_input, "response": response})


# Initialize clients with keys

openai_api_key=os.getenv('OPENAI_API_KEY')
anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')

client_openai = OpenAI(api_key=openai_api_key)
client_anthropic = Anthropic(api_key=anthropic_api_key)

# Define function to get API client based on the model name
def get_client(model_name):
    if 'claude' in model_name:
        return client_anthropic
    else:
        return client_openai

# Copy button
def create_copy_button(text, key):
    button_html = f"""<button onclick='navigator.clipboard.writeText("{text}")'>Copy to clipboard</button>"""
    st.markdown(button_html, unsafe_allow_html=True)

#Define system prompt
with open("./system_prompt.txt","r") as f:
    system_prompt = f.read()

#count tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# OpenAI
def generate_response_openai(prompt, selected_model, temperature, max_tokens):
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
        return
    full_prompt = f"{system_prompt}\n{prompt}"
    prompt_token_count = num_tokens_from_string(full_prompt, "cl100k_base")
    st.markdown(f"**Prompt Token Count(system prompt added):** {prompt_token_count}")
    completion = client_openai.chat.completions.create(
    model=selected_model,
    temperature = temperature,
    max_tokens=max_tokens,
    messages=[{"role": "user", "content": full_prompt}],
    stream=False,
    )
    response = completion.choices[0].message.content
    return response

# Anthropic
def generate_response_anthropic(prompt, selected_model, temperature, max_tokens):
    if not anthropic_api_key.startswith('sk-'):
        st.warning('Please enter your Anthropic API key!', icon='⚠')
    if submitted and anthropic_api_key.startswith('sk-'):
        prompt_token_count = num_tokens_from_string(prompt, "cl100k_base")
        st.markdown(f"**Prompt Token Count:** {prompt_token_count}")
        response = client_anthropic.messages.create(
        system = system_prompt,
        max_tokens=max_tokens,
        model=selected_model,
        temperature=temperature,
        messages=[
            {"role":"user", "content": prompt}
        ]
    ) 
    return response.content[0].text

# Define a Streamlit form for input
with st.form(key='my_form'):
    prompt = st.text_area('Enter prompt:', 'ペンギンに関する短いエッセイを書いてください。')
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
    max_tokens = st.number_input("Max tokens", min_value=1, max_value=4000, value=2000)
    model_names = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229", "gpt-3.5-turbo", "gpt-4-turbo"]
    model_selection = {model: st.checkbox(model, value=False) for model in model_names}
    submitted = st.form_submit_button("Submit")

    if submitted:
        selected_models = [model for model, checked in model_selection.items() if checked]
        with st.spinner('Generating.....🤖💤'):
            if selected_models:
                tab_container = st.tabs([f"{model}" for model in selected_models])
                for i, model in enumerate(selected_models):
                    with tab_container[i]:
                        if 'claude' in model:
                            response_text = generate_response_anthropic(prompt, model, temperature, max_tokens)
                        else:
                            response_text = generate_response_openai(prompt, model, temperature, max_tokens)
                        st.text_area("Response:", value=response_text, height=500, disabled=False)
                        completion_token_count = num_tokens_from_string(response_text, "cl100k_base")
                        st.markdown(f"**Completion Token Count:** {completion_token_count}")
            else:
                st.error("Please select at least one model.")