import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import tiktoken

load_dotenv()

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
        if selected_models:
            tab_container = st.tabs([f"{model}" for model in selected_models])
            for i, model in enumerate(selected_models):
                with tab_container[i]:
                    if 'claude' in model:
                        response_text = generate_response_anthropic(prompt, model, temperature, max_tokens)
                    else:
                        response_text = generate_response_openai(prompt, model, temperature, max_tokens)
                    st.text_area("Response:", value=response_text, height=500, disabled=False)
        else:
            st.error("Please select at least one model.")