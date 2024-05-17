import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

import os
import tiktoken

import pandas as pd
import numpy as np

import time
from datetime import datetime

import deepl

st.set_page_config("Agent Instruction Generator", page_icon=":robot:")

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
st.markdown(f'<div class="title-style">Agent Instruction(Prompt) \n Generator🤖💬</div>', unsafe_allow_html=True)

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


# Add a new section for how to use the Agent Instruction Generator
st.markdown("For instructions on how to use the Agent Instruction Generator, visit the following link:")
st.markdown("[How to Use Agent Instruction Generator](https://github.com/aisprint/agent-instruction-generator/blob/main/README.md)", unsafe_allow_html=True)


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

google_api_key = st.sidebar.text_input('Gemini API Key', type='password')
st.sidebar.markdown(
    "[How to get Gemini API key?](https://ai.google.dev/gemini-api/docs/api-key)", 
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "[Gemini API Pricing](https://ai.google.dev/pricing)", 
    unsafe_allow_html=True
)

# Sidebar for API keys (including DeepL)
deepL_api_key = st.sidebar.text_input('DeepL API Key', type='password')
st.sidebar.markdown("[How to get DeepL API key?](https://www.deepl.com/pro#developer)", unsafe_allow_html=True)

# API Pricing
data = {
    "Model": [
        "claude-3-haiku-20240307", 
        "claude-3-sonnet-20240229", 
        "claude-3-opus-20240229",
        "gpt-3.5-turbo", 
        "gpt-4-turbo", 
        "gpt-4o",
        "gemini-1.5-pro-latest"
    ],
    "Prompt Cost": [
        "$0.0025", 
        "$0.03", 
        "$0.15",
        "$0.005", 
        "$0.1", 
        "$0.05",
        "$0.07"
    ],
    "Completion Cost": [
        "$0.0025", 
        "$0.03", 
        "$0.15",
        "$0.003", 
        "$0.06", 
        "$0.03",
        "$0.042"
    ],
    "Total Cost": [
        "$0.005", 
        "$0.06", 
        "$0.3",
        "$0.008", 
        "$0.16", 
        "$0.08",
        "$0.112"
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

client_openai = OpenAI(api_key=openai_api_key)
client_anthropic = Anthropic(api_key=anthropic_api_key)
genai.configure(api_key=google_api_key)

# Define function to get API client based on the model name
def get_client(model_name):
    if 'claude' in model_name:
        return client_anthropic
    else:
        return client_openai

#Define a prompt for agent brief
with open("./prompt_agent_brief.txt","r") as f:
    prompt_agent_brief = f.read()

#Define a prompt for persona
with open("./prompt_persona.txt","r") as f:
    prompt_persona = f.read()

#Define a promot for Step flow
with open("./prompt_step_flow.txt","r") as f:
    prompt_step_flow = f.read()

#Define a promot for final remark
with open("./prompt_final_remark.txt","r") as f:
    prompt_final_remark = f.read()

#count tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# OpenAI
def generate_response_openai(prompt, selected_model, temperature, top_p, max_tokens):
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
        return
    prompt_token_count = num_tokens_from_string(prompt, "cl100k_base")
    # st.markdown(f"**Prompt Token Count(system prompt added):** {prompt_token_count}")
    completion = client_openai.chat.completions.create(
    model=selected_model,
    temperature = temperature,
    top_p=top_p,
    max_tokens=max_tokens,
    messages=[{"role": "user", "content": prompt}],
    stream=False,
    )
    response = completion.choices[0].message.content
    return response

# Anthropic
def generate_response_anthropic(prompt, selected_model, temperature, top_p, max_tokens):
    if not anthropic_api_key.startswith('sk-'):
        st.warning('Please enter your Anthropic API key!', icon='⚠')
        return
    if submitted and anthropic_api_key.startswith('sk-'):
        prompt_token_count = num_tokens_from_string(prompt, "cl100k_base")
        # st.markdown(f"**Prompt Token Count:** {prompt_token_count}")
        response = client_anthropic.messages.create(
        system = 'You are a helpful assistant',
        max_tokens=max_tokens,
        model=selected_model,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role":"user", "content": prompt}
        ]
    ) 
    return response.content[0].text

#Google Gemini Pro 1.5
def generate_response_gemini(prompt, selected_model, temperature, top_p, max_token):
    # if not google_api_key.startswith('AI'):
    #     st.warning('Please enter correct Gemini API key!', icon='⚠')   
    #     return
        generation_config = {
            'temperature': temperature,
            'top_k': 0,  # Assuming top_k is not to be modified per call in this setup
            'top_p': top_p,  # Fixed top_p for more creative responses
            'max_output_tokens': max_token,
        }
        model = genai.GenerativeModel(model_name=selected_model,generation_config=generation_config)
        results = model.generate_content(prompt)
        response = results.candidates[0].content.parts[0].text
        return response

# Function to handle translation
def translate_text(input_text, target_language):
    if not deepL_api_key:
        st.error("Please set your DeepL API key to use the translation feature.")
        return input_text  # return original text if no API key
    translator = deepl.Translator(deepL_api_key)
    translated_text = translator.translate_text(input_text, target_lang=target_language)
    return translated_text  


# Scroll text area
def scroll_text_area(key):
    st.markdown(f"""
        <script>
            const textarea = document.getElementById("{key}");
            textarea.scrollTop = textarea.scrollHeight;
        </script>
        """, unsafe_allow_html=True)

# Instruction generator
with st.form(key='agent_instruction'):
    prompt = st.text_area("1.Please enter what you need help:",'Blog writing assistants', height=50)
    output_language = st.selectbox("2.Please set output language", ["English", "Japanese", "German", "Spanish", "French", "Italian", "Korean", "Polish", "Russian", "Turkish", "Chinese"]
, index=0)
    full_prompt = f"{prompt_agent_brief}\n{prompt}\n\n<Output language>\n{output_language}"
    temperature = st.slider("3.Please set Temperature", 0.0, 1.0, 0.5)
    top_p = st.slider("4.Please set Top_p", 0.0, 1.0, 0.9)
    max_tokens = st.number_input("5.Please set max tokens", min_value=1, max_value=4000, value=2000)
    st.markdown("6.Please select LLMs:")
    model_names = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", "gemini-1.5-pro-latest"]
    model_selection = {model: st.checkbox(model, value=False) for model in model_names}
    # Translation dropdown in the form
    translation_options = st.selectbox("7.Please set language if you want to generate the second language by DeepL(See codes in [DeepL Docs](https://developers.deepl.com/docs/resources/supported-languages))", ["NONE", "EN-US", "JA", "DE", "ES", "FR", "IT", "KO", "PL", "RU", "TR", "ZH"], index=0)
    submitted = st.form_submit_button("Submit")

if submitted:
    selected_models = [model for model, checked in model_selection.items() if checked]
    response_text=""

    if selected_models:
        tab_container = st.tabs([f"{model}" for model in selected_models])
        response_container = st.empty()
        final_instructions=[]

        for i, model in enumerate(selected_models):
            with tab_container[i]:

                # Generate Agent Brief
                with st.spinner(f'Generating AGENT BRIEF from {model}...🤖💤'):
                    if 'claude' in model:
                        response_text_agent_brief = generate_response_anthropic(full_prompt, model, temperature, top_p, max_tokens)
                    elif 'gpt' in model:
                        response_text_agent_brief = generate_response_openai(full_prompt, model, temperature, top_p, max_tokens)
                    else:
                        response_text_agent_brief=generate_response_gemini(full_prompt, model, temperature, top_p, max_tokens)
                    
                    response_text += f"# Agent Brief\n{response_text_agent_brief}\n\n"
                    response_container.text_area("Generated Content log", value=response_text, height=800)
                    scroll_text_area("dynamic_text_area")

            
                # Generate Persona
                with st.spinner(f'Generating PERSONA from {model}...🤖💤'):
                    full_prompt=f"{response_text_agent_brief}\n{prompt_persona}\n\n<Output language>\n{output_language}"
                    if 'claude' in model:
                        response_text_persona = generate_response_anthropic(full_prompt, model, temperature, top_p, max_tokens)
                    elif 'gpt' in model:
                        response_text_persona = generate_response_openai(full_prompt, model, temperature, top_p, max_tokens)
                    else:
                        response_text_persona=generate_response_gemini(full_prompt, model, temperature, top_p, max_tokens)

                    response_text += f"# Persona\n{response_text_persona}\n\n"
                    response_container.text_area("Generated Content log", value=response_text, height=800)

                # Generate Step flow
                with st.spinner(f'Generating STEP FLOW from {model}...🤖💤'):
                    full_prompt=f"{response_text_persona}\n{prompt_step_flow}\n\n<Output language>\n{output_language}"
                    if 'claude' in model:
                        response_text_step_flow = generate_response_anthropic(full_prompt, model, temperature, top_p, max_tokens)
                    elif 'gpt' in model:
                        response_text_step_flow = generate_response_openai(full_prompt, model, temperature, top_p, max_tokens)
                    else:
                        response_text_step_flow=generate_response_gemini(full_prompt, model, temperature, top_p, max_tokens)

                    response_text += f"# Step Flow\n{response_text_step_flow}\n\n"
                    response_container.text_area("Generated Content log", value=response_text, height=800)

                # Generate Final remark
                with st.spinner(f'Generating FINAL REMARK from {model}...🤖💤'):
                    full_prompt=f"{response_text_agent_brief}\n{prompt_final_remark}\n\n<Output language>\n{output_language}"
                    if 'claude' in model:
                        response_text_final_remark = generate_response_anthropic(full_prompt, model, temperature, top_p, max_tokens)
                    elif 'gpt' in model:
                        response_text_final_remark = generate_response_openai(full_prompt, model, temperature, top_p, max_tokens)
                    else:
                        response_text_final_remark=generate_response_gemini(full_prompt, model, temperature, top_p, max_tokens)

                    response_text += f"# Final Remark\n{response_text_final_remark}\n\n"
                    response_container.text_area("Generated Content log", value=response_text, height=800)
                
                # Final instruction
                st.markdown(f"\n\n")
                st.markdown(f"**Agent Instruction (Final)**")
                full_instruction=f"# PERSONA\n{response_text_persona}\n\n# STEP FLOW\n{response_text_step_flow}\n\n{response_text_final_remark}"
                final_instructions.append(full_instruction)
                st.text_area("Please copy this instruction and paste into system prompt in any LLMs", value=full_instruction, height=800, disabled=False)
                
                # Translate response if needed and if NOT 'NONE'
                if translation_options != "NONE":
                    translated_response = translate_text(full_instruction, translation_options)
                    st.text_area("Translated Instrcution:", value=translated_response, height=800, disabled=False)

        # # Translate response if needed and if NOT 'NONE'
        # translated_responses=[]
        # if translation_options != "NONE":
        #     for i, final_instruction in enumerate(final_instructions):
        #         translated_response = translate_text(final_instruction, translation_options)
        #         translated_responses.append(translated_response)
        #         with tab_container[i]:
        #             st.text_area("Translated Instruction:", value=translated_responses[i], height=800, disabled=False)

    else:
        st.error("Please select at least one model.")
        

    