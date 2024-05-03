# Paper Summarizer

## Overview
This project includes a Python application that summarizes PDF documents using AI models from OpenAI and Anthropic. This is my first Python project. I've created an app that summarizes scientific papers using LLMs (like GPT and Claude 3).

## Installation
Clone this repository and install required packages:

git clone https://github.com/aisprint/paper_summarizer.git

cd paper_summarizer

pip install -r requirements.txt


## Usage
Run the Streamlit application:

streamlit run main.py

### How to Use the App

1. Get an API key for the model you intend to use and enter it in the API Key field in the left sidebar. Refer to the respective links for instructions on how to obtain the API keys.
2. Upload the paper (PDF file) you want to summarize by clicking "Browse Files" or drag and drop the file into the designated area. The content of the paper will be automatically extracted and displayed in text format.
3. Set the Model, Temperature, and Max Tokens. The prompt is preloaded with instructions for summarizing the paper, so no changes are necessary unless you have specific requests. In that case, enter your modifications in the Prompt box.
4. Click Submit to start the summarization process.
5. Use the copy button at the top right of the displayed results to copy the summary. If the response is truncated, increase the Max Tokens value and submit again.
6. History can be viewed from the dropdown list at the bottom of the page, but note that all history will be cleared when the app window is closed.

## Demo
The app is available at the Streamlit server:
[Paper Summarizer V2](https://papersummarizerv2-ertgst946lkt3qpqf5cxup.streamlit.app/)

## Contributing
Contributions are welcome. If you have any advice or suggestions, especially since I am new to GitHub, please feel free to open an issue or pull request.

## Note
As this is my first experience with Python projects and using GitHub, I am still getting familiar with these platforms. Any guidance or advice would be greatly appreciated.


