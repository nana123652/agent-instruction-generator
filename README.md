# Multi-LLM Tester

## Overview
The Multi-LLM Tester is a Streamlit application designed to facilitate easy comparison of responses from different large language models (LLMs) including OpenAI and Anthropic models. This tool is especially useful for developers and researchers interested in evaluating the capabilities and outputs of various LLMs under identical input conditions.

## Features
- **Multiple LLM Integration**: Supports various models from OpenAI and Anthropic, allowing users to test and compare responses side-by-side.
- **Dynamic Input Configuration**: Users can customize the prompt, temperature, and max tokens to tailor the testing environment according to their needs.
- **API Key Management**: Secure sidebar inputs for API keys with links to relevant documentation and pricing information.
- **Cost Estimation**: Provides a detailed breakdown of estimated costs per model based on the input configuration.
- **Privacy Policy Overview**: Includes an expander with privacy information related to the use of APIs.
- **Stylish UI**: Custom CSS for titles, authors, and update dates enhances the user interface experience.

## Getting Started

### Prerequisites
Before you can run the application, you'll need:
- Python 3.6 or later.
- Streamlit
- The required Python libraries listed in `requirements.txt`.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-llm-tester.git
   ```
2. Navigate to the cloned directory:
   ```bash
   cd multi-llm-tester
   ```
3. Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
To run the application, execute:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your web browser to see the app in action.

## Documentation
- **Privacy Policy**: Each LLM provider has specific guidelines and policies regarding data privacy and usage, accessible directly through links provided in the app.
- **API Keys**: Instructions for obtaining and managing API keys are provided in the sidebar.
- **Model Selection**: Users can select multiple models to test simultaneously, with the response from each displayed in a separate tab.

## Contributing
Contributions to improve Multi-LLM Tester are welcome. Please feel free to fork the repository and submit pull requests.

## Author
- Mamoru

Enjoy testing and comparing the capabilities of different large language models with ease!
