# Agent Instruction Generator

## Overview

The Agent Instruction Generator is a Streamlit application designed to aid in generating detailed instructions for agents based on user-provided tasks. This tool utilizes various large language models (LLMs) including those from OpenAI, Anthropic, and Google Gemini.

## Features

- **Multiple LLM Integration**: Supports models like GPT-3.5, GPT-4, Claude-3 variations, and Gemini-1.5.
- **Custom Instruction Generation**: Generates specific instructions for agents, including persona and step-by-step workflows.
- **Translation Features**: Offers translation capabilities to Japanese and English for generated instructions.

## How to Use

1. **Input Task**: Enter the task you need assistance with in the designated text area.
2. **Set Output Language**: Select primary output language from the dropdown menu.
3. **Set Parameters**: Adjust the `Temperature`, `Top_p`, and `Max tokens` settings according to your preferences.
4. **Model Selection**: Choose one or more language models to generate responses.
5. **Translation Option**: If you need the second language, select it from the dropdown menu. For a complete list of language codes and their corresponding languages, please refer to the [DeepL documentation](https://developers.deepl.com/docs/resources/supported-languages). 
6. **Submit**: Click the 'Submit' button to process the input through the selected models.
7. **View Responses**: The generated content will appear in the "Generated Content Log" area as it is created.
8. **Final Instructions**: The complete and final instruction set will be displayed in a separate text area, ready to be copied and used as needed.

How To Use Video is available on Youtube: [https://youtu.be/xCJTE5-rVmk](https://youtu.be/xCJTE5-rVmk)

## Getting Started

### Prerequisites

- Python 3.8+
- pip packages: streamlit, openai, anthropic, google-generativeai, pandas, numpy, deepl, tiktoken

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/aisprint/agent-instruction-generator.git
   ```
2. Navigate to the project directory:
   ```
   cd agent-instruction-generator
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

Launch the application by running:
```
streamlit run app.py
```
Follow the on-screen instructions to generate agent instructions.

## Contributing

Contributions to improve Agent Instructon Generator are welcome. Please feel free to fork the repository and submit pull requests.

## Author

- **Mamoru** - Initial work and ongoing maintenance.

