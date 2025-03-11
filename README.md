# AI-Powered Meeting Transcription and Summarization Application
## Project Overview

In this project, we aim to build an advanced AI application that listens to business meetings, transcribes the conversations, and provides a concise summary highlighting key points and decisions made during the meeting. We will leverage OpenAI's Whisper model to transcribe the speech and IBM Watson's AI to summarize and extract key insights from the conversation. The user interface will be built using Hugging Face's Gradio library, allowing users to upload audio files and receive both the transcription and the summary of the meeting in real-time.

This project demonstrates the use of cutting-edge machine learning technologies such as automatic speech recognition (ASR) with OpenAI's Whisper and AI-based summarization using IBM Watson's powerful language models. By combining these technologies, the application provides a user-friendly tool for quickly summarizing business meetings, helping users to focus on critical decisions and takeaways.

---

## Introduction

Business meetings often involve long discussions and decision-making processes, which can be overwhelming to track manually. Transcribing and summarizing the conversations are vital to ensure that key points are not missed. This project solves this problem by automating the transcription and summarization process.

The core functionality of this project can be broken down into three key components:
1. **Speech-to-Text**: Using OpenAI's Whisper model to convert spoken language from an audio file into a transcribed text format.
2. **Summarization and Key Point Extraction**: Leveraging IBM Watson's advanced language models to analyze the transcription and generate a concise summary while highlighting the key decisions and points discussed.
3. **User Interface**: Building a simple and intuitive UI using Gradio, allowing users to upload their meeting audio and receive the transcription and summary.

---

## Requirements

To run the project, you'll need to install the following dependencies:

- **Gradio**: For creating the web-based user interface.
- **Torch**: For handling machine learning models.
- **Transformers**: For Hugging Face pipeline functionality.
- **Langchain**: For chaining LLMs (Language Models) to prompt templates.
- **IBM Watson SDK**: For interacting with IBM Watson’s services.

Install dependencies with the following commands:

```bash
pip install torch gradio transformers langchain ibm-watson
```

You will also need to set up IBM Watson API credentials, as they are necessary for accessing Watson’s language models.

---

## Backend Code

### Importing Libraries and Setting Up Credentials

The first step is to import the necessary libraries. This includes `torch`, `gradio`, and a variety of tools from Hugging Face and Langchain for model management and prompt templating.

```python
import torch
import os
import gradio as gr
from langchain.llms import HuggingFaceHub
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
```

Here, we define the API credentials required to access IBM Watson's AI models.

```python
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}
params = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}
```

### IBM Watson Model Initialization

Next, we define the IBM Watson model. We use a pre-trained model provided by IBM Watson for text generation.

```python
LLAMA2_model = Model(
    model_id='meta-llama/llama-3-2-11b-vision-instruct', 
    credentials=my_credentials,
    params=params,
    project_id="skills-network",  
)
llm = WatsonxLLM(LLAMA2_model)
```

### Prompt Template for Summarization

The `PromptTemplate` is used to format the input for the summarization model. The input will contain the transcribed text, and the model will generate a summary with key points.

```python
temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""

pt = PromptTemplate(input_variables=["context"], template=temp)
prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)
```

### Speech-to-Text Function

The core of this project is the speech-to-text function, which converts uploaded audio into text using OpenAI's Whisper model.

```python
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    # Transcribe the audio file and return the result
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    result = prompt_to_LLAMA2.run(transcript_txt)

    return result
```

### Gradio User Interface

To make the application interactive, we use Gradio to create a user-friendly interface. Users can upload audio files, and the app will display both the transcription and summary.

```python
audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(fn=transcript_audio, 
                    inputs=audio_input, outputs=output_text, 
                    title="Audio Transcription App",
                    description="Upload the audio file")

iface.launch(server_name="0.0.0.0", server_port=8860)
```

### Explanation of Workflow

1. **Audio Input**: The user uploads an audio file through the Gradio interface.
2. **Speech-to-Text**: The audio file is passed to the Whisper model, which transcribes it into text.
3. **Summarization**: The transcribed text is then sent to IBM Watson's LLM to generate a summary and extract the key points.
4. **Display**: The resulting summary and key points are displayed back to the user in the interface.

---

## Conclusion

This project demonstrates how to combine multiple AI technologies to create a powerful tool for transcribing and summarizing business meetings. By leveraging OpenAI's Whisper for speech-to-text conversion and IBM Watson's language model for summarization, the application provides real-time insights into meetings. The intuitive Gradio interface ensures that users can easily interact with the application, making it an excellent tool for professionals looking to save time and stay focused on the most important takeaways from their meetings.

This project can be extended further by integrating additional AI models for specific industries, improving real-time performance, or adding additional features like sentiment analysis or speaker identification.

