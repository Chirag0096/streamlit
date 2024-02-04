import subprocess

# Install streamlit and login to Hugging Face Hub
subprocess.run(["pip", "install", "streamlit"])
subprocess.run(["pip", "install", "huggingface-hub"])




# Install required libraries
libraries_to_install = [
    "transformers",
    "datasets",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "auto-gptq",
    "optimum",
    "playwright",
    "langchain",
    "html2text",
    "sentence_transformers",
    "faiss-gpu",
    "pyttsx3",
    "SpeechRecognition",
]

for lib in libraries_to_install:
    subprocess.run(["pip", "install", "-qU", lib])

# Install Playwright dependencies
subprocess.run(["playwright", "install"])
subprocess.run(["playwright", "install-deps"])




import os
from huggingface_hub import notebook_login

# Paste your API key here
api_key = "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"

# Set the Hugging Face Hub API key as an environment variable
os.environ["HF_HOME"] = os.path.expanduser("~/.huggingface")
os.environ["HF_HOME_WRITE"] = os.environ["HF_HOME"]
os.environ["HF_HOME_READ"] = os.environ["HF_HOME"]
os.environ["HF_HOME_TRANSFORMERS"] = os.path.join(os.environ["HF_HOME"], "transformers")
os.environ["HF_HOME_DATASETS"] = os.path.join(os.environ["HF_HOME"], "datasets")
os.environ["HUGGINGFACE_TOKEN"] = api_key

# Login to the Hugging Face Hub
notebook_login()













import streamlit as st
#import whisper
import pyttsx3
import speech_recognition as sr


c = 0
f=0
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments
from trl import SFTTrainer
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
import requests
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoConfig
from transformers import AutoTokenizer
import torch


class Input:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Chirag0123/zephyr_beta")
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            "Chirag0123/zephyr_beta",
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        self.generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=0.1,
            max_new_tokens=1500,
            pad_token_id=self.tokenizer.eos_token_id
        )
    def base(self, text_input):
        return self.loop(text_input)
    def loop(self, text_input):
        global c
        self.inp_str = self.process_data_sample(
            {
                "instruction": text_input,
            }
        )
        self.inputs = self.tokenizer(self.inp_str, return_tensors="pt").to("cuda")
        import time
        st_time = time.time()
        self.outputs = self.model.generate(**self.inputs, generation_config=self.generation_config)
        import re
        self.a = self.tokenizer.decode(self.outputs[0], skip_special_tokens=True)
        # Find the indices of the substrings
        self.start_index = (self.a).find("asdfghjk")
        self.end_index_first_occurrence = (self.a).find("poiuytttr")
        self.end_index_second_occurrence = (self.a).find("poiuytttr", self.end_index_first_occurrence + 1)

# Extract the substring between the first occurrence of "asdfghjk" and the second occurrence of "poiuytttr"
        self.result_string = (self.a)[self.start_index + len("asdfghjk"):self.end_index_second_occurrence]
        global voice_out
        voice_out=self.result_string
        return self.result_string
    def process_data_sample(self, example):
          self.processed_example = "poiuytttr<|system|>\n You are a legal assistant. Provide your response in context of preparing a legal case to be fought in a court of law in India.\n<|user|>\n" + example["instruction"] + "\n asdfghjk<|assistant|>\n"
          return self.processed_example
# Define a function that takes user input and returns the chatbot response



class Input2:
    
    def __init__(self):
        import torch
        from datasets import load_dataset, Dataset

        self.tokenizer = AutoTokenizer.from_pretrained("Bhagya17/SumCase")
        self.config = AutoConfig.from_pretrained("Bhagya17/SumCase")
        # config.quantization_config["use_exllama"] = True
        # config.quantization_config["exllama_config"] = {"version":2}
        self.config.quantization_config["disable_exllama"] = True
        self.model = AutoModelForCausalLM.from_pretrained("Bhagya17/SumCase", low_cpu_mem_usage=True, torch_dtype=torch.float16, config=self.config)
        self.pipe = pipeline(task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device="cuda:0",
                max_new_tokens=8096,
                temperature=0.1,
                #repetition_penalty=1.1,
                return_full_text=True,
                do_sample= True)
        self.llm=HuggingFacePipeline(pipeline=self.pipe)
                
        self.prompt_template = """
        <|system|>\n: You are a legal assistant. Provide your response in context of preparing a legal case to be fought in a court of law in India. Your response should always be in the following format:
        Case Details:
        Case Name: .
        Case Number:
        Court and Jurisdiction:
        Date of Filing:
        Parties Involved:
        Petitioner(s):
        Respondent(s):
        Legal Representatives:
        Counsel for Petitioner:
        Counsel for Respondent:
        Type of Writ:
        Factual Background:
        Key Facts:
        Context:
        Legal Issues and Questions:
        Primary Legal Issues:
        Subsidiary Issues:
        Arguments and Grounds:
        Petitioner’s Arguments:
        Legal Basis:
        Precedents Cited:
        Relief Sought:
        Specific Relief Requested:
        Decision/Outcome (if available):
        Court’s Decision:
        Reasoning:
        
        
        {context}
        
        \n<|user|>\n:
        {question}
         """
        
        # Create prompt from prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.prompt_template,
        )
        
        # Create llm chain
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
    def base(self, text_input):
        return self.loop(text_input)
    def loop(self, text_input):
        self.api_token = '5c7717123da9cf13289e41185fb4d95726f8062e'
        self.document_urls = self.get_document_urls(text_input, num_documents=1, api_token=self.api_token)
        import nest_asyncio
        nest_asyncio.apply()

        # Articles to index
        self.articles = []
        self.articles = self.document_urls

        # Scrapes the blogs above
        self.loader = AsyncChromiumLoader(self.articles)
        self.docs = self.loader.load()
        self.html2text = Html2TextTransformer()
        self.docs_transformed = self.html2text.transform_documents(self.docs)

        # Chunk text
        self.text_splitter = CharacterTextSplitter(chunk_size=100,
                                              chunk_overlap=0)
        self.chunked_documents = self.text_splitter.split_documents(self.docs_transformed)

        # Load chunked documents into the FAISS index
        self.db = FAISS.from_documents(self.chunked_documents,
                                  HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

        self.retriever = self.db.as_retriever()
        self.rag_chain = (
         {"context": self.retriever, "question": RunnablePassthrough()}
            | self.llm_chain
        )

        self.result = self.rag_chain.invoke(f"Can you please provide a summary of the {text_input}?")
        return str(self.result['text'])
        
    def search_api(self,text_input, pagenum=0, format='json', api_token='5c7717123da9cf13289e41185fb4d95726f8062e'):
        self.url = f'https://api.indiankanoon.org/search/'
        self.headers = {'Authorization': f'Token {api_token}', 'Accept': f'application/{format}'}
        self.data = {'formInput': text_input, 'pagenum': pagenum}

        self.response = requests.post(self.url, headers=self.headers, data=self.data)

        if self.response.status_code == 200:
            return self.response.json()
        else:
            print(f"Error {self.response.status_code}: {self.response.text}")
            return None

    def get_document_urls(self,text_input, num_documents=1, api_token='5c7717123da9cf13289e41185fb4d95726f8062e'):
        self.result = self.search_api(text_input, format='json', api_token=api_token)
        self.document_urls = []
        if self.result:
            self.docs = self.result.get('docs', [])

            for i, doc in enumerate(self.docs[:num_documents]):
                self.doc_id = doc.get('tid')
                self.document_urls.append(f'https://indiankanoon.org/doc/{self.doc_id}/')

        return self.document_urls
    
main=Input()

summerize = Input2()

st.set_page_config(
    page_title="LawGPT",
    page_icon="🚀",  # You can use emoji or an image URL for the favicon
    layout="wide",
    
    # "wide" or "centered"
)







with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
logo_image = "new.jpeg"  # Replace with the path to your logo image

st.image(
    logo_image,
    width=200
)

def text_to_speech(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

    # Convert text to speech
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait()

def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio."
        except sr.RequestError as e:
            return "Error with the API request; {0}".format(e)

# def transcribe(audio):

#     # load audio and pad/trim it to fit 30 seconds
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)

#     # make log-Mel spectrogram and move to the same device as the model
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)

#     # detect the spoken language
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")

#     # decode the audio
#     options = whisper.DecodingOptions()
#     result = whisper.decode(model, mel, options)
#     return result.text


    
titles = st.container()

with titles:
    
    st.markdown(
    """
    <style>
    .new {
        
        text-align: center;
        margin-top:200px;
    }
    </style>
    <div class="all">
    <div >
        <h4 class="new">Your personal legal assistant</h4>
    </div>
    """,
    unsafe_allow_html=True,
    )
    input_with_icons = """
    <style>
    .container {
        width: 100%;
        height: 60px;
        background-color: white;
        border-radius: 15px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        border: 2px solid black;
    }
    .input {
        align-content: center;
    }
    .mic-icon, .clip-icon {
        font-size: 24px;
    }
    .mic-icon {
        margin-right: 80rem;
    }
    </style>
    </head>
    <body>
    <div class="input">
    <p>Input the query</p>

    <div class="container">
        <a href="#" onclick="Streamlit.buttonClicked(0)">
            <i class="fas fa-microphone mic-icon"></i>
        </a>
        <i class="fas fa-paperclip clip-icon"></i>
    </div>
    </div>
    """

    # Render the HTML with the external function
    st.markdown(input_with_icons, unsafe_allow_html=True)


    
    col1, col2, col3 = st.columns([1, 95, 1])

    # Define the vertical position (adjust the margin-top value as needed)
    vertical_position = "-95px"
    b="-50px"
    
    # Apply style to the middle column
    col2.markdown(f"""
        <style>
            .stTextInput {{
                width: 1250px;
                margin: {vertical_position} auto 0;
            }}
            .stButton {{
                margin: {b} auto 0;
                margin-left:9px;
            }}
        </style>
         
    """, unsafe_allow_html=True)



    submit_button = col3.button("⬇️")
    

    global recognized_text

    mic_clicked = col1.button("🎙️",key="microphone_key")
    # Handle button clicks
    if mic_clicked:
      recognized_text = speech_to_text()
    else:
      recognized_text=""
 
    # Use st.text_input in the middle column
    user_input = col2.text_input("", key="user_input",value=recognized_text)

    # Create the research section with options
    research_options = ["Summarize", "Draf-Petition"]
    selected_research_option = st.selectbox("Selection", research_options)
    

if (selected_research_option=="Summarize") and submit_button:
    if (f == 0):  
        Output=summerize.base(user_input)
        st.write(Output)
        f+=1
    else:
        Output=summerize.loop(user_input)
        st.write(Output)        
elif (selected_research_option=="Draf-Petition") and submit_button:
    if (c == 0):  
        Output=main.base(user_input)
        st.write(Output)
        c+=1
    else:
        Output=main.loop(user_input)
        st.write(Output)








# Set the title in the Streamlit sidebar
st.sidebar.title("ThemisX")



# Add buttons to the sidebar
button1 = st.sidebar.button("➕New Chat")
button2 = st.sidebar.button("✍🏻Compose")
button3 = st.sidebar.button("🔬Research")
button4 = st.sidebar.button("📑Summarize")
button5 = st.sidebar.button("👨‍💻Analyze")
button6 = st.sidebar.button("💬Argue")

# You can add functionality based on button clicks
if button1:
    st.write("Button 1 clicked!")

if button2:
    st.write("Button 2 clicked!")

if button3:
    st.write("Button 3 clicked!")

css_example = '''
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    

'''



Output = "This is a dynamic value" * 40  # Creating a longer text for demonstration

styled_output = f"""
                <p style="text-align: center;">Output</p>
<div style="display: flex; justify-content: center; align-items: center; ">
    <div style="width: 1024px; height: 330px; overflow-y: scroll; color: black;
                border: 2px solid #ccc; padding: 10px; border-radius: 10px;">
        {Output}
    </div>
</div>
"""

st.markdown(styled_output, unsafe_allow_html=True)



from streamlit.components.v1 import html

# Create a button
Output_mic = st.button("🔊", key="mic_button")

if Output_mic:
    text_to_speech(Output)

    


