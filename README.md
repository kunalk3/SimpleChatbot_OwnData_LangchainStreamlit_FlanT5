# SimpleChatbot_OwnData_LangchainStreamlit_FlanT5

<div align="right">
<img src="https://user-images.githubusercontent.com/41562231/141720820-090897f9-f564-45e2-9265-15c1269db795.png" height="120" width="900">
</div>

## Question Answer System To Your Own Data (LangChain + HuggingFace + Streamlit + QA + VectorDB)

I have created a __Question Answer System__ using LangChain framework and HF model (API based) with demo. User need to load own PDF data and ask the questions in chatbot window. Behind the backend, pdf is being process and we are creating embeddings vector of chunk data. These embeddings can be store to vector database and we are going to retrieve the embeddings as per user ask. 

<div align="center">
  <a href="https://github.com/kunalk3/SimpleChatbot_OwnData_LangchainStreamlit_FlanT5/issues"><img src="https://img.shields.io/github/issues/kunalk3/Llama3_chatbot_LangSmithLangChain" alt="Issues Badge"></a>
  <a href="https://github.com/kunalk3/SimpleChatbot_OwnData_LangchainStreamlit_FlanT5/graphs/contributors"><img src="https://img.shields.io/github/contributors/kunalk3/Llama3_chatbot_LangSmithLangChain?color=872EC4" alt="GitHub contributors"></a>
  <a href="https://www.python.org/downloads/release/python-390/"><img src="https://img.shields.io/static/v1?label=python&message=v3.12&color=faff00" alt="Python3.12"</a>
  <a href="https://github.com/kunalk3/SimpleChatbot_OwnData_LangchainStreamlit_FlanT5/blob/main/LICENSE"><img src="https://img.shields.io/github/license/kunalk3/Llama3_chatbot_LangSmithLangChain?color=019CE0" alt="License Badge"/></a>
  <a href="https://github.com/kunalk3/SimpleChatbot_OwnData_LangchainStreamlit_FlanT5"><img src="https://img.shields.io/badge/lang-eng-ff1100"></img></a>
  <a href="https://github.com/kunalk3/SimpleChatbot_OwnData_LangchainStreamlit_FlanT5"><img src="https://img.shields.io/github/last-commit/kunalk3/Llama3_chatbot_LangSmithLangChain?color=309a02" alt="GitHub last commit">
</div>
  
<div align="center">   
  
  [![Windows](https://img.shields.io/badge/WindowsOS-000000?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/en-in/) 
  [![Visual Studio Code](https://img.shields.io/badge/VSCode-0078d7.svg?style=flat-square&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/) 
  [![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=Jupyter&logoColor=white)](https://jupyter.org/) 
  [![Pycharm](https://img.shields.io/badge/Pycharm-41c907.svg?style=flat-square&logo=Pycharm&logoColor=white)](https://www.jetbrains.com/pycharm/) 
  [![Colab](https://img.shields.io/badge/Colab-F9AB00.svg?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/?utm_source=scs-index/)
  [![Spyder](https://img.shields.io/badge/Spyder-838485.svg?style=flat-square&logo=spyder%20ide&logoColor=white)](https://www.spyder-ide.org/) 
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)](https://share.streamlit.io/)

  [![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=flat-square&logo=openai&logoColor=white)](https://chatgpt.com/) 
  [![LangChain](https://img.shields.io/badge/langchain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/) 
  [![HuggingFace](https://img.shields.io/badge/-HuggingFace-FDEE21?style=flat-square&logo=HuggingFace&logoColor=black)](https://huggingface.co/) 
  [![GoogleGemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)](https://gemini.google.com/) 
  [![Keras](https://img.shields.io/badge/Keras-FF0000?style=flat-square&logo=Keras&logoColor=white)](https://keras.io/) 
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
  [![GCP](https://img.shields.io/badge/Google-4285F4?style=flat-square&logo=google-cloud&logoColor=white)](https://cloud.google.com/) 
  [![AWS](https://img.shields.io/badge/AWS-FF9900?style=flat-square&logo=iCloud&logoColor=white)](https://aws.amazon.com/) 
  [![Azure](https://img.shields.io/badge/Azure-0089D6?style=flat-square&logo=Cloudflare&logoColor=white)](https://azure.microsoft.com/)

</div>
  
<div align="center">
  
  [![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-Profile-informational?style=flat&logo=linkedin&logoColor=white&color=0078d7)](https://www.linkedin.com/in/kunalkolhe3/)
  [![Github Badge](https://img.shields.io/badge/Github-Profile-informational?style=flat&logo=github&logoColor=white&color=black)](https://github.com/kunalk3/)
  [![Porfolio Badge](https://img.shields.io/badge/Porfolio-Profile-informational?style=flat&logo=Twilio&logoColor=white&color=FDEE21)](https://kunalk3.github.io/Portfolio-Website-Kunalk3/)
  [![Gmail Badge](https://img.shields.io/badge/Gmail-Profile-informational?style=flat&logo=Gmail&logoColor=white&color=e44e4e)](mailto:kunalkolhe333@gmail.com)
  [![Facebook Badge](https://img.shields.io/badge/Facebook-Profile-informational?style=flat&logo=facebook&logoColor=white&color=0078d7)](https://www.facebook.com/kunal.kolhe.98/)
  [![Instagram Badge](https://img.shields.io/badge/Instagram-Profile-informational?style=flat&logo=Instagram&logoColor=white&color=c90076)](https://www.instagram.com/kkunalkkolhe/)
</div>
  
---
  
## :books: Overview:
- __LangChain__ it’s an open-source framework that acts as a catalyst for creativity and innovation. 
- __HuggingFace__ is a powerful AI-ML model platform.
- __Streamlit__ is a framework for turning Python scripts into beautiful web applications. 
- __FAISS__ is popular for storing vector embeddings to vector store, so we can retrieve embeddings based on similarity score.

<img width="50%"  alt="1" src="https://github.com/user-attachments/assets/81c3c24c-bd3c-4968-ad7a-ff675e538e7b">


## :wrench: Installation:

- Create __virtual environment__ `python -m venv VIRTUAL_ENV_NAME` and activate it `.\VIRTUAL_ENV_NAME\Scripts\activate`.
- Install necessary library for this project from the file `requirements.txt` or manually install by `pip`.
  ```
  pip install -r requirements.txt
  ```
  To create project library requirements, use below command,
  ```
  pip freeze > requirements.txt
- We need API to call HuggingFace model from HuggingFace Hub. You can get API from portals.
  1) HUGGINGFACEHUB_API_TOKEN ([HuggingFace API](https://huggingface.co/)) - We are calling FLAT-T5 model via HF API endpoints."
- Mention the API keys to your .env file. We are going to use API key from .env file using  load_dotenv() method in python code. Below is the .env file contents.
  ```
  HUGGINGFACEHUB_API_TOKEN = "YOUR_API_KEY"       # replace API KEY with your KEY
  ```
- You can run the python file using streamlit command and you are able to see the chatbot interface in browser.
  ``` 
  streamlit run app.py
  ```

---  

## :bulb: Demo
#### :bookmark: _Output_ - 

https://github.com/user-attachments/assets/a99229d9-6891-4e76-a1e4-c964a816ab1f

---  
  
## :bookmark: Directory Structure 
```bash
    .                                            # Root directory
    ├── .env                                     # API environment
    ├── demo.py                                  # Application main file
    ├── code.ipynb                               # Notebook file with code explain
    ├── assets_img                               # Supportive images and files
    │   ├── picture1.png                         
    │   ├── picture2.png                        
    ├── requirements.txt                         # Project requirements library with versions
    ├── README.md                                # Project README file
    └── LICENSE                                  # Project License file
```

---  
  
### :iphone: Connect with me
`You say freak, I say unique. Don't wait for an opportunity, create it.`
  
__Let’s connect, share the ideas and feel free to ping me...__
  
<div align="center"> 
  <p align="left">
    <a href="https://linkedin.com/in/kunalkolhe3" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/linkedin.svg" alt="kunalkolhe3" height="30" width="40"/></a>
    <a href="https://github.com/kunalk3/" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/github.svg" alt="kunalkolhe3" height="30" width="40"/></a>
    <a href="https://fb.com/kunal.kolhe.98" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/facebook.svg" alt="kunal.kolhe.98" height="30" width="40"/></a>
    <a href="mailto:kunalkolhe333@gmail.com" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/gmail.svg" alt="kunalkolhe333" height="30" width="40"/></a>
    <a href="https://instagram.com/kkunalkkolhe" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/instagram.svg" alt="kkunalkkolhe" height="30" width="40"/></a>
    <a href="https://www.hackerrank.com/kunalkolhe333" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/hackerrank.svg" alt="kunalkolhe333" height="30" width="40"/></a>
    <a href="https://kunalk3.github.io/Portfolio-Website-Kunalk3/" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/googlecloud.svg" alt="kunalkolhe333" height="30" width="40"/></a>
  </p>
</div>
  
<div align="left">
<img src="https://user-images.githubusercontent.com/41562231/141720940-53eb9b25-777d-4057-9c2d-8e22d2677c7c.png" height="120" width="900">
</div>
