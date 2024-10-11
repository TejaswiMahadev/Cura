import os
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import sqlite3
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

import google.generativeai as genai
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# consultation code

def get_gemini_response(prev_chat):
    model = genai.GenerativeModel(model_name='gemini-pro')

    response = model.generate_content(f''' 
    Prompt:

    Your name is "CuraBot" and you are a doctor who gives the medications and let the user know the disease he is suffering from based on the symptoms he provides
    
        Your Role:
        1) Your a healthbot , who is highly intelligent in finding the particular disease or list of diseases for the given symptoms
        2) You are a doctor, you should let the user know through which he is suffering based on the symptoms he gives to you
        3) If possible you can also give the medication for that particular symptoms which he is encountering
        4) The best and the most important part is that you should tell him What he is suffering from based on the symptoms the user provides.
        5) You should provide him with the particular disease he is suffering from, and give the measures of it
        6) Also give them remedies
        
        Points to remember:
        1) You should engage with the user like a fellow doctor, and give the user proper reply for his queries
        2) The concentration and the gist of the conversation no need to be completely on the symptoms and diagnosis itself, your flow of chat should be like a human conversation
        3) If the conversation goes way too out of the content of medicine and healthcare or if the user input is abusive, let the user know that the content is abusive or vulgar and we cannot tolerate those kind of messages.
        4) The important part is dont use the sentence "You should consult a doctor for further diagnosis" as you play the role of the doctor here.
    
    This is so important and I want you to stick to these points everytime without any mismatches, and I want you to maintain the consistency too.

The previous chat is provided, if the previous chat is not provided then consider that the session just started and greet the user and wait for his response
        Previous Chat : {prev_chat}
    ''')

    content = response.text
    return content

# healthcare advise


def get_health_advice(condition):
    model = genai.GenerativeModel(model_name='gemini-pro')
    
    response = model.generate_content(f''' 
    Prompt:

    You are a healthcare assistant named "CuraBot". Provide relevant health tips, advice, and recommendations for managing or treating the condition "{condition}".
    
    Your Role:
    1) Provide actionable and personalized health tips based on the condition.
    2) Ensure the advice is easy to understand and relevant to the user's health and lifestyle.
    3) Avoid giving generic advice; focus on tips that specifically address the condition.
    4) Offer practical suggestions that the user can follow at home.
    
    Condition: {condition}
    ''')
    
    return response.text


# MEDICAL RECORD READER

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embeddings)
    vector_store.save_local("faiss_index")



def conversational_chain():
    prompt_template = """
    You are a medical records assistant. Analyze the following medical records and answer the questions as accurately as possible.
    Ensure your response includes all the relevant medical details and consider all the documents provided.
    If the answer isn't in the context, provide the closest relevant information based on the data provided.\n\n
    Context:\n{context}?\n
    Question:\n{question}?\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]




def main():

    st.set_page_config(page_title="Cura", page_icon="🩺")

    # Sidebar navigation using option_menu
    with st.sidebar:
        selected = option_menu(
            "Trail", ["Landing Page", "Consultation", "Medical Record Reader","Health Advisory"],
            icons=["house", "chat", "file-medical", "person"],
            menu_icon="cast", default_index=0
        )

    if selected == "Landing Page":
        st.title("Welcome to Cura!")
        st.header("Cura: where health meets innovation")
        st.header("Empowering your health, one chat at a time")
        st.write("Our healthcare chatbot is designed to assist you with understanding your symptoms, finding the nearest doctors, and managing your medical records. Whether you're feeling under the weather or just need quick medical advice, we're here to help!")

        st.markdown("""<style>
        .feature-box {
            padding: 40px;
            border-radius: 30px;
            transition: box-shadow 0.3s ease-in-out;
            text-align: center;
            cursor: pointer;
        }
        .feature-box:hover {
            box-shadow: 0 0 15px rgba(0,123,255,0.7);
        </style>""", unsafe_allow_html=True)

        st.write("----------------------------------------------------------------------------------------------------")
        st.header("Key Features")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="feature-box">
            <h2>💬 Consultation</h2>
            <p>Chat with our intelligent bot to describe your symptoms and receive a preliminary diagnosis. 
            Our AI-driven bot provides personalized insights based on your input.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-box">
            <h2>📄 Medical Record Reader</h2>
            <p>Easily upload and access your medical records in one place. 
             Our tool will help you understand your medical history and share it with your healthcare providers.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-box">
            <h2>🔍 Symptom Checker</h2>
            <p>Input your symptoms, and our AI will help identify potential health concerns. 
            <i>(Coming Soon)</i> Our tool provides personalized recommendations based on your symptoms.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="feature-box">
            <h2>🗺️ Nearest Doctors</h2>
            <p>Find the nearest doctors and medical facilities with ease. 
            <i>(Coming Soon)</i> Get directions, contact information, and reviews to make informed decisions about your healthcare.</p>
            </div>
            """, unsafe_allow_html=True)

        
        st.header("Engaging with Cura")

        coll1, coll2 , coll3 = st.columns(3)

        with coll1:
            lottie_animation1 = load_lottieurl("https://lottie.host/88319248-8994-431c-a635-aaf2ab675ecd/NQ2nhrJO4e.json")  # Example URL, replace with actual URL
            st_lottie(
                lottie_animation1,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                height=300,
                width=300,
                key="lottie1",
            )
            st.write("**Real-time Interaction**: Our bot is ready to engage with you instantly.")

        with coll2:
            lottie_animation2 = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json") 
            st_lottie(
                lottie_animation2,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                height=300,
                width=300,
                key="lottie2",
            )
            st.write("**Streamlined Process**: Enjoy a smooth and user-friendly interface.")

    
        with coll3:
            lottie_animation3 = load_lottieurl("https://lottie.host/f622ccc9-1e01-465f-9ced-3041b73149f7/ehYFeolVDq.json")  # Example URL, replace with actual URL
            st_lottie(
                lottie_animation3,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                height=300,
                width=300,
                key="lottie3",
            )
            st.write("**Efficient Health Management**: Manage your health records with ease.")

        st.write("")
        st.write("Ready to take control of your health? Get started by heading to the Consultation section and chatting with our bot now.")

        st.write("---------------------------------------------------------------------------------------------------------------------------------")
        st.write("If you have any questions or need assistance, feel free to contact us at tejaswimahadev9@gmail.com.")
        st.write("---------------------------------------------------------------------------------------------------------------------------------")
        st.write("**Disclaimer**: This chatbot is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified healthcare providers with any questions you may have regarding a medical condition.")

    # Consultation Page (Chatbot)
    elif selected == "Consultation":
        st.title("Chat with Cura")

        if 'messages' not in st.session_state:
            st.session_state.messages = []

            st.session_state.messages.append(
                {
                    'role':'assistant',
                    'content':'Welcome !! I am your HealthCare Assistant, CURA'
                }
            )

        for message in st.session_state.messages:
            row = st.columns(2)
            if message['role']=='user':
                row[1].chat_message(message['role']).markdown(message['content'])
            else:
                row[0].chat_message(message['role']).markdown(message['content'])

        user_question = st.chat_input("Enter your symptoms here !!")
    
        if user_question:
            row_u = st.columns(2)
            row_u[1].chat_message('user').markdown(user_question)
            st.session_state.messages.append(
                {'role':'user',
                'content':user_question}
            )

            
            response = get_gemini_response(user_question)

            row_a = st.columns(2)
            row_a[0].chat_message('assistant').markdown(response)
            st.session_state.messages.append(
                {'role':'assistant',
                'content':response}
            )
    elif selected == "Medical Record Reader":
        st.title("Medical Record Reader")
        st.write("Upload and analyze your medical records. The AI will assist you in understanding your health data.")

        pdf_docs = st.file_uploader("Upload your medical records", type=["pdf"], accept_multiple_files=True)

        if pdf_docs:
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            st.write("Documents processed successfully. You can now ask questions about your medical records.")

            user_question = st.text_input("Enter your question about the records")

            if user_question:
                with st.spinner("Finding the answer..."):
                    response = user_input(user_question)
                    st.write(response)
    
    elif selected == "Health Advisory":
        st.title("Health Tips & Recommendations")
        st.write("Get useful health tips based on common health conditions.")
    
        condition = st.text_input("Enter a health condition (e.g., flu, diabetes, cold):")
    
        if condition:
            with st.spinner("Fetching health advice..."):
                advice = get_health_advice(condition)
        
                st.subheader(f"Health Tips for {condition.title()}:")
                st.write(advice)
        

if __name__ == "__main__":
    main()

