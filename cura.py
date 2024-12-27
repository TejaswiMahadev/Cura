import os
import json
import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Consultation functions
def get_diagnosis_chain():
    prompt_template = """
    Given the following symptoms: {symptoms}, provide a possible diagnosis and suggest prevention measures,
    and recommend possible cures and treatments and list any relevant medication that could help.Ensure the response is clear, concise, and considers all listed symptoms.
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["symptoms"])
    chain = LLMChain(llm=model, prompt=prompt, output_key='diagnosis', memory=ConversationBufferMemory())
    return chain

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist with your symptoms today?"}]
    st.session_state.symptoms = []

def process_input(user_input):
    greetings = ["hey", "hello", "helloo"]
    symptom_queries = ["Could you help me analyze my symptoms", "I need a diagnosis", "symptom analysis", "what's wrong with me", "diagnosis"]
    user_input_lower = user_input.lower()

    if any(greet in user_input_lower for greet in greetings):
        response = "Hello! How can I assist you today?"

    elif any(query in user_input_lower for query in symptom_queries):
        if st.session_state.symptoms:
            symptoms = ','.join(sorted(set(st.session_state.symptoms)))
            chain = get_diagnosis_chain()
            response = chain.run({"symptoms": symptoms})

            if len(response.split()) < 20 or response.count('diagnosis') > 1:
                response += "The response may be incomplete or repetitive. Please provide more detailed symptoms or clarify the current ones."
            st.session_state.symptoms = []
        else:
            response = "I can help with diagnosing your symptoms. Please list all your symptoms so I can assist you effectively. Prompt (done) after you are done listing your symptoms."

    elif user_input_lower == 'done':
        if st.session_state.symptoms:
            symptoms = ','.join(sorted(set(st.session_state.symptoms)))
            chain = get_diagnosis_chain()
            response = chain.run({"symptoms": symptoms})

            if len(response.split()) < 20 or response.count('diagnosis') > 1:
                response += " The response may be incomplete or repetitive. Please provide more detailed symptoms or clarify the current ones."
            
            response+= "\n\nI hope you get well soon!"

            st.session_state.symptoms = []  # Clear symptoms after diagnosis
        else:
            response = "You haven't listed any symptoms. Please list your symptoms before typing 'done'."

    else:
        st.session_state.symptoms.append(user_input)
        response = f"Thank you for sharing. Current symptoms: {', '.join(st.session_state.symptoms)}"

    return response

#health advisory section 
def get_health_advice(condition):
    model = genai.GenerativeModel(model_name='gemini-pro')
    
    # Main health tips and advice generation
    response = model.generate_content(f'''
    Prompt:
    
    You are a healthcare assistant named "CuraBot". Provide relevant health tips, advice, and recommendations for managing or treating the condition "{condition}".
    
    Your Role:
    1) Provide actionable and personalized health tips based on the condition.
    2) Include relevant lifestyle changes, such as diet and exercise recommendations.
    3) Ensure the advice is easy to understand and relevant to the user's health and lifestyle.
    4) Avoid giving generic advice; focus on tips that specifically address the condition.
    5) Offer practical suggestions that the user can follow at home.
    6) If possible, recommend over-the-counter medication or natural remedies for the condition, along with their side effects.
    7) Provide preventive tips that help avoid worsening the condition.
    
    Condition: {condition}
    ''')
    
    health_tips = response.text

    # Severity assessment and urgency indicator
    severity_response = model.generate_content(f'''
    Prompt:
    
    You are a healthcare bot. Based on the condition "{condition}", rate the severity of this condition on a scale from 1 (mild) to 10 (critical). Include a brief explanation of why.
    Also, suggest whether the user should consult a doctor urgently or if it is safe to manage the condition at home.
    
    Condition: {condition}
    ''')
    
    severity_assessment = severity_response.text

    # General lifestyle advice
    lifestyle_response = model.generate_content(f'''
    Prompt:
    
    As a healthcare assistant named "CuraBot", provide additional lifestyle and preventive tips that would benefit someone suffering from "{condition}". These could include diet, exercise, sleep, and stress management recommendations.
    
    Condition: {condition}
    ''')
    
    lifestyle_tips = lifestyle_response.text

    return f"{health_tips}\n\n**Severity Assessment**:\n{severity_assessment}\n\n**Lifestyle & Preventive Tips**:\n{lifestyle_tips}"




# Medical record reader
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks 

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks provided for embeddings.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Generate embeddings for chunks directly using FAISS
    try:
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Error creating FAISS vector store: {e}")


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
# Main function
def main():
    st.set_page_config(page_title="Cura", page_icon="🩺")


    with st.sidebar:
        selected = option_menu(
            "Trail", ["Landing Page","Consultation","Health Advisory", "Medical Record Reader", "Nearest Doctors"],
            icons=["house", "chat", "file-medical", "map-marker-alt"],
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
    
    elif selected == "Consultation":
        st.title("Chat with Cura")
        st.write("Describe your symptoms, ask for a diagnosis, or simply say hello!")

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "How can I assist you with your symptoms today?"}]
        if "symptoms" not in st.session_state:
            st.session_state.symptoms = []

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            if st.session_state.messages[-1]["role"] == "user":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = process_input(prompt)
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.symptoms:
            st.write(f"Current Symptoms: {', '.join(st.session_state.symptoms)}")
    
    elif selected == "Health Advisory":
        st.title("Health Tips & Recommendations")
        st.write("Get useful health tips based on common health conditions.")
    
        condition = st.text_input("Enter a health condition (e.g., flu, diabetes, cold):")
    
        if condition:
            with st.spinner("Fetching health advice..."):
                advice = get_health_advice(condition)
            
                st.subheader(f"Health Tips for {condition.title()}:")
                st.write(advice)

                # Add an interactive severity self-assessment for user engagement
                st.write("### Severity Self-Assessment")
                st.write(f"To better assess your condition based on {condition}, answer the following questions:")

            # Add relevant symptom questions based on condition
                if condition.lower() == "flu":
                    st.checkbox("Do you have a fever?")
                    st.checkbox("Do you have a cough or sore throat?")
                    st.checkbox("Are you experiencing body aches or fatigue?")


    
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
    
    elif selected == "Nearest Doctors":
        st.title("Nearest Doctors")
        st.write("Find the closest medical facilities and healthcare professionals in your area")
        st.write("Features coming soon.")
       
        lottie_animation5 = load_lottieurl("https://lottie.host/24b45693-6a34-4e45-8a25-375233585951/DpxdFWCVeG.json")  # Example URL, replace with actual URL
        st_lottie(
                lottie_animation5,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                height=300,
                width=300,
                key="lottie3",
            )


if __name__ == "__main__":
    main()
