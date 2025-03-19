import streamlit as st
from openai import OpenAI
import os
import io
import tiktoken
import PyPDF2
import pandas as pd
import concurrent.futures
from pydub import AudioSegment
from pydub.playback import play
import logging
from typing import Optional, List
from streamlit.logger import get_logger
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import tempfile
from langchain.schema import Document
import speech_recognition as sr
import threading
import time
import queue
import numpy as np
from streamlit.runtime.scriptrunner import add_script_run_ctx
import uuid
import random
import pyaudio
import zipfile
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit configuration
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# Initialize logger
logger = get_logger(__name__)

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'voice_ai_active' not in st.session_state:
    st.session_state.voice_ai_active = False
if 'last_user_input_time' not in st.session_state:
    st.session_state.last_user_input_time = time.time()
if 'document_content' not in st.session_state:
    st.session_state.document_content = ""
if 'queries_left' not in st.session_state:
    st.session_state.queries_left = 10
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = None

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize audio queue
audio_queue = queue.Queue()

# Exit keyword
EXIT_KEYWORD = "goodbye assistant"

# Conversation fillers
FILLERS = ["Um", "Uh", "Let's see", "Well", "Hmm"]

def extract_account_and_client_names(filename: str):
    """
    Extracts account name and client name from the uploaded filename.
    The filename format is expected to be:
    accountname_intelligence_report_on_clientname_...
    """
    try:
        # Match common patterns in filenames
        filename = filename.lower()
        match_intelligence = re.search(r"(.+?)_intelligence_report_for_(.+?)(?:_|$)", filename)
        match_ABM = re.search(r"(.+?)_abm_report_for_(.+?)(?:_|$)", filename)
        if match_intelligence:
            account_name = match_intelligence.group(1).replace("_", " ").title()
            client_name = match_intelligence.group(2).replace("_", " ").title()
            return account_name, client_name, "intelligence_report"
        elif match_ABM:
            account_name = match_ABM.group(1).replace("_", " ").title()
            client_name = match_ABM.group(2).replace("_", " ").title()
            return account_name, client_name, "ABM_report"
        else:
            logger.warning(f"Filename does not match expected pattern: {filename}")
            return "Unknown Account", "Unknown Client", None
    except Exception as e:
        logger.error(f"Error extracting account and client names: {str(e)}")
        return "Unknown Account", "Unknown Client", None


def add_filler():
    return random.choice(FILLERS) + ", " if random.random() < 0.3 else ""

def speak(text):
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="coral",
            input=text
        )
        audio = AudioSegment.from_mp3(io.BytesIO(response.content))
        play(audio)
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}")

def read_file_content(uploaded_file):
    
    account_name, client_name, report = extract_account_and_client_names(uploaded_file.name)
        
    file_type = uploaded_file.type
    file_content = io.BytesIO(uploaded_file.read())
    
    try:
        if file_type == "text/plain":
            return file_content.getvalue().decode("utf-8", errors="replace"), account_name, client_name, report
        elif file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file_content)
            return " ".join(page.extract_text() for page in pdf_reader.pages), account_name, client_name, report
        elif file_type == "text/csv":
            df = pd.read_csv(file_content)
            return df.to_string(), account_name, client_name, report
        else:
            return "Error: Unsupported file type. Please upload a TXT, PDF, or CSV file."
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        return f"Error: Unable to read file. Please try again or contact support."

def generate_suggested_questions(content):
    try:
        prompt = f"""
        Based on the following content, generate 3-5 contextually relevant questions that a user might ask:

        Content:
        {content}  # Limit content to first 2000 characters

        Generate questions:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Generate relevant questions based on the given content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=None
        )
        questions = response.choices[0].message.content.split('\n')
        return [q.strip() for q in questions if q.strip()]
    except Exception as e:
        logger.error(f"Error generating suggested questions: {str(e)}")
        return []

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        account_name, client_name, report = extract_account_and_client_names(uploaded_file.name)
        
        # Store extracted names in session state
        st.session_state.account_name = account_name
        st.session_state.client_name = client_name

        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.name.endswith('.csv'):
            loader = CSVLoader(temp_file_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(temp_file_path)
        else:
            st.error("Unsupported file format")
            return None, None, []
        
        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
        
        conversation = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0),
            retriever=vectorstore.as_retriever()
        )
        
        content = "\n".join([doc.page_content for doc in documents])
        suggested_questions = generate_suggested_questions(content)
        
        st.session_state.conversation = conversation
        st.session_state.document_content = content
        st.session_state.suggested_questions = suggested_questions
        
        return conversation, content, suggested_questions
    finally:
        os.unlink(temp_file_path)

def clear_chat_history():
    st.session_state.chat_history = []
    st.session_state.conversation = None
    st.session_state.document_content = ""
    st.session_state.queries_left = 10
    st.session_state.suggested_questions = []
    st.success("Chat history cleared!")

def get_ai_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Respond in a natural, conversational manner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=None
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request. Please try again later."

def continuous_listening():
    with sr.Microphone() as source:
        while st.session_state.voice_ai_active:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=10)
                audio_queue.put(audio)
            except sr.WaitTimeoutError:
                pass

def transcribe_audio(audio):
    try:
        audio_data = audio.get_wav_data()
        result = client.audio.transcriptions.create(model="whisper-1", file=("audio.wav", audio_data))
        return result.text.strip().lower()
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return ""

def handle_conversation(text, is_voice=False):
    if text.lower() == EXIT_KEYWORD:
        st.session_state.voice_ai_active = False
        response = "Goodbye! Voice AI is now deactivated."
        if is_voice:
            speak(response)
        return response

    if st.session_state.queries_left <= 0:
        response = "You have reached the maximum number of queries. Please clear the chat history to continue."
        if is_voice:
            speak(response)
        return response

    st.session_state.last_user_input_time = time.time()
    
    if st.session_state.conversation:
        chat_history = []
        for msg in st.session_state.chat_history:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            else:
                chat_history.append(HumanMessage(content=msg[0]))
                chat_history.append(AIMessage(content=msg[1]))
        
        response = st.session_state.conversation.invoke({"question": text, "chat_history": chat_history})['answer']
    else:
        context = "\n".join([f"{msg['role'] if isinstance(msg, dict) else msg[0]}: {msg['content'] if isinstance(msg, dict) else msg[1]}" for msg in st.session_state.chat_history[-5:]])
        prompt = f"""
        Previous conversation:
        {context}

        Document content:
        {st.session_state.document_content}

        Human: {text}
        AI: Let's respond in a natural, conversational manner, taking into account the document content if relevant.
        """
        response = get_ai_response(prompt)
    
    if is_voice:
        speak(response)
    
    st.session_state.chat_history.append({"role": "user", "content": text})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.queries_left -= 1
    
    return response

def process_audio_queue():
    while st.session_state.voice_ai_active:
        try:
            audio = audio_queue.get(timeout=1)
            text = transcribe_audio(audio)
            if text:
                response = handle_conversation(text, is_voice=True)
        except queue.Empty:
            pass

def generate_podcast_script_intelligence(file_contents: str, podcast, account_name, client_name) -> str:
    if podcast == 'exec':
        prompt = f"""
Create a dynamic and engaging TED Talk script for "Next Quarter's Executive Briefing" featuring a single speaker. The talk should focus on strategic priorities and key initiatives for {account_name}, using the provided content as the foundation:

{file_contents}

Follow these detailed guidelines to ensure the talk is conversational, insightful, and actionable:

1. Introduction:
   - Start exactly with "Welcome to Next Quarter's briefing on {account_name}, prepared exclusively for {client_name}."
   - Provide an outline of what will be covered in the talk to help listeners follow along.
   - Set the tone and start the introduction with an engaging opening statement.

2. First 4 Initiatives:
   - Count the total number of initiatives mentioned in the content and state: "Out of X initiatives, we are focusing on the top 4."
   - For each of these first four initiatives:
     - Summarize its context or importance to the client in an engaging way.
     - Discuss recommended alignment with practical, actionable suggestions.
     - Highlight one relevant case study that demonstrates success and aligns with the initiative.
     - Keep each initiative discussion concise while ensuring clarity and depth.
   - Present each initiative in a natural flow without alternating speakers.

3. Other Sections:
   - Summarize content from other sections (apart from initiatives) from a sales executive's perspective.
   - Focus on actionable insights or opportunities for selling into the account.
   - Use storytelling techniques to naturally connect ideas, incorporating relevant sales and marketing buzzwords.
   - Present these sections without alternating speakers.

4. Tone and Delivery:
   - Maintain an enthusiastic and conversational tone throughout.
   - Maitain a natural pace.
   - Avoid overly long pauses and don't include forced transitions like "oh great" or "absolutely."
   - Allow for natural reactions but avoid making them feel scripted or excessive.
   - Keep transitions smooth by leading entire sections without frequent back-and-forth exchanges.

5. Structure:
   - Conclude with a summary of key takeaways and a motivational call-to-action for listeners to drive engagement.
   - End with: "You can always find more details about {account_name} in the full intelligence report provided by Next Quarter."

6. Dialogue Style:
   - Focus on natural conversational phrasing without naming the speaker.
   - Encourage lighthearted yet professional dialogue when appropriate.

7. TED Talk-Style Approach:
   - Structure the talk like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
   - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

By following these guidelines, craft a compelling TED Talk script that informs, motivates, and captivates listeners while addressing strategic priorities effectively.

Script:
"""
        
# prompt = f"""
# Create a dynamic and engaging TED Talk script for "Next Quarter's Executive Briefing" featuring a single speaker. The talk should focus on strategic priorities and key initiatives for {account_name}, using the provided content as the foundation:

# {file_contents}

# Follow these detailed guidelines to ensure the talk is conversational, insightful, and actionable:

# 1. Introduction:
#    - Start exactly with "Welcome to Next Quarter's briefing on {account_name}, prepared exclusively for {client_name}."
#    - Provide an outline of what will be covered in the talk to help listeners follow along.
#    - Set the tone and start the introduction with an engaging opening statement.

# 2. First 4 Initiatives:
#    - Count the total number of initiatives mentioned in the content and state: "Out of X initiatives, we are focusing on the top 4."
#    - For each of these first four initiatives:
#      - Summarize its context or importance to the client in an engaging way.
#      - Discuss recommended alignment with practical, actionable suggestions.
#      - Highlight one relevant case study that demonstrates success and aligns with the initiative.
#      - Keep each initiative discussion concise (approximately 130 words) while ensuring clarity and depth.
#    - Present each initiative in a natural flow without alternating speakers.

# 3. Other Sections:
#    - Summarize content from other sections (apart from initiatives) from a sales executive's perspective.
#    - Focus on actionable insights or opportunities for selling into the account.
#    - Use storytelling techniques to naturally connect ideas, incorporating relevant sales and marketing buzzwords.
#    - Present these sections without alternating speakers.

# 4. Tone and Delivery:
#    - Maintain an enthusiastic and conversational tone throughout.
#    - Maitain a natural pace.
#    - Avoid overly long pauses and don't include forced transitions like "oh great" or "absolutely."
#    - Allow for natural reactions but avoid making them feel scripted or excessive.
#    - Keep transitions smooth by leading entire sections without frequent back-and-forth exchanges.

# 5. Structure:
#    - Conclude with a summary of key takeaways and a motivational call-to-action for listeners to drive engagement.
#    - End with: "You can always find more details about {account_name} in the full intelligence report provided by Next Quarter."
#    - Ensure the entire script exceeds 1000 words for depth and coverage.

# 6. Dialogue Style:
#    - Focus on natural conversational phrasing without naming the speaker.
#    - Encourage lighthearted yet professional dialogue when appropriate.

# 7. TED Talk-Style Approach:
#    - Structure the talk like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
#    - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

# By following these guidelines, craft a compelling TED Talk script that informs, motivates, and captivates listeners while addressing strategic priorities effectively.

# Script:
# """
        
#    - Use this dialogue format for every dialogue of Host:
#      - Host: [Host's dialogue]
#      - Continue this pattern throughout the script, leading each section for better flow.




    

    if podcast == 'businessoverview':
        prompt = f"""
Continue the "Next Quarter's Executive Briefing" TED Talk, focusing on the Business Overview and SWOT analysis based on the provided content:

{file_contents}

Follow these detailed guidelines to ensure a seamless, engaging, and insightful continuation of the talk:

1. Transition:
   - Begin with a smooth transition from the previous segment about initiatives.
   - Provide a brief outline of what will be covered in this segment (Business Overview, SWOT Analysis, Roadmap Commentary, Market and Industry Trends).

2. SWOT Analysis:
   - Highlight one key strength and one significant weakness from the SWOT analysis:
     - Provide real-world examples or context for each point to make them relatable.
     - Discuss their impact on business strategy or operations in practical terms.
   - Present both the strength and weakness in a natural flow without alternating speakers.

3. Roadmap Commentary:
   - Offer concise insights into the current and future roadmap:
     - Mention key milestones or goals that stand out.
     - Explain how the roadmap addresses both the highlighted strength and weakness effectively.
   - Present this section as part of the overall narrative.

4. Market Trend:
   - Identify one relevant market trend:
     - Discuss its potential impact on business strategy.
     - Suggest actionable ways the company could leverage this trend to its advantage.
   - Integrate this section smoothly into the talk.

5. Industry Trend:
   - Highlight one significant industry trend:
     - Analyze its implications for competitors and overall market dynamics.
     - Propose strategies for addressing this trend to maintain a competitive edge.
   - Present this section as a continuation of the market trend discussion.

6. Tone and Storytelling:
   - Use storytelling techniques to dynamically connect insights and ideas, keeping the dialogue natural and engaging.
   - Maintain an upbeat tone while focusing on smooth, natural phrasing.
   - Avoid overly long pauses or forced transitions like "oh great" or "absolutely."

7. Conclusion:
   - Summarize key points discussed in this segment concisely.
   - End with a teaser for the upcoming Competitors' section to keep listeners intrigued.

8. Dialogue Style:
   - Focus on natural conversational phrasing without naming the speaker.
   - Encourage lighthearted yet professional dialogue when appropriate.

9. TED Talk-Style Approach:
    - Structure this segment like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
    - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

By following these updated guidelines, craft a concise yet impactful TED Talk segment that informs, engages, and transitions smoothly into the next topic.

Script:
"""
        

# prompt = f"""
# Continue the "Next Quarter's Executive Briefing" TED Talk, focusing on the Business Overview and SWOT analysis based on the provided content:

# {file_contents}

# Follow these detailed guidelines to ensure a seamless, engaging, and insightful continuation of the talk:

# 1. Transition:
#    - Begin with a smooth transition from the previous segment about initiatives.
#    - Provide a brief outline of what will be covered in this segment (Business Overview, SWOT Analysis, Roadmap Commentary, Market and Industry Trends).

# 2. SWOT Analysis:
#    - Highlight one key strength and one significant weakness from the SWOT analysis:
#      - Provide real-world examples or context for each point to make them relatable.
#      - Discuss their impact on business strategy or operations in practical terms.
#    - Present both the strength and weakness in a natural flow without alternating speakers.

# 3. Roadmap Commentary:
#    - Offer concise insights into the current and future roadmap:
#      - Mention key milestones or goals that stand out.
#      - Explain how the roadmap addresses both the highlighted strength and weakness effectively.
#    - Present this section as part of the overall narrative.

# 4. Market Trend:
#    - Identify one relevant market trend:
#      - Discuss its potential impact on business strategy.
#      - Suggest actionable ways the company could leverage this trend to its advantage.
#    - Integrate this section smoothly into the talk.

# 5. Industry Trend:
#    - Highlight one significant industry trend:
#      - Analyze its implications for competitors and overall market dynamics.
#      - Propose strategies for addressing this trend to maintain a competitive edge.
#    - Present this section as a continuation of the market trend discussion.

# 6. Tone and Storytelling:
#    - Use storytelling techniques to dynamically connect insights and ideas, keeping the dialogue natural and engaging.
#    - Maintain an upbeat tone while focusing on smooth, natural phrasing.
#    - Avoid overly long pauses or forced transitions like "oh great" or "absolutely."

# 7. Conclusion:
#    - Summarize key points discussed in this segment concisely.
#    - End with a teaser for the upcoming Competitors' section to keep listeners intrigued.

# 8. Length:
#    - Keep this segment concise, approximately 500 words, while ensuring depth and clarity.

# 9. Dialogue Style:
#    - Focus on natural conversational phrasing without naming the speaker.
#    - Encourage lighthearted yet professional dialogue when appropriate.

# 10. TED Talk-Style Approach:
#     - Structure this segment like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
#     - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

# By following these updated guidelines, craft a concise yet impactful TED Talk segment that informs, engages, and transitions smoothly into the next topic.

# Script:
# """
        

    #        - Use this dialogue format for every dialogue of Host:
    #  - Host: [Host's dialogue]
    #  - Continue this pattern throughout the script, leading each section for better flow.





    if podcast == 'competitors':
        prompt = f"""
Continue the "Next Quarter's Executive Briefing" TED Talk, shifting the focus to Competitors based on the provided content:

{file_contents}

Follow these detailed guidelines to ensure a seamless, engaging, and insightful continuation of the talk:

1. Transition:
   - Begin with a smooth transition from the Business Overview segment.
   - Provide a brief outline of what will be covered in this segment (Competitor Analysis, Client Gaps, Detailed Competitor Spotlight, and Countermeasures) to help listeners follow along.

2. Competitor Analysis:
   - Discuss key competitors:
     - Provide an overview of their strengths relative to the client.
     - Use specific examples to illustrate competitive dynamics and market positioning.
   - Present this section as part of the overall narrative.

3. Client Gaps:
   - Identify two areas where the client is lagging behind competitors:
     - Analyze the implications of these gaps on market position or sales performance.
     - Suggest actionable strategies to address these weaknesses effectively.
   - Integrate this section smoothly into the talk.

4. Detailed Competitor Spotlight:
   - Highlight one specific competitor in detail:
     - Discuss their unique sales tactics or strategies.
     - Analyze how these contribute to their success and differentiate them in the market.
   - Present this section as a continuation of the competitor analysis.

5. Countermeasures:
   - Present practical countermeasures for the client:
     - Suggest strategies that could bolster competitive positioning and mitigate risks effectively.
   - Integrate this section naturally into the narrative.

6. Tone and Engagement:
   - Maintain an engaging and conversational tone throughout, focusing on smooth, natural phrasing.
   - Avoid overly long pauses or forced transitions like "oh great" or "absolutely."
   - Use relevant buzzwords naturally within discussions to resonate with a professional audience.

7. Conclusion:
   - Summarize key points discussed in this segment concisely.
   - End with a teaser for the upcoming Stakeholders' section to keep listeners intrigued.

8. Dialogue Style:
   - Focus on natural conversational phrasing without naming the speaker.
   - Encourage lighthearted yet professional dialogue when appropriate.

9. TED Talk-Style Approach:
    - Structure this segment like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
    - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

By following these updated guidelines, craft a concise yet impactful TED Talk segment that informs, engages, and transitions smoothly into the next topic.

Script:
"""
        

# prompt = f"""
# Continue the "Next Quarter's Executive Briefing" TED Talk, shifting the focus to Competitors based on the provided content:

# {file_contents}

# Follow these detailed guidelines to ensure a seamless, engaging, and insightful continuation of the talk:

# 1. Transition:
#    - Begin with a smooth transition from the Business Overview segment.
#    - Provide a brief outline of what will be covered in this segment (Competitor Analysis, Client Gaps, Detailed Competitor Spotlight, and Countermeasures) to help listeners follow along.

# 2. Competitor Analysis:
#    - Discuss key competitors:
#      - Provide an overview of their strengths relative to the client.
#      - Use specific examples to illustrate competitive dynamics and market positioning.
#    - Present this section as part of the overall narrative.

# 3. Client Gaps:
#    - Identify two areas where the client is lagging behind competitors:
#      - Analyze the implications of these gaps on market position or sales performance.
#      - Suggest actionable strategies to address these weaknesses effectively.
#    - Integrate this section smoothly into the talk.

# 4. Detailed Competitor Spotlight:
#    - Highlight one specific competitor in detail:
#      - Discuss their unique sales tactics or strategies.
#      - Analyze how these contribute to their success and differentiate them in the market.
#    - Present this section as a continuation of the competitor analysis.

# 5. Countermeasures:
#    - Present practical countermeasures for the client:
#      - Suggest strategies that could bolster competitive positioning and mitigate risks effectively.
#    - Integrate this section naturally into the narrative.

# 6. Tone and Engagement:
#    - Maintain an engaging and conversational tone throughout, focusing on smooth, natural phrasing.
#    - Avoid overly long pauses or forced transitions like "oh great" or "absolutely."
#    - Use relevant buzzwords naturally within discussions to resonate with a professional audience.

# 7. Conclusion:
#    - Summarize key points discussed in this segment concisely.
#    - End with a teaser for the upcoming Stakeholders' section to keep listeners intrigued.

# 8. Dialogue Style:
#    - Focus on natural conversational phrasing without naming the speaker.
#    - Encourage lighthearted yet professional dialogue when appropriate.

# 9. TED Talk-Style Approach:
#     - Structure this segment like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
#     - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

# By following these updated guidelines, craft a concise yet impactful TED Talk segment that informs, engages, and transitions smoothly into the next topic.

# Script:
# """





    if podcast == 'stakeholders':
        prompt = f"""
Conclude the "Next Quarter's Executive Briefing" TED Talk by focusing on Key Stakeholders based on the provided content:

{file_contents}

Follow these detailed guidelines to craft a smooth, engaging, and impactful conclusion to the talk:

1. Transition:
   - Begin with a seamless and engaging transition from the previous segment about Competitors.
   - Emphasize that this is the final and most critical piece of the briefing.
   - Provide a brief outline of what will be covered in this segment (Stakeholder Identification, Key Executives, Stakeholder Contributions, and Relationship Strategies).

2. Stakeholder Identification:
   - Identify stakeholders frequently mentioned across initiatives:
     - Refer to Personas supporting these initiatives.
     - Explain why these stakeholders are crucial to multiple efforts and how they influence success.
   - Present this section as part of the overall narrative.

3. Key Executives:
   - Select 3-5 key executives from the identified stakeholders and for each:
     - Highlight the specific initiatives they are focused on.
     - Provide a brief bio from the key contacts summary section, including:
       * Their role and responsibilities.
       * Notable achievements or areas of expertise.
       * How their background aligns with the initiatives they support.
   - Integrate this section smoothly into the talk.

4. Stakeholder Contributions:
   - Draw connections between stakeholders’ expertise and their contributions to initiatives:
     - Discuss how their unique skills drive success.
     - Highlight potential synergies between different stakeholders' efforts.
   - Present this section as a continuation of the key executives discussion.

5. Relationship Strategies:
   - Discuss strategies for building strong relationships with these key stakeholders:
     - Suggest practical engagement methods tailored to each stakeholder’s priorities.
     - Emphasize how strong relationships can directly impact initiative outcomes.
   - Integrate this section naturally into the narrative.

6. Tone and Storytelling:
   - Incorporate relevant sales and marketing buzzwords naturally into the conversation while maintaining an upbeat tone.
   - Use storytelling techniques to make the discussion relatable and engaging, referencing real-world examples where appropriate.

7. Conclusion of Segment:
   - Summarize key takeaways about stakeholders’ roles in driving initiatives:
     - Reinforce the importance of understanding their motivations and aligning strategies accordingly.
     - Provide a motivational call-to-action for listeners, encouraging them to leverage these insights in their own strategic planning.

8. Talk Closing:
   - End with a reflective closing statement:
     - Thank listeners for joining this deep dive into strategic priorities.
     - Encourage them to apply what they’ve learned to build stronger partnerships and achieve their goals.

9. Dialogue Style:
    - Focus on natural conversational phrasing without naming the speaker.
    - Encourage lighthearted yet professional dialogue when appropriate.

10. TED Talk-Style Approach:
    - Structure this segment like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
    - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

By following these updated guidelines, craft a concise yet impactful TED Talk segment that concludes the series with actionable insights while leaving listeners motivated and inspired.

Script:
"""
        

#         prompt = f"""
# Conclude the "Next Quarter's Executive Briefing" TED Talk by focusing on Key Stakeholders based on the provided content:

# {file_contents}

# Follow these detailed guidelines to craft a smooth, engaging, and impactful conclusion to the talk:

# 1. Transition:
#    - Begin with a seamless and engaging transition from the previous segment about Competitors.
#    - Emphasize that this is the final and most critical piece of the briefing.
#    - Provide a brief outline of what will be covered in this segment (Stakeholder Identification, Key Executives, Stakeholder Contributions, and Relationship Strategies).

# 2. Stakeholder Identification:
#    - Identify stakeholders frequently mentioned across initiatives:
#      - Refer to Personas supporting these initiatives.
#      - Explain why these stakeholders are crucial to multiple efforts and how they influence success.
#    - Present this section as part of the overall narrative.

# 3. Key Executives:
#    - Select 3-5 key executives from the identified stakeholders and for each:
#      - Highlight the specific initiatives they are focused on.
#      - Provide a brief bio from the key contacts summary section, including:
#        * Their role and responsibilities.
#        * Notable achievements or areas of expertise.
#        * How their background aligns with the initiatives they support.
#    - Integrate this section smoothly into the talk.

# 4. Stakeholder Contributions:
#    - Draw connections between stakeholders’ expertise and their contributions to initiatives:
#      - Discuss how their unique skills drive success.
#      - Highlight potential synergies between different stakeholders' efforts.
#    - Present this section as a continuation of the key executives discussion.

# 5. Relationship Strategies:
#    - Discuss strategies for building strong relationships with these key stakeholders:
#      - Suggest practical engagement methods tailored to each stakeholder’s priorities.
#      - Emphasize how strong relationships can directly impact initiative outcomes.
#    - Integrate this section naturally into the narrative.

# 6. Tone and Storytelling:
#    - Incorporate relevant sales and marketing buzzwords naturally into the conversation while maintaining an upbeat tone.
#    - Use storytelling techniques to make the discussion relatable and engaging, referencing real-world examples where appropriate.

# 7. Conclusion of Segment:
#    - Summarize key takeaways about stakeholders’ roles in driving initiatives:
#      - Reinforce the importance of understanding their motivations and aligning strategies accordingly.
#      - Provide a motivational call-to-action for listeners, encouraging them to leverage these insights in their own strategic planning.

# 8. Talk Closing:
#    - End with a reflective closing statement:
#      - Thank listeners for joining this deep dive into strategic priorities.
#      - Encourage them to apply what they’ve learned to build stronger partnerships and achieve their goals.

# 9. Length:
#    - Keep this segment concise yet impactful, approximately 400 words.

# 10. Dialogue Style:
#     - Focus on natural conversational phrasing without naming the speaker.
#     - Encourage lighthearted yet professional dialogue when appropriate.

# 11. TED Talk-Style Approach:
#     - Structure this segment like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
#     - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

# By following these updated guidelines, craft a concise yet impactful TED Talk segment that concludes the series with actionable insights while leaving listeners motivated and inspired.

# Script:
# """




    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional podcast scriptwriter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=None
        )
        print(response)
        raw_script = response.choices[0].message.content
        return process_script(raw_script)
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        return f"Error: Unable to generate script. Please try again or contact support."
    
def generate_podcast_script_ABM(file_contents: str, account_name, client_name) -> str:
    prompt = f"""
Create a dynamic and engaging TED Talk on "Next Quarter's ABM Deep Dive" focusing on strategically important personas for {account_name}, using the provided content as the foundation:

{file_contents}

Follow these detailed guidelines to ensure the talk is conversational, insightful, and actionable:

1. Introduction:
   - Start exactly with "Welcome to Next Quarter's ABM summary on {account_name}'s key personas, prepared exclusively for {client_name}."
   - Provide a concise outline of the talk's key areas: top ABM strategies, persona overview with themes and initiatives, and Account-Based Marketing channel strategies. Emphasize how understanding these elements can drive more effective engagement.

2. Top ABM Strategies:
   - Summarize the top 3 recommendations from the "Potential ABM Strategies Based on Intelligence" section.
   - For each strategy:
     - Provide a brief overview of the strategy.
     - Highlight key actions and targeted personas.
     - Discuss the potential impact and alignment with {client_name}'s offerings (approximately 50 words).

3. Persona Deep Dive:
   - Reference the "Persona Based Overlap of Themes and Initiatives" and "Channel Prediction for ABM to Identified Personas" sections.
   - Merge the information from these tables to create a comprehensive view of each persona.
   - Identify the top 4 personas based on strategic importance and relevance to {client_name}.
   - For each selected persona:
     - Persona Summary:
       - Clearly state the persona's name and job title. Provide a concise summary of their core responsibilities and overall influence within {account_name}.
     - Themes and Initiatives:
       - Summarize the key themes and initiatives associated with this persona, focusing on those most relevant to {client_name}'s offerings.
     - ABM Channels:
       - Detail the recommended ABM channels for engaging this persona. Explain how these channels align with the persona's preferences and information consumption habits.
   - Keep each persona discussion concise, while focusing on actionable insights.

4. Tone and Delivery:
   - Maintain an enthusiastic and conversational tone throughout the talk.
   - Avoid overly long pauses and forced transitions.
   - Use storytelling techniques to make the discussion relatable and engaging, referencing real-world examples where appropriate.

5. Structure:
   - Conclude with a summary of key takeaways, reinforcing the importance of understanding these personas, aligning with their initiatives, and leveraging appropriate ABM channels. Include a clear and motivational call to action, encouraging listeners to take concrete steps to engage these individuals.
   - End with: "You can always find more details about {account_name}'s key influencers in the full ABM report provided by Next Quarter."

6. Dialogue Style:
   - Focus on natural conversational phrasing without naming the speaker.
   - Encourage lighthearted yet professional dialogue when appropriate.

7. Strategic Focus (TED Talk Style):
   - Emulate a TED Talk by clearly outlining key points at the beginning, delving into each strategy and persona with engaging, story-driven insights, and concluding with an inspiring and actionable summary.
   - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights about influencing these personas and driving meaningful engagement through appropriate ABM channels.

By following these guidelines, create a compelling TED Talk that is informative, motivating, and captivates listeners while effectively addressing strategic priorities. Focus on providing actionable advice for engaging key personas through targeted ABM strategies and channels to drive results for {client_name}.

Script:
"""
    
# prompt = f"""
# Create a dynamic and engaging TED Talk on "Next Quarter's ABM Deep Dive" focusing on strategically important personas for {account_name}, using the provided content as the foundation:

# {file_contents}

# Follow these detailed guidelines to ensure the talk is conversational, insightful, and actionable:

# 1. Introduction:
#    - Start exactly with "Welcome to Next Quarter's ABM summary on {account_name}'s key personas, prepared exclusively for {client_name}."
#    - Provide a concise outline of the talk's key areas: top ABM strategies, persona overview with themes and initiatives, and Account-Based Marketing channel strategies. Emphasize how understanding these elements can drive more effective engagement.

# 2. Top ABM Strategies:
#    - Summarize the top 3 recommendations from the "Potential ABM Strategies Based on Intelligence" section.
#    - For each strategy:
#      - Provide a brief overview of the strategy (approximately 50 words).
#      - Highlight key actions and targeted personas (approximately 50 words).
#      - Discuss the potential impact and alignment with {client_name}'s offerings (approximately 50 words).

# 3. Persona Deep Dive:
#    - Reference the "Persona Based Overlap of Themes and Initiatives" and "Channel Prediction for ABM to Identified Personas" sections.
#    - Merge the information from these tables to create a comprehensive view of each persona.
#    - Identify the top 4 personas based on strategic importance and relevance to {client_name}.
#    - For each selected persona:
#      - Persona Summary:
#        - Clearly state the persona's name and job title. Provide a concise summary of their core responsibilities and overall influence within {account_name} (approximately 60 words).
#      - Themes and Initiatives:
#        - Summarize the key themes and initiatives associated with this persona, focusing on those most relevant to {client_name}'s offerings (approximately 100 words).
#      - ABM Channels:
#        - Detail the recommended ABM channels for engaging this persona. Explain how these channels align with the persona's preferences and information consumption habits (approximately 60 words).
#    - Keep each persona discussion concise (approximately 220 words total per persona), while focusing on actionable insights.

# 4. Tone and Delivery:
#    - Maintain an enthusiastic and conversational tone throughout the talk.
#    - Avoid overly long pauses and forced transitions.
#    - Use storytelling techniques to make the discussion relatable and engaging, referencing real-world examples where appropriate.

# 5. Structure:
#    - Conclude with a summary of key takeaways, reinforcing the importance of understanding these personas, aligning with their initiatives, and leveraging appropriate ABM channels. Include a clear and motivational call to action, encouraging listeners to take concrete steps to engage these individuals.
#    - End with: "You can always find more details about {account_name}'s key influencers in the full ABM report provided by Next Quarter."
#    - Ensure the entire script is substantial and well-developed. The target length is 900-1100 words to provide sufficient depth and coverage.

# 6. Dialogue Style:
#    - Focus on natural conversational phrasing without naming the speaker.
#    - Encourage lighthearted yet professional dialogue when appropriate.

# 7. Strategic Focus (TED Talk Style):
#    - Emulate a TED Talk by clearly outlining key points at the beginning, delving into each strategy and persona with engaging, story-driven insights, and concluding with an inspiring and actionable summary.
#    - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights about influencing these personas and driving meaningful engagement through appropriate ABM channels.

# By following these guidelines, create a compelling TED Talk that is informative, motivating, and captivates listeners while effectively addressing strategic priorities. Focus on providing actionable advice for engaging key personas through targeted ABM strategies and channels to drive results for {client_name}.

# Script:
# """
##    - Use this dialogue format for every dialogue of Host:
#      - Host: [Host's dialogue]
#      - Continue this pattern throughout the script, leading each section for better flow.


    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional podcast scriptwriter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=None
        )
        print(response)
        raw_script = response.choices[0].message.content
        return process_script(raw_script)
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        return f"Error: Unable to generate script. Please try again or contact support."

def process_script(script: str) -> str:
    processed_lines = []
    for line in script.split('\n'):
        line = line.strip()
        line = line.replace('*', '')
        # if line.startswith("Host:"):
        processed_lines.append(line)
    return '\n'.join(processed_lines)

def text_to_speech_stream(text: str, voice: str):
    try:
        logger.info(f"Starting text-to-speech conversion for voice: {voice}")
        logger.info(f"Text to convert: {text[:50]}...")  # Log first 50 characters
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text
        )
        logger.info(f"Text-to-speech conversion successful for voice: {voice}")
        return response.content
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion for voice {voice}: {str(e)}")
        return None

def create_podcast(script: str) -> Optional[AudioSegment]:
    lines = script.split('\n')
    audio_segments: List[AudioSegment] = []
    
    def process_line(line: str):
        # if line.startswith("Host:"):
        # return text_to_speech_stream(line[6:].strip(), voice="ash")
        return text_to_speech_stream(line.strip(), voice="ash")
        # return None
    
    logger.info(f"Starting audio generation for {len(lines)} lines")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        audio_streams = list(executor.map(process_line, lines))
    
    for i, (line, audio_content) in enumerate(zip(lines, audio_streams)):
        if audio_content:
            try:
                # if line.startswith("Host:"):
                voice = "ash"
                logger.info(f"Processing line {i+1}/{len(lines)} for voice: {voice}")
                logger.info(f"Generating new audio for line {i+1}")
                
                # Convert the audio content to an AudioSegment
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_content))
                
                # Adjust playback speed for "coral" voice
                # if voice == "coral":
                    # logger.info(f"Adjusting playback speed to 90% for line {i+1}")
                    # audio_segment = audio_segment.speedup(playback_speed=1.25)
                
                # Append the processed segment
                audio_segments.append(audio_segment)
                logger.info(f"Audio generated for line {i+1}")
            except Exception as e:
                logger.error(f"Error processing audio stream for line {i+1}: {line}")
                logger.error(str(e))
        else:
            logger.warning(f"No audio stream generated for line {i+1}: {line}")
    
    if audio_segments:
        try:
            logger.info(f"Combining {len(audio_segments)} audio segments")
            final_audio = sum(audio_segments)
            logger.info(f"Final audio duration: {len(final_audio)/1000:.2f} seconds")
            return final_audio
        except Exception as e:
            logger.error(f"Error combining audio segments: {str(e)}")
    else:
        logger.error("No audio segments were generated")
    
    return None

def main():
    st.title("AI Assistant")

    tab1, tab2 = st.tabs(["Conversational AI", "Audio Briefing Generator"])

    with tab1:
        st.header("Chat with AI")
        st.write(f"Queries left: {st.session_state.queries_left}")
        uploaded_file = st.file_uploader("Upload a file (PDF, TXT, CSV)", type=['pdf', 'txt', 'csv'])

        if uploaded_file is not None:
            if 'file_processed' not in st.session_state or st.session_state.file_processed != uploaded_file.name:
                st.session_state.conversation, file_contents, suggested_questions = process_uploaded_file(uploaded_file)
                if st.session_state.conversation:
                    st.success("File processed and added to knowledge base successfully!")
                    st.text_area("File Contents:", value=file_contents, height=300)
                    st.session_state.document_content = file_contents
                    st.session_state.suggested_questions = suggested_questions
                    st.session_state.file_processed = uploaded_file.name
                    # st.rerun()

        if st.button("Clear Chat History"):
            clear_chat_history()

        if st.session_state.suggested_questions:
            st.subheader("Suggested Questions:")
            for i, question in enumerate(st.session_state.suggested_questions):
                if st.button(question, key=f"question_{i}"):
                    response = handle_conversation(question)
                    st.rerun()

        for message in st.session_state.chat_history:
            role = message["role"] if isinstance(message, dict) else message[0]
            content = message["content"] if isinstance(message, dict) else message[1]
            with st.chat_message(role):
                st.markdown(content)

        prompt = st.chat_input("What is your question?")
        
        voice_ai_active = st.toggle("Activate Voice AI", key="voice_ai_toggle")

        if voice_ai_active != st.session_state.voice_ai_active:
            st.session_state.voice_ai_active = voice_ai_active
            if voice_ai_active:
                st.session_state.listening_thread = threading.Thread(target=continuous_listening, daemon=True)
                st.session_state.processing_thread = threading.Thread(target=process_audio_queue, daemon=True)
                add_script_run_ctx(st.session_state.listening_thread)
                add_script_run_ctx(st.session_state.processing_thread)
                st.session_state.listening_thread.start()
                st.session_state.processing_thread.start()
                welcome_message = "Voice AI is now active. How can I assist you today?"
                speak(welcome_message)
                st.session_state.chat_history.append({"role": "assistant", "content": welcome_message})
            else:
                if hasattr(st.session_state, 'listening_thread'):
                    st.session_state.listening_thread.join()
                if hasattr(st.session_state, 'processing_thread'):
                    st.session_state.processing_thread.join()

        if voice_ai_active:
            st.write("Voice AI is active. Speak to interact or say 'Goodbye Assistant' to exit.")
        else:
            st.write("Voice AI is inactive. Toggle the switch above to activate.")

        if prompt:
            if st.session_state.queries_left > 0:
                response = handle_conversation(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.queries_left -= 1
            else:
                st.error("You have reached the maximum number of queries. Please clear the chat history to continue.")

    with tab2:
        st.header("Audio Briefing Generator")
        podcast_file = st.file_uploader("Upload a file for briefing generation", type=["txt", "pdf", "csv"])

        if podcast_file is not None:
            file_contents, account_name, client_name, report = read_file_content(podcast_file)
            
            if file_contents.startswith("Error:"):
                st.error(file_contents)
            else:
                if st.button("Generate Briefings"):
                    with st.spinner("Generating briefing transcripts..."):
                        if report == "intelligence_report":
                            script = generate_podcast_script_intelligence(file_contents, 'exec', account_name, client_name)
                            script_businessoverview = generate_podcast_script_intelligence(file_contents, 'businessoverview', account_name, client_name)
                            script_competitors = generate_podcast_script_intelligence(file_contents, 'competitors', account_name, client_name)
                            script_stakeholders = generate_podcast_script_intelligence(file_contents, 'stakeholders', account_name, client_name)

                            if script.startswith("Error:"):
                                st.error(script)
                            else:
                                st.subheader("Generated Transcript Executive Briefing")
                                st.text_area("Transcript", script, height=300)

                                st.subheader("Generated Transcript Businness Overview")
                                st.text_area("Transcript", script_businessoverview, height=300)

                                st.subheader("Generated Transcript Competitors")
                                st.text_area("Transcript", script_competitors, height=300)

                                st.subheader("Generated Transcript Stakeholders")
                                st.text_area("Transcript", script_stakeholders, height=300)
                                
                                with st.spinner("Creating Briefing audio..."):
                                    try:
                                        podcast_audio = create_podcast(script)
                                        podcast_audio_businessoverview = create_podcast(script_businessoverview)
                                        podcast_audio_competitors = create_podcast(script_competitors)
                                        podcast_audio_stakeholders = create_podcast(script_stakeholders)

                                    except Exception as e:
                                        st.error("An error occurred while creating the briefing. Please try again or contact support.")
                                        logger.error(f"Briefing creation error: {str(e)}")
                                        podcast_audio = None
                                
                                if podcast_audio:
                                    st.subheader("Generated Executive Summary Briefing")
                                    with io.BytesIO() as audio_buffer:
                                        try:
                                            podcast_audio.export(audio_buffer, format="mp3")
                                            audio_buffer.seek(0)
                                            st.audio(audio_buffer, format="audio/mp3")
                                        
                                            st.download_button(
                                                label="Download Executive Summary Briefing",
                                                data=audio_buffer.getvalue(),
                                                file_name=f"{account_name}_executive_briefing_for_{client_name}.mp3",
                                                mime="audio/mpeg"
                                            )
                                        except Exception as e:
                                            st.error("Error exporting audio. Please try again or contact support.")
                                            logger.error(f"Audio export error: {str(e)}")
                                else:
                                    st.error("Failed to generate briefing audio. Please try again or contact support.")
                            
                                if podcast_audio_businessoverview:
                                    st.subheader("Generated Business Overview Briefing")
                                    with io.BytesIO() as audio_buffer:
                                        try:
                                            podcast_audio_businessoverview.export(audio_buffer, format="mp3")
                                            audio_buffer.seek(0)
                                            st.audio(audio_buffer, format="audio/mp3")
                                        
                                            st.download_button(
                                                label="Download Business Overview Briefing",
                                                data=audio_buffer.getvalue(),
                                                file_name=f"{account_name}_businessoverview_briefing_for_{client_name}.mp3",
                                                mime="audio/mpeg"
                                            )
                                        except Exception as e:
                                            st.error("Error exporting audio. Please try again or contact support.")
                                            logger.error(f"Audio export error: {str(e)}")
                                else:
                                    st.error("Failed to generate briefing audio. Please try again or contact support.")
                                
                                if podcast_audio_competitors:
                                    st.subheader("Generated Competitiors Briefing")
                                    with io.BytesIO() as audio_buffer:
                                        try:
                                            podcast_audio_competitors.export(audio_buffer, format="mp3")
                                            audio_buffer.seek(0)
                                            st.audio(audio_buffer, format="audio/mp3")
                                        
                                            st.download_button(
                                                label="Download Competitiors Briefing",
                                                data=audio_buffer.getvalue(),
                                                file_name=f"{account_name}_competitors_briefing_for_{client_name}.mp3",
                                                mime="audio/mpeg"
                                            )
                                        except Exception as e:
                                            st.error("Error exporting audio. Please try again or contact support.")
                                            logger.error(f"Audio export error: {str(e)}")
                                else:
                                    st.error("Failed to generate briefing audio. Please try again or contact support.")
                                
                                if podcast_audio_stakeholders:
                                    st.subheader("Generated Stakeholders Briefing")
                                    with io.BytesIO() as audio_buffer:
                                        try:
                                            podcast_audio_stakeholders.export(audio_buffer, format="mp3")
                                            audio_buffer.seek(0)
                                            st.audio(audio_buffer, format="audio/mp3")
                                        
                                            st.download_button(
                                                label="Download Stakeholders Briefings",
                                                data=audio_buffer.getvalue(),
                                                file_name=f"{account_name}_stakeholders_briefing_for_{client_name}.mp3",
                                                mime="audio/mpeg"
                                            )
                                        except Exception as e:
                                            st.error("Error exporting audio. Please try again or contact support.")
                                            logger.error(f"Audio export error: {str(e)}")
                                else:
                                    st.error("Failed to generate briefing audio. Please try again or contact support.")

                                
                                if podcast_audio_stakeholders or podcast_audio_competitors or podcast_audio_businessoverview or podcast_audio:
                                    try:
                                        # Create a ZIP file in memory
                                        with io.BytesIO() as zip_buffer:
                                            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                                                # Add each audio file to the ZIP archive
                                                if podcast_audio_stakeholders:
                                                    with io.BytesIO() as audio_buffer:
                                                        podcast_audio_stakeholders.export(audio_buffer, format="mp3")
                                                        audio_buffer.seek(0)
                                                        zip_file.writestr(f"{account_name}_stakeholders_briefing_for_{client_name}.mp3", audio_buffer.read())
                                                
                                                if podcast_audio_competitors:
                                                    with io.BytesIO() as audio_buffer:
                                                        podcast_audio_competitors.export(audio_buffer, format="mp3")
                                                        audio_buffer.seek(0)
                                                        zip_file.writestr(f"{account_name}_competitors_briefing_for_{client_name}.mp3", audio_buffer.read())
                                                
                                                if podcast_audio_businessoverview:
                                                    with io.BytesIO() as audio_buffer:
                                                        podcast_audio_businessoverview.export(audio_buffer, format="mp3")
                                                        audio_buffer.seek(0)
                                                        zip_file.writestr(f"{account_name}_businessoverview_briefing_for_{client_name}.mp3", audio_buffer.read())
                                                
                                                if podcast_audio:
                                                    with io.BytesIO() as audio_buffer:
                                                        podcast_audio.export(audio_buffer, format="mp3")
                                                        audio_buffer.seek(0)
                                                        zip_file.writestr(f"{account_name}_executive_briefing_for_{client_name}.mp3", audio_buffer.read())
                                            
                                            # Finalize the ZIP file
                                            zip_buffer.seek(0)
                                
                                            # Provide a download button for the ZIP file
                                            st.download_button(
                                                label="Download All Briefings",
                                                data=zip_buffer.getvalue(),
                                                file_name=f"{account_name}_all_briefings_for_{client_name}.zip",
                                                mime="application/zip"
                                            )
                                    except Exception as e:
                                        st.error("Error exporting audio. Please try again or contact support.")
                                        logger.error(f"Audio export error: {str(e)}")
                                else:
                                    st.error("Failed to generate briefing audio. Please try again or contact support.")
                        elif report == "ABM_report":
                            script = generate_podcast_script_ABM(file_contents, account_name, client_name)
                            script_businessoverview = None
                            script_competitors = None
                            script_stakeholders = None

                            if script.startswith("Error:"):
                                st.error(script)
                            else:
                                st.subheader("Generated Transcript ABM Briefing")
                                st.text_area("Transcript", script, height=300)

                                with st.spinner("Creating Briefing audio..."):
                                    try:
                                        podcast_audio = create_podcast(script)
                                    except Exception as e:
                                        st.error("An error occurred while creating the briefing. Please try again or contact support.")
                                        logger.error(f"Briefing creation error: {str(e)}")
                                        podcast_audio = None
                                
                                if podcast_audio:
                                    st.subheader("Generated ABM Briefing")
                                    with io.BytesIO() as audio_buffer:
                                        try:
                                            podcast_audio.export(audio_buffer, format="mp3")
                                            audio_buffer.seek(0)
                                            st.audio(audio_buffer, format="audio/mp3")
                                        
                                            st.download_button(
                                                label="Download ABM Briefing",
                                                data=audio_buffer.getvalue(),
                                                file_name=f"{account_name}_abm_briefing_for_{client_name}.mp3",
                                                mime="audio/mpeg"
                                            )
                                        except Exception as e:
                                            st.error("Error exporting audio. Please try again or contact support.")
                                            logger.error(f"Audio export error: {str(e)}")
                                else:
                                    st.error("Failed to generate briefing audio. Please try again or contact support.")
                        else:
                            st.error("Unsupported filename format.")
                            script = None
                            script_businessoverview = None
                            script_competitors = None
                            script_stakeholders = None
                        
                    

    st.markdown("---")
    st.markdown("Created by NextQAI")
                    
if __name__ == "__main__":
    main()
