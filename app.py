# -*- coding: utf-8 -*-
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import shutil

# Suppress Streamlit torch warnings
import os
import sys

# Set environment variables before any torch-related imports
os.environ["PYTORCH_DISABLE_PER_OP_PROFILING"] = "1"
os.environ["TORCH_LOGS"] = ""
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from enhanced_retrieval import enhance_query, rerank_retrieved_docs, create_enhanced_context, validate_response_completeness
from config import RETRIEVAL_CONFIG, TEXT_CONFIG, TFIDF_CONFIG, LLM_CONFIG, PROMPT_TEMPLATES
from intent_detection import detect_intent, generate_greeting_response, generate_simple_chat_response, should_use_enhanced_retrieval
import warnings
import subprocess
import re

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*torch.*")

# Redirect torch warnings to suppress them completely
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents with OCR support for scanned PDFs"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            pdf_text = ""
            
            # First try normal text extraction
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text
            
            # If no text found, try OCR with PyMuPDF
            if not pdf_text.strip():
                try:
                    import fitz  # PyMuPDF
                    import pytesseract
                    from PIL import Image
                    import io
                    
                    # Check if Tesseract is available
                    try:
                        # Set Tesseract path for Windows
                        import platform
                        if platform.system() == 'Windows':
                            possible_paths = [
                                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                                r"C:\Users\Eagle\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
                                r"C:\Users\Eagle\AppData\Local\Microsoft\WinGet\Packages\UB-Mannheim.TesseractOCR_Microsoft.Winget.Source_8wekyb3d8bbwe\tesseract.exe",
                                r"C:\tools\tesseract\tesseract.exe"
                            ]
                            tesseract_found = False
                            for path in possible_paths:
                                if os.path.exists(path):
                                    pytesseract.pytesseract.tesseract_cmd = path
                                    tesseract_found = True
                                    st.info(f"🔍 Found Tesseract at: {path}")
                                    break
                            
                            if not tesseract_found:
                                # Try to find tesseract in PATH
                                try:
                                    import shutil
                                    tesseract_path = shutil.which("tesseract")
                                    if tesseract_path:
                                        pytesseract.pytesseract.tesseract_cmd = tesseract_path
                                        tesseract_found = True
                                        st.info(f"🔍 Found Tesseract in PATH: {tesseract_path}")
                                except:
                                    pass
                        
                        # Test if Tesseract works
                        pytesseract.get_tesseract_version()
                        
                    except Exception as e:
                        st.error(f"❌ {pdf.name} is a scanned PDF but Tesseract OCR is not installed.")
                        st.info("🔧 **Install Tesseract OCR:**")
                        st.code("winget install UB-Mannheim.TesseractOCR")
                        st.info("Or download from: https://github.com/UB-Mannheim/tesseract/wiki")
                        st.info("After installation, restart the app.")
                        continue
                    
                    # Reset file pointer
                    pdf.seek(0)
                    pdf_bytes = pdf.read()
                    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                    
                    st.info(f"🔍 Using OCR for {pdf.name} (scanned document)")
                    
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        # Convert page to image
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Perform OCR
                        try:
                            ocr_text = pytesseract.image_to_string(image, lang='eng')
                            if ocr_text.strip():
                                pdf_text += ocr_text + "\n"
                        except Exception as ocr_error:
                            st.warning(f"⚠️ OCR failed for page {page_num + 1}: {str(ocr_error)}")
                    
                    pdf_document.close()
                    
                except ImportError:
                    st.error(f"❌ {pdf.name} appears to be a scanned PDF but OCR libraries are missing.")
                    st.info("Run: pip install pymupdf pytesseract pillow")
                    continue
                except Exception as e:
                    st.error(f"❌ OCR failed for {pdf.name}: {str(e)}")
                    continue
            
            if pdf_text.strip():
                text += pdf_text
                st.success(f"✅ Processed {pdf.name}")
            else:
                st.error(f"❌ No text found in {pdf.name}")
                
        except Exception as e:
            st.error(f"❌ Error reading {pdf.name}: {str(e)}")
    
    return text

def get_text_chunks(text):
    """Split text into chunks with improved overlapping for better context retention"""
    # Use configuration for optimal chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CONFIG["chunk_size"],
        chunk_overlap=TEXT_CONFIG["chunk_overlap"],
        length_function=len,
        separators=TEXT_CONFIG["separators"]
    )
    chunks = text_splitter.split_text(text)
    
    # Add metadata to chunks for better tracking
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        # Add chunk number and context markers
        enhanced_chunk = f"[CHUNK {i+1}] {chunk}"
        enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks

def get_vectorstore(text_chunks):
    """Create vector store from text chunks using torch-free TF-IDF embeddings"""
    try:
        # Use sklearn TF-IDF to completely avoid torch
        from sklearn.feature_extraction.text import TfidfVectorizer
        from langchain.embeddings.base import Embeddings
        import numpy as np
        
        class TorchFreeTFIDFEmbeddings(Embeddings):
            def __init__(self):
                self.vectorizer = TfidfVectorizer(
                    max_features=TFIDF_CONFIG["max_features"],
                    stop_words=TFIDF_CONFIG["stop_words"],
                    ngram_range=TFIDF_CONFIG["ngram_range"],
                    min_df=TFIDF_CONFIG["min_df"],
                    max_df=TFIDF_CONFIG["max_df"],
                    sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
                    analyzer='word'
                )
                self.fitted = False
                self.feature_dim = TFIDF_CONFIG["max_features"]
                
            def embed_documents(self, texts):
                if not self.fitted:
                    vectors = self.vectorizer.fit_transform(texts)
                    self.fitted = True
                else:
                    vectors = self.vectorizer.transform(texts)
                
                # Convert sparse matrix to dense and then to list
                dense_vectors = vectors.toarray()
                # Pad or truncate to fixed dimension
                result = []
                for vector in dense_vectors:
                    if len(vector) < self.feature_dim:
                        padded = np.pad(vector, (0, self.feature_dim - len(vector)), 'constant')
                    else:
                        padded = vector[:self.feature_dim]
                    result.append(padded.tolist())
                return result
                
            def embed_query(self, text):
                if not self.fitted:
                    return [0.0] * self.feature_dim
                    
                vector = self.vectorizer.transform([text])
                dense_vector = vector.toarray()[0]
                
                # Pad or truncate to fixed dimension
                if len(dense_vector) < self.feature_dim:
                    padded = np.pad(dense_vector, (0, self.feature_dim - len(dense_vector)), 'constant')
                else:
                    padded = dense_vector[:self.feature_dim]
                    
                return padded.tolist()
        
        st.info("🔧 Using Enhanced TF-IDF embeddings (100% torch-free)")
        embeddings = TorchFreeTFIDFEmbeddings()
        
        # Create vectorstore with enhanced retrieval settings
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        # Configure the retriever for better precision using config
        retriever = vectorstore.as_retriever(
            search_type=RETRIEVAL_CONFIG["search_type"],
            search_kwargs={
                "k": RETRIEVAL_CONFIG["initial_k"],
                "lambda_mult": RETRIEVAL_CONFIG["lambda_mult"],
                "fetch_k": RETRIEVAL_CONFIG["fetch_k"]
            }
        )
        
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ Error with TF-IDF embeddings: {e}")
        
        # If even TF-IDF fails, use a simple fallback
        try:
            st.warning("🔄 Using basic text matching as fallback...")
            
            class SimpleFallbackEmbeddings(Embeddings):
                def __init__(self):
                    self.feature_dim = 300
                    
                def embed_documents(self, texts):
                    # Very simple hash-based embeddings
                    embeddings = []
                    for text in texts:
                        # Create a simple embedding based on text properties
                        words = text.lower().split()
                        embedding = [0.0] * self.feature_dim
                        
                        for i, word in enumerate(words[:self.feature_dim]):
                            embedding[i % self.feature_dim] += hash(word) % 1000 / 1000.0
                            
                        embeddings.append(embedding)
                    return embeddings
                    
                def embed_query(self, text):
                    words = text.lower().split()
                    embedding = [0.0] * self.feature_dim
                    
                    for i, word in enumerate(words[:self.feature_dim]):
                        embedding[i % self.feature_dim] += hash(word) % 1000 / 1000.0
                        
                    return embedding
            
            embeddings = SimpleFallbackEmbeddings()
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            st.warning("⚠️ Using basic text matching (limited accuracy)")
            return vectorstore
            
        except Exception as e2:
            st.error(f"❌ All embedding methods failed: {e2}")
            return None

def get_conversation_chain(vectorstore):
    """Create conversation chain"""
    # Get available models
    available_models = get_available_ollama_models()
    if not available_models:
        st.error("❌ No Ollama models found. Please install a model first.")
        st.info("Run: ollama pull llama3.2:1b")
        return None
    
    # Use the first available model
    model_name = available_models[0]
    
    try:
        llm = OllamaLLM(
            model=model_name,
            temperature=LLM_CONFIG["temperature"],
            top_p=LLM_CONFIG["top_p"],
            top_k=LLM_CONFIG["top_k"]
        )
        
        # Use enhanced prompt template from config
        PROMPT = PromptTemplate(
            template=PROMPT_TEMPLATES["main_template"],
            input_variables=["context", "question"]
        )

        # Enhanced retriever configuration using config
        retriever = vectorstore.as_retriever(
            search_type=RETRIEVAL_CONFIG["search_type"],
            search_kwargs={
                "k": RETRIEVAL_CONFIG["initial_k"],
                "lambda_mult": RETRIEVAL_CONFIG["lambda_mult"],
                "fetch_k": RETRIEVAL_CONFIG["fetch_k"]
            }
        )

        # Use the updated ConversationalRetrievalChain without deprecated memory
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

        st.session_state.model_info = {'model': model_name}
        return conversation_chain
        
    except Exception as e:
        st.error(f"❌ Error creating conversation chain: {e}")
        return None

def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        return []
    except:
        return []

def check_ollama_installation():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def handle_user_input(user_question):
    """Handle user input and generate response with enhanced accuracy"""
    try:
        # Detect user intent first
        intent = detect_intent(user_question)
        
        # Handle greetings and simple chat without document retrieval
        if intent == "greeting":
            response_text = generate_greeting_response()
            
            # Update chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append(f"Human: {user_question}")
            st.session_state.chat_history.append(f"Assistant: {response_text}")
            return
            
        elif intent == "simple_chat":
            response_text = generate_simple_chat_response(user_question)
            
            # Update chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append(f"Human: {user_question}")
            st.session_state.chat_history.append(f"Assistant: {response_text}")
            return
        
        # For document analysis questions, use enhanced retrieval
        elif intent == "document_analysis":
            with st.spinner("🔍 Analyzing documents thoroughly..."):
                # Enhance the user query for better retrieval
                enhanced_question = enhance_query(user_question)
                
                # Get existing chat history from session state
                chat_history = st.session_state.get('chat_history', [])
                
                # Convert string format to tuples for LangChain compatibility
                formatted_history = []
                for i in range(0, len(chat_history), 2):
                    if i + 1 < len(chat_history):
                        human_msg = chat_history[i].replace("Human: ", "")
                        ai_msg = chat_history[i + 1].replace("Assistant: ", "")
                        formatted_history.append((human_msg, ai_msg))
                
                # Check if this is a phase counting question
                is_phase_counting = any(phrase in user_question.lower() for phrase in 
                    ["how many phases", "combien de phases", "number of phases", "phases are", "phases dans"])
                
                # Get retriever from conversation chain
                retriever = st.session_state.conversation.retriever
                
                # For phase counting, use more aggressive retrieval
                if is_phase_counting:
                    retriever.search_kwargs["k"] = 15  # Get more documents
                    retriever.search_kwargs["fetch_k"] = 25  # Consider more candidates
                
                # Retrieve documents with enhanced query
                retrieved_docs = retriever.get_relevant_documents(enhanced_question)
                
                # Re-rank documents for better relevance
                reranked_docs = rerank_retrieved_docs(retrieved_docs, user_question)
                
                # Create enhanced context
                enhanced_context = create_enhanced_context(reranked_docs, user_question)
                
                # Use specialized prompt for phase counting
                if is_phase_counting:
                    # Create a temporary conversation chain with phase counting prompt
                    from langchain.prompts import PromptTemplate
                    phase_prompt = PromptTemplate(
                        template=PROMPT_TEMPLATES["phase_counting_template"],
                        input_variables=["context", "question"]
                    )
                    
                    # Enhanced context with emphasis on accuracy
                    phase_context = f"""INSTRUCTIONS D'ANALYSE PRÉCISE:
Analysez UNIQUEMENT les phases explicitement mentionnées dans les documents ci-dessous.
N'ajoutez AUCUNE phase qui n'est pas clairement documentée.

{enhanced_context}

RAPPEL: Comptez uniquement ce qui existe réellement dans les documents."""
                    
                    # Use the conversation chain with specialized prompt and enhanced context
                    response = st.session_state.conversation.combine_docs_chain.run(
                        input_documents=reranked_docs[:8], 
                        question=f"CONSIGNE STRICTE: Basez-vous UNIQUEMENT sur les documents fournis. Ne mentionnez que les phases qui existent vraiment. Question: {user_question}"
                    )
                    response = {"answer": response, "source_documents": reranked_docs[:5]}
                else:
                    # Use the regular conversation chain
                    response = st.session_state.conversation.invoke({
                        'question': user_question,
                        'chat_history': formatted_history
                    })
                
                # Validate response completeness (this will catch missing Phase 00)
                is_complete, warning_msg = validate_response_completeness(response['answer'], user_question)
                
                # Display warning if response seems incomplete
                if not is_complete:
                    st.warning(f"⚠️ {warning_msg}")
                    
                    # Try to get more context and regenerate with specific emphasis on Phase 00
                    with st.spinner("🔄 Searching for ALL phases including Phase 00..."):
                        # Enhanced search specifically for Phase 00
                        phase_00_query = f"{enhanced_question} phase00 phase 00 étude chiffrage study estimation"
                        extended_docs = retriever.get_relevant_documents(phase_00_query)
                        extended_context = create_enhanced_context(extended_docs, user_question)
                        
                        # Regenerate with explicit Phase 00 instruction
                        enhanced_response = st.session_state.conversation.invoke({
                            'question': f"IMPORTANT: Liste TOUTES les phases y compris la Phase 00 (Étude et chiffrage). Question: {user_question}",
                            'chat_history': formatted_history
                        })
                        
                        response = enhanced_response
                
                # Reset retriever parameters to normal
                if is_phase_counting:
                    retriever.search_kwargs["k"] = RETRIEVAL_CONFIG["initial_k"]
                    retriever.search_kwargs["fetch_k"] = RETRIEVAL_CONFIG["fetch_k"]
                
                # Update chat history manually
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Add question and answer to history
                st.session_state.chat_history.append(f"Human: {user_question}")
                st.session_state.chat_history.append(f"Assistant: {response['answer']}")
                
                # Show source documents for transparency
                if 'source_documents' in response and response['source_documents']:
                    with st.expander("📚 Sources utilisées"):
                        for i, doc in enumerate(response['source_documents'][:3]):
                            st.text(f"Source {i+1}: {doc.page_content[:200]}...")
                            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        # Try fallback approach
        try:
            st.info("🔄 Trying alternative approach...")
            
            # Check if we have a conversation chain available
            if hasattr(st.session_state, 'conversation') and st.session_state.conversation:
                simple_response = st.session_state.conversation.invoke({
                    'question': user_question,
                    'chat_history': []
                })
                
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append(f"Human: {user_question}")
                st.session_state.chat_history.append(f"Assistant: {simple_response['answer']}")
            else:
                # If no conversation chain, provide a helpful message
                fallback_msg = "Je suis désolé, mais je n'ai pas encore de documents à analyser. Veuillez d'abord uploader et traiter vos documents PDF."
                
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append(f"Human: {user_question}")
                st.session_state.chat_history.append(f"Assistant: {fallback_msg}")
            
        except Exception as e2:
            st.error(f"❌ Fallback also failed: {str(e2)}")

def main():
    load_dotenv()

    st.set_page_config(
        page_title="PDF Chat",
        page_icon="💬",
        layout="wide"
    )
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Simple header
    st.title("💬 PDF Chat")
    
    # Check Ollama installation
    if not check_ollama_installation():
        st.error("❌ Ollama not detected. Please install Ollama first.")
        st.info("Download from: https://ollama.ai")
        if st.button("🔄 Check Again"):
            st.rerun()
        st.stop()
    
    # **INPUT AT TOP - ALWAYS VISIBLE**
    st.markdown("---")
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_question = st.text_input(
                "Ask a question:",
                placeholder="Type your question here...",
                label_visibility="collapsed"
            )
        with col2:
            submit_button = st.form_submit_button("💬 Send", use_container_width=True)
    
    # Handle user input
    if submit_button and user_question:
        if st.session_state.conversation:
            handle_user_input(user_question)
            st.rerun()
        else:
            st.warning("📁 Please upload and process documents first!")
    
    # **SIMPLE SIDEBAR FOR DOCUMENTS**
    with st.sidebar:
        st.header("📄 Documents")
        
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type="pdf",
            label_visibility="collapsed"
        )
        
        if st.button("🔄 Process", use_container_width=True):
            if not pdf_docs:
                st.warning("⚠️ Upload PDFs first")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Extract text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("❌ No text found")
                        else:
                            # Create chunks
                            text_chunks = get_text_chunks(raw_text)
                            
                            # Create vector store
                            vectorstore = get_vectorstore(text_chunks)
                            if not vectorstore:
                                st.error("❌ Processing failed")
                                return

                            # Create conversation chain
                            conversation_chain = get_conversation_chain(vectorstore)
                            
                            if conversation_chain:
                                st.session_state.conversation = conversation_chain
                                st.success("✅ Ready!")
                            else:
                                st.error("❌ Setup failed")
                                
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)[:50]}...")
        
        # Show status
        if st.session_state.conversation:
            st.success("✅ AI Ready")
            if hasattr(st.session_state, 'model_info'):
                model = st.session_state.model_info.get('model', 'AI')
                st.caption(f"Model: {model}")
        else:
            st.info("📁 Upload documents to start")
    
    # **CHAT AREA - MAIN CONTENT**
    st.markdown("---")
    
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        # Display conversation (most recent first)
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            message = st.session_state.chat_history[i]
            if i % 2 == 0:  # User messages
                user_msg = message.replace("Human: ", "")
                st.write(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
            else:  # Bot messages
                bot_msg = message.replace("Assistant: ", "")
                st.write(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)
    else:
        st.info("💬 Start a conversation by typing a question above")

if __name__ == '__main__':
    main()
