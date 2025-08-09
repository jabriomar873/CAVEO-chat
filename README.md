# 💬 PDF Chat - AI-Powered Document Q&A System

A powerful, torch-free PDF chat application that allows you to upload PDF documents and ask questions about their content using local AI models. The application features OCR support for scanned documents and uses TF-IDF embeddings for efficient document retrieval.

## ✨ Features

- **📄 PDF Processing**: Upload and process multiple PDF files simultaneously
- **🔍 OCR Support**: Automatic text extraction from scanned PDFs using Tesseract OCR
- **🤖 Local AI**: Powered by Ollama for privacy-focused, offline AI conversations
- **🧠 Smart Retrieval**: TF-IDF-based document embeddings (100% torch-free)
- **💬 Interactive Chat**: Streamlit-based web interface with conversation history
- **⚡ Optimized Performance**: Minimal dependencies and efficient processing
- **🔒 Privacy First**: Everything runs locally - no data sent to external servers

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** AI platform installed
3. **Tesseract OCR** (for scanned PDF support)

### Installation

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   cd chatchot-main
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama** (if not already installed)
   - Download from: https://ollama.ai
   - Follow the installation instructions for your OS

4. **Install an AI Model**
   ```bash
   ollama pull llama3.2:1b
   ```
   *Note: You can use other models like `llama3.2:3b`, `mistral`, etc.*

5. **Install Tesseract OCR** (for scanned PDF support)
   
   **Windows:**
   ```bash
   winget install UB-Mannheim.TesseractOCR
   ```
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Linux:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

### Running the Application

**Option 1: Using the Batch File (Windows)**
```bash
start_app.bat
```

**Option 2: Using Streamlit Directly**
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload Documents**
   - Click "Choose PDF files" to select your PDF documents
   - Multiple files can be uploaded at once
   - Both text-based and scanned PDFs are supported

2. **Process Documents**
   - Click "🔄 Process Documents" to analyze your PDFs
   - The app will extract text, create chunks, and build a knowledge base
   - Wait for the "✅ Ready to chat!" confirmation

3. **Ask Questions**
   - Type your questions in the chat input field
   - Ask anything about the content in your uploaded documents
   - The AI will provide answers based on the document content

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │ ─→ │  Text Extraction │ ─→ │  Text Chunking  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                          ┌────▼────┐
                          │   OCR   │ (for scanned PDFs)
                          └─────────┘
                               │
┌─────────────────┐    ┌───────▼──────────┐    ┌─────────────────┐
│   User Query    │ ─→ │  TF-IDF Vector   │ ─→ │  FAISS Vector   │
└─────────────────┘    │    Embeddings    │    │     Store       │
                       └──────────────────┘    └─────────────────┘
                               │                        │
┌─────────────────┐    ┌───────▼──────────┐           │
│  AI Response    │ ←─ │     Ollama       │ ←─────────┘
└─────────────────┘    │   Local LLM      │
                       └──────────────────┘
```

## 📁 Project Structure

```
chatchot-main/
├── app.py                 # Main application file
├── htmlTemplates.py       # UI templates for chat interface
├── requirements.txt       # Python dependencies
├── start_app.bat         # Windows launcher script
├── README.md             # This file
└── __pycache__/          # Python cache files
```

## 🔧 Configuration

### Supported AI Models

The application works with any Ollama-compatible model:
- `llama3.2:1b` (recommended for faster performance)
- `llama3.2:3b` (better quality, slower)
- `mistral`
- `codellama`
- And many more available through Ollama

### OCR Configuration

Tesseract OCR is automatically detected in these locations (Windows):
- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
- System PATH

### Environment Variables

The application sets these environment variables for optimal performance:
```bash
TORCH_LOGS=0
PYTHONWARNINGS=ignore::UserWarning:torch,ignore::DeprecationWarning:langchain
PYTORCH_DISABLE_PER_OP_PROFILING=1
```

## 🛠️ Troubleshooting

### Common Issues

**1. "Ollama not detected"**
- Ensure Ollama is installed and running
- Try running `ollama list` in terminal to verify installation

**2. "No Ollama models found"**
- Install a model: `ollama pull llama3.2:1b`
- Restart the application

**3. "Tesseract OCR is not installed" (for scanned PDFs)**
- Install Tesseract OCR using the instructions above
- Restart the application

**4. "No text found in PDF"**
- The PDF might be password-protected
- Try a different PDF file
- For scanned PDFs, ensure Tesseract is installed

### Performance Tips

- Use smaller models like `llama3.2:1b` for faster responses
- Process fewer documents at once for better performance
- Close other applications to free up system resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 Dependencies

### Core Dependencies
- **Streamlit**: Web interface framework
- **LangChain**: AI/LLM integration framework
- **Ollama**: Local AI model runtime
- **FAISS**: Vector similarity search
- **scikit-learn**: TF-IDF embeddings

### PDF Processing
- **PyPDF**: PDF text extraction
- **PyMuPDF**: Advanced PDF processing
- **Pytesseract**: OCR for scanned documents
- **Pillow**: Image processing

### Utilities
- **python-dotenv**: Environment variable management
- **numpy**: Numerical computations

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space (more for AI models)
- **Python**: 3.8 or higher

### Recommended Requirements
- **RAM**: 8GB or more
- **CPU**: Multi-core processor
- **Storage**: 5GB+ for multiple AI models

## 🔐 Privacy & Security

- **100% Local Processing**: No data is sent to external servers
- **Offline Capable**: Works completely offline once set up
- **No Data Storage**: Documents are processed in memory only
- **Open Source**: Full transparency of code and functionality

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify Ollama and Tesseract installations
4. Check the terminal/console for error messages

## 📜 License

This project is open source. Please check the license file for details.

---

**Made with ❤️ for document analysis and AI-powered Q&A**