from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Ensure the 'uploads' directory exists
os.makedirs('./uploads', exist_ok=True)

# OpenRouter API Key (Replace with your key)
api_key = os.getenv("OPENROUTER_API_KEY")
SECRET_KEY = os.environ.get("PTE")
MODEL_NAME = "deepseek/deepseek-r1"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Global variable to store extracted text
extracted_text = None

# Function to process PDFs and extract text
def process_pdf(pdf_files):
    global extracted_text
    extracted_text = []

    try:
        for pdf_file in pdf_files:
            file_path = os.path.join('./uploads', pdf_file.filename)
            pdf_file.save(file_path)

            print(f"Processing PDF: {file_path}")

            loader = PyMuPDFLoader(file_path)
            data = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)

            for chunk in chunks:
                extracted_text.append(chunk.page_content)

        extracted_text = " ".join(extracted_text)  # Combine text from all chunks

        print(f"Final Extracted Text: {extracted_text}")  # Debugging

        return True
    except Exception as e:
        print(f"Error during PDF processing: {str(e)}")
        return False

# Function to call OpenRouter API and generate an answer
def get_deepseek_response(question, context):
    headers = {
        "Authorization": f"Bearer {SECRET_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
        ],
        "temperature": 0.5,
        "max_tokens": 1000
    }

    print(f"Payload to OpenRouter: {payload}")  # Debugging

    response = requests.post(API_URL, json=payload, headers=headers)

    print(f"Response from OpenRouter: {response.json()}")  # Debugging

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.json()}"

# Route to serve UI
@app.route('/')
def index():
    return render_template('index.html')

# Combined API route to upload PDF and ask a question in one request
@app.route('/upload_and_ask', methods=['POST'])
def upload_and_ask():
    global extracted_text
    try:
        # Check if PDF file is uploaded
        if 'pdf_files' not in request.files or request.files.getlist('pdf_files') == []:
            return jsonify({"error": "No PDF files uploaded. Please upload a file and try again."}), 400

        pdf_files = request.files.getlist('pdf_files')
        success = process_pdf(pdf_files)  # Extract text from PDF

        if not success:
            return jsonify({"error": "Failed to extract text from PDFs."}), 400

        # Get question from request
        question = request.form.get('question', '').strip()
        if not question:
            return jsonify({"error": "Please provide a question."}), 400

        # Ask AI with extracted text as context
        answer = get_deepseek_response(question, extracted_text)

        return jsonify({
            "message": "PDF uploaded and processed successfully.",
            "answer": answer
        })

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment, default to 5000
    app.run(host="0.0.0.0", port=port)
