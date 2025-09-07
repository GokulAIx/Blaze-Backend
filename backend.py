from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever 
import uuid
import os
from dotenv import load_dotenv
import time 


from flask_cors import CORS

app = Flask(__name__)
CORS(app)  


load_dotenv()
app = Flask(__name__)



retriever_cache = {}
llm = None
embed_model = None


try:
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY not found in .env file!")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

except Exception as e:
    print(f"Error initializing the model: {e}")
    llm = None


@app.route("/summarize", methods=["POST"])
def summarize_text():
    if llm is None:
        return jsonify({"error": "Model could not be initialized. Check API key."}), 500
        
    try:
        data = request.json
        text = data.get("text", "")
        print(text)
        
        if not text:
            return jsonify({"error": "No text provided."}), 400

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)

        summaries = []
        for chunk in chunks:
            prompt = f"Please provide a concise summary of the following text:\n\n{chunk}"
            summary_result = llm.invoke(prompt)
            summaries.append(summary_result.content)
            time.sleep(4) 
        
        combined_summaries = " ".join(summaries)
        
        final_prompt = f"Please create one final, cohesive summary from the following collection of summaries:\n\n{combined_summaries}"
        final_summary_result = llm.invoke(final_prompt)
        
        return jsonify({"summary": final_summary_result.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/index", methods=["POST"])
def index_page():
    if llm is None or embed_model is None:
        return jsonify({"error": "Models not initialized."}), 500
    
    data = request.json
    page_content = data.get("page_content", "")
    if not page_content:
        return jsonify({"error": "No page content provided."}), 400

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(page_content)
        vector_store = Chroma.from_texts(texts=chunks, embedding=embed_model)
        retriever = vector_store.as_retriever()

        session_id = str(uuid.uuid4())
        retriever_cache[session_id] = retriever
        
        print(f"Page indexed successfully. Session ID: {session_id}")
        return jsonify({"session_id": session_id})
    except Exception as e:
        print(f"Error during indexing: {e}")
        return jsonify({"error": "Failed to index the page."}), 500

@app.route("/chat", methods=["POST"])
def chat_with_page():
    if llm is None:
        return jsonify({"error": "LLM not initialized."}), 500
    
    data = request.json
    session_id = data.get("session_id")
    question = data.get("question", "")

    if not session_id or not question:
        return jsonify({"error": "Session ID and question are required."}), 400
    
    retriever = retriever_cache.get(session_id)
    if retriever is None:
        return jsonify({"error": "Invalid session. Please analyze the page again."}), 404
    
    try:
        retrieved_docs = retriever.get_relevant_documents(question)
        
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        prompt = f"""Answer the question based only on the following context:

{context}

Question: {question}"""
        
        response = llm.invoke(prompt)
        answer = response.content
        
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": "An error occurred during chat processing."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)