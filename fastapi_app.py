
import os
import threading
import subprocess
import requests
import json
import logging
import time
import faiss
import uvicorn
import numpy as np
from PyPDF2 import PdfReader
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEndpoint
from langchain.llms import HuggingFaceHub
from transformers import pipeline
from pymongo import MongoClient
from huggingface_hub import InferenceApi

# Configure logging
logging.basicConfig(level=logging.INFO)

# FastAPI app initialization
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your actual Vercel app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB client
mongo_client = MongoClient("mongodb+srv://anisha22320184:iUZu6XHwm8Zss7lX@cluster0.de929.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your MongoDB URI
db = mongo_client["legal_cases_db"]
cases_collection = db["madras_hc"]


# Initialize the sentence transformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')


# Vector store class
class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.chunks = []

    def load_data_from_mongodb(self):
        """
        Fetch embeddings and chunks from MongoDB and add them to the FAISS index.
        """
        cases = cases_collection.find({})
        embeddings_list = []
        self.chunks = []
        for case in cases:
            case_id = case["case_id"]
            chunks = case["chunks"]
            embeddings = case["embeddings"]
            # Associate each chunk with its case_id
            for chunk, embedding in zip(chunks, embeddings):
                self.chunks.append({'chunk': chunk, 'case_id': case_id})
                embeddings_list.append(embedding)
        # Convert embeddings list to numpy array
        if embeddings_list:
            embeddings = np.array(embeddings_list)
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
        else:
            logging.warning("No embeddings found in MongoDB")

    def search(self, query, k=5):
        """
        Perform a similarity search using FAISS based on the query.
        """
        query_vector = self.embedder.encode([query])
        distances, indices = self.index.search(query_vector, k)
        return [self.chunks[i] for i in indices[0]] if indices.size > 0 else []

# Define the request body model for FastAPI
class QueryRequest(BaseModel):
    question: str

# Initialize the vector store
vector_store = VectorStore(embedder)
vector_store.load_data_from_mongodb()

huggingface_api_token = os.getenv("HF_TOKEN")
# Load the Hugging Face model for Mistral-7B text generation
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1"
)

# Prompt template for legal QA
qa_template = """
You are an assistant providing precise legal analysis and case summaries for Indian commercial court judges.
Using the provided context of similar cases, generate a predictive analysis for the user's query.

Context:
{context}

User Query: {user_query}

Predictive Analysis:
"""

def generate_prompt(context, user_query):
    return qa_template.format(context=context, user_query=user_query)


# Function to log detailed context information for debugging
def log_context_info(context, query):
    logging.info(f"Query: {query}")
    logging.info(f"Context: {context}")


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API endpoint 7'}

@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        user_query = query.question  # User's query

        # Step 1: Perform the vector search to get similar chunks
        similar_chunks = vector_store.search(user_query, k=4)

        # Step 2: Get unique case_ids from similar chunks
        case_ids = list(set(chunk['case_id'] for chunk in similar_chunks))

        # Step 3: Fetch case details from MongoDB
        cases_cursor = cases_collection.find({"case_id": {"$in": case_ids}})
        cases = list(cases_cursor)

        # Step 4: Prepare context for LLM
        context_pieces = []
        for case in cases:
            # Prepare a summary for each case
            case_summary = f"Case Name: {case.get('case_name', 'N/A')}\n" \
                           f"Date: {case.get('date', 'N/A')}\n" \
                           f"Decision: {case.get('decision', 'N/A')}\n" \
                           f"Summary: {case.get('case_summary', '')}\n"
            context_pieces.append(case_summary)
        context = "\n".join(context_pieces)

        # Step 5: Generate prompt
        prompt = generate_prompt(context=context, user_query=user_query)

        # Log the prompt
        logging.info(f"Generated prompt: {prompt}")

        # Step 6: Call the LLM
        response = llm.invoke(input=prompt, max_length=1000)

        # Process the response
        if isinstance(response, str):
            predictive_analysis = response.strip()
        elif isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]:
            predictive_analysis = response[0]["generated_text"].strip()
        else:
            logging.error("Unexpected response format from LLM")
            raise HTTPException(status_code=500, detail="Error in processing the LLM response.")

        # Step 7: Prepare the similar cases data to return
        similar_cases_data = []
        for case in cases:
            case_data = {
                "case_id": case.get("case_id"),
                "case_name": case.get("case_name"),
                "court": case.get("court"),
                "date": case.get("date"),
                "case_status": case.get("case_status"),
                "judges_involved": case.get("judges_involved"),
                "sections_clauses": case.get("sections_clauses"),
                "facts": case.get("facts"),
                "petition_filed": case.get("petition_filed"),
                "legal_issues": case.get("legal_issues"),
                "key_legal_questions": case.get("key_legal_questions"),
                "plaintiff_arguments": case.get("plaintiff_arguments"),
                "defendant_arguments": case.get("defendant_arguments"),
                "court_reasoning": case.get("court_reasoning"),
                "decision": case.get("decision"),
                "conclusion": case.get("conclusion"),
                "case_summary": case.get("case_summary"),
            }
            similar_cases_data.append(case_data)

        # Step 8: Return the response
        return {
            "Data": {
                "Predictive_analysis": predictive_analysis,
                "Similar_cases": similar_cases_data
            }
        }
    except Exception as e:
        logging.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in processing the QA request.")

# Endpoint to get related cases based on metadata
@app.get("/related_cases")
async def get_related_cases(case_id: str, visualization_type: str = "court"):
    # Find the current case
    current_case = cases_collection.find_one({"case_id": case_id})
    if not current_case:
        return {"error": "Case not found"}

    # Find related cases based on selected visualization type (court, section, judge)
    related_cases = []
    if visualization_type == "court":
        related_cases = list(cases_collection.find({"court": current_case["court"], "case_id": {"$ne": case_id}}))
    elif visualization_type == "section":
        related_cases = list(cases_collection.find({"sections_clauses": current_case["sections_clauses"], "case_id": {"$ne": case_id}}))
    elif visualization_type == "judge":
        related_cases = list(cases_collection.find({"judges_involved": current_case["judges_involved"], "case_id": {"$ne": case_id}}))

    # Collect the related case_ids and information
    result = []
    for case in related_cases:
        result.append({
            "case_id": case["case_id"],
            "case_name": case.get("case_name", "N/A"),
            "court": case.get("court", "N/A"),
            "judges_involved": case.get("judges_involved", "N/A"),
            "sections_clauses": case.get("sections_clauses", "N/A")
        })

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
