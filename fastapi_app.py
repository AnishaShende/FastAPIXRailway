
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

# Paths to precomputed data
# embeddings_path = "data/embeddings.npy"
# chunks_path = "data/chunks.json"

embeddings_path = "https://raw.githubusercontent.com/AnishaShende/FastAPIXRailway/main/Data/chunks.json"
chunks_path = "https://github.com/AnishaShende/FastAPIXRailway/raw/main/Data/embeddings.npy"

# Initialize the sentence transformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load precomputed embeddings and chunks
def load_precomputed_data(embeddings_path, chunks_path):
    embeddings = np.load(embeddings_path)
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)
    return embeddings, chunks

embeddings, chunks = load_precomputed_data(embeddings_path, chunks_path)

# Vector store class
class VectorStore:
    def __init__(self, embeddings, chunks, embedder):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.chunks = chunks
        self.embedder = embedder

    def search(self, query, k=5):
        query_vector = self.embedder.encode([query])
        distances, indices = self.index.search(query_vector, k)
        return [self.chunks[i] for i in indices[0]]

# Define the request body model for FastAPI
class QueryRequest(BaseModel):
    question: str

# Initialize the vector store
vector_store = VectorStore(embeddings, chunks, embedder)
huggingface_api_token = os.getenv("HF_TOKEN")
# Load the Hugging Face model for Mistral-7B text generation
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1"
)

# QA system template
qa_template = r"""
<SYS>
You are an assistant providing precise legal analysis and case summaries for Indian commercial court judges. Generate content that directly fills the required fields without any extra formatting or headings. Ensure that each response is concise, clear, and strictly adheres to the specific field it is meant for.
</SYS>

Context:
{context}

1. Predictive Analysis:
{{predictive_summary}} [Generate a concise summary of the predictive analysis relevant to the case, without any extra text.]

2. Similar Cases:
Case Name: {{case_name}} [Provide only the case name.]
Date: {{case_date}} [Provide only the date of the case.]
Decision: {{case_decision}} [Provide a summary of the decision.]

Case Details:
Case No: {{case_number}} [Provide only the case number.]
Court: {{case_court}} [Provide only the court name.]
Case Status: {{case_status}} [Provide only the case status.]
Judge: {{case_judge}} [Provide only the judge's name.]
Section: {{case_section}} [Provide only the relevant section.]
Facts: {{case_facts}} [Provide a brief summary of the case facts.]
Legal Issues: {{case_legal_issues}} [List the key legal issues.]
Key Legal Questions: {{case_key_questions}} [List the key legal questions.]
Plaintiff Arguments: {{plaintiff_arguments}} [Summarize the plaintiff's arguments.]
Defendant Arguments: {{defendant_arguments}} [Summarize the defendant's arguments.]
Court's Reasoning: {{court_reasoning}} [Summarize the court's reasoning.]
Decision: {{court_decision}} [Provide the final decision.]
Conclusion: {{court_conclusion}} [Summarize the conclusion of the case.]
Case Summary: {{case_summary}} [Provide a concise summary of the case.]

User Query: {user_query}
You:
"""

def generate_prompt(context, user_query):
    """
    Generate a prompt for the model using the user's query and context from similar cases.

    Parameters:
    - context (str): A string containing the context information about similar cases.
    - user_query (str): The user's query, asking for predictive analysis and advice on the case.

    Returns:
    - str: A prompt formatted for use with the LLM to generate predictive analysis and fetch similar cases.
    """
    # Structure the predictive analysis section of the prompt
    predictive_analysis_section = f"""
    Based on the user's query: '{user_query}', provide a predictive analysis of this case.
    Consider similar cases and the legal context provided.
    """

    # Structure the similar cases section of the prompt
    similar_cases_section = f"""
    Consider the following context from previous similar cases:
    {context}

    Predictive Analysis:
    - Predict the possible outcome of this case based on the context and legal precedents.

    Similar Cases:
    - Provide a list of at least 3 similar cases with the following details:
      - Case Name
      - Date
      - Court Decision
      - Key Legal Issues
      - Brief Summary of Facts
    """

    # Combine the sections into the final prompt
    final_prompt = predictive_analysis_section + similar_cases_section

    return final_prompt

# Function to log detailed context information for debugging
def log_context_info(context, query):
    logging.info(f"Query: {query}")
    logging.info(f"Context: {context}")

class ContextAwareQA:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def validate_json(self, raw_output):
        try:
            # Attempt to load the response as JSON
            output_json = json.loads(raw_output)
            return output_json
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse output as JSON: {str(e)}")
            raise HTTPException(status_code=500, detail="Invalid JSON format from model output.")

    def clean_response(raw_output: str) -> str:
        """
        Clean up the model output to ensure it's valid JSON.
        """
        cleaned_output = raw_output.strip()
        return cleaned_output

    def extract_field(self, raw_output, start_token, end_token=None):
        try:
            start_index = raw_output.lower().find(start_token.lower())  # Case insensitive search
            if start_index == -1:
                logging.warning(f"Start token '{start_token}' not found in the output.")
                return None
            start_index += len(start_token)
            if end_token:
                end_index = raw_output.lower().find(end_token.lower(), start_index)
                if end_index == -1:
                    logging.warning(f"End token '{end_token}' not found in the output after '{start_token}'.")
                    return raw_output[start_index:].strip()
                return raw_output[start_index:end_index].strip()
            return raw_output[start_index:].strip()
        except Exception as e:
            logging.error(f"Error extracting field with start_token '{start_token}': {str(e)}")
            return None

    def process_output(self, raw_output):
        json_output = {
            "Data": {
                "Predictive_analysis": "",
                "Similar_cases": []
            }
        }

        # Extracting the Predictive Analysis
        json_output["Data"]["Predictive_analysis"] = self.extract_field(raw_output, "1. Predictive Analysis:", "2. Similar Cases:")

        # Extracting the Similar Cases
        case_name = self.extract_field(raw_output, "Case Name:", "Date:")
        case_date = self.extract_field(raw_output, "Date:", "Decision:")
        case_decision = self.extract_field(raw_output, "Decision:", "Case No:" )

        # Extracting Case Details
        case_details = {
            "Case No": self.extract_field(raw_output, "Case No:", "Court:"),
            "Court": self.extract_field(raw_output, "Court:", "Case Status:"),
            "Case Status": self.extract_field(raw_output, "Case Status:", "Judge:"),
            "Judge": self.extract_field(raw_output, "Judge:", "Section:"),
            "Section": self.extract_field(raw_output, "Section:", "Facts:"),
            "Facts": self.extract_field(raw_output, "Facts:", "Legal Issues:"),
            "Legal Issues": self.extract_field(raw_output, "Legal Issues:", "Key Legal Questions:"),
            "Key Legal Questions": self.extract_field(raw_output, "Key Legal Questions:", "Plaintiff Arguments:"),
            "Plaintiff Arguments": self.extract_field(raw_output, "Plaintiff Arguments:", "Defendant Arguments:"),
            "Defendant Arguments": self.extract_field(raw_output, "Defendant Arguments:", "Court's Reasoning:"),
            "Court's Reasoning": self.extract_field(raw_output, "Court's Reasoning:", "Decision:"),
            "Decision": self.extract_field(raw_output, "Decision:", "Conclusion:"),
            "Conclusion": self.extract_field(raw_output, "Conclusion:", "Case Summary:"),
            "Case Summary": self.extract_field(raw_output, "Case Summary:")
        }

        # Creating a case entry
        case_entry = {
            "Case_name": case_name,
            "Date": case_date,
            "Decision": case_decision,
            "case_details": case_details
        }

        # Adding the case entry to the list of similar cases
        json_output["Data"]["Similar_cases"].append(case_entry)

        return json_output

    def call(self, input: dict) -> dict:
        try:
            query = input['input_str']  # Fetch query from input dictionary
            context = "\n".join(self.vector_store.search(query))

            # Fill in the template with the context and query
            prompt = qa_template.format(context=context, user_query=query)

            # Log the prompt for debugging purposes
            log_context_info(context, query)

        #     logging.info(f"Prompt for the generator: {prompt}")

        #     # Use the Hugging Face Inference API for text generation
        #     response = llm.invoke(input=prompt, max_length=1000)

        #     # Log the response for debugging
        #     logging.info(f"Response from Hugging Face API: {response}")

        #     # Ensure response is in JSON format
        #     if isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]:
        #         raw_output = response[0]["generated_text"]
        #         # cleaned_output = ContextAwareQA.clean_response(raw_output)
        #         logging.info(f"Raw generated text: {raw_output}")

        #         # Process and validate the output as JSON
        #         processed_output = self.process_output(raw_output)
        #         return processed_output
        #     else:
        #         logging.error("Unexpected response format from Hugging Face API.")
        #         raise HTTPException(status_code=500, detail="Unexpected response format from Hugging Face API.")

        # except Exception as e:
        #     logging.error(f"Error in ContextAwareQA: {str(e)}")
        #     raise HTTPException(status_code=500, detail=f"Error in QA processing: {str(e)}")
            logging.info(f"Prompt for the generator: {prompt}")

            response = llm.invoke(input=prompt, max_length=1000)

            logging.info(f"Response from Hugging Face API: {response}")

            if isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]:
                raw_output = response[0]["generated_text"]
            else:
                raw_output = response

            cleaned_output = self.clean_response(raw_output)

            output_json = self.validate_json(cleaned_output)
            return output_json

        except Exception as e:
          logging.error(f"Error occurred while processing the request: {str(e)}")
          raise HTTPException(status_code=500, detail="Internal server error")


# Initialize your QA system
qa_pipeline = ContextAwareQA(vector_store)

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API endpoint 7'}

@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        # Your query
        user_query = query.question

        # Search context (dummy function, replace with actual implementation)
        context = context = "\n".join(vector_store.search(user_query))  # Replace with your vector store search

        # Generate the dynamic prompt
        prompt = generate_prompt(context=context, user_query=user_query)

        # Pass the prompt to Mistral via LangChain
        raw_output = llm(prompt)

        # Log the raw response for debugging
        logging.info(f"Raw response: {raw_output}")

        # Extract and process output (like in your earlier Llama code)
        processed_output = qa_pipeline.process_output(raw_output)  # Define process_output similar to your previous code

        # Return the processed response
        return {
            "Data": {
                "query": user_query,
                "predictive_analysis": processed_output.get("Data", {}).get("Predictive_analysis"),
                "similar_cases": processed_output.get("Data", {}).get("Similar_cases")
            }
        }

    except Exception as e:
        logging.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in processing the QA request.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
