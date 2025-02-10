from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os

# Flask app initialization
app = Flask(__name__)

# Set up the GROQ API key
os.environ["GROQ_API_KEY"] = ""  # Replace with your actual API key
client = Groq()

# Load the dataset and embeddings
dataset_path = "medicalcheck_dataset_health_chatbot-dataset.csv"
embeddings_path = "medicalquestion_embedding.npy"
df = pd.read_csv(dataset_path)
embeddings = np.load(embeddings_path)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to retrieve top N relevant answers
def get_top_n_relevant_answers(query, dataset, embeddings, model, n=3):
    query_embedding = model.encode([query], batch_size=1)
    similarities = cosine_similarity(query_embedding, embeddings)
    top_n_idx = np.argsort(similarities[0])[::-1][:n]
    return dataset.iloc[top_n_idx]['responses'].tolist()

# Function to refine the answer using Groq API
def get_refined_answer_from_groq(query, top_answers):
    messages = [
        {
            "role": "user",
            "content": f"This is the user query: {query}\n\n"
                       f"Here are the top matched results from my knowledge base:\n"
                       + "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(top_answers)]) +
                       "\n\nFormulate a final answer most relevant to the user query."
        },
        {
            "role": "assistant",
            "content": "The final answer is:"
        }
    ]
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        top_p=1
    )
    return completion.choices[0].message.content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    if user_query.lower() in ["hi", "hello", "hai", "hii"]:
        return jsonify({'response': "Hello! Welcome to the Healthcare Assistant Chatbot. How can I assist you today?"})

    # Retrieve top answers and refine response
    top_answers = get_top_n_relevant_answers(user_query, df, embeddings, model)
    refined_answer = get_refined_answer_from_groq(user_query, top_answers)

    return jsonify({'response': refined_answer})

if __name__ == '__main__':
    app.run(debug=True)


