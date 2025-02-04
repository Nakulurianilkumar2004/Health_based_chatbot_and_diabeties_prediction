from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os

# Flask app initialization
app = Flask(__name__)

# Set up the GROQ API key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key
client = Groq()

# Load chatbot dataset and embeddings
chatbot_dataset_path = "medicalcheck_dataset_health_chatbot-dataset.csv"
chatbot_embeddings_path = "medicalquestion_embedding.npy"
chatbot_df = pd.read_csv(chatbot_dataset_path)
chatbot_embeddings = np.load(chatbot_embeddings_path)

# Load chatbot model
chatbot_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load diabetes prediction model and scaler
diabetes_model = pickle.load(open('model.pkl', 'rb'))
diabetes_dataset = pd.read_csv('diabetes.csv')
scaler = MinMaxScaler(feature_range=(0, 1))
diabetes_X = diabetes_dataset.iloc[:, [1, 2, 5, 7]].values
scaler.fit(diabetes_X)


# Function to retrieve top N relevant answers for chatbot
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


@app.route('/diabetes')
def diabetes_page():
    return render_template('diabetes.html')


@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')


@app.route('/query', methods=['POST'])
def chatbot_query():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    if user_query.lower() in ["hi", "hello", "hai", "hii"]:
        return jsonify({'response': "Hello! Welcome to the Healthcare Assistant Chatbot. How can I assist you today?"})

    # Retrieve top answers and refine response
    top_answers = get_top_n_relevant_answers(user_query, chatbot_df, chatbot_embeddings, chatbot_model)
    refined_answer = get_refined_answer_from_groq(user_query, top_answers)

    return jsonify({'response': refined_answer})


@app.route('/predict', methods=['POST'])
def predict_diabetes():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    scaled_features = scaler.transform(final_features)
    prediction = diabetes_model.predict(scaled_features)

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('diabetes.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)

