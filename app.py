from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('cleaned_jobs.csv')

# Fill missing values to avoid errors
df['title'] = df['title'].fillna('')
df['link'] = df['link'].fillna('#')
df['salary'] = df['salary'].fillna(0.0)
df['country'] = df['country'].fillna('Unknown')

# Vectorize job titles
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['title'])

# Recommendation function
def recommend_jobs(user_query, top_n=5):
    if not user_query or not isinstance(user_query, str):
        return pd.DataFrame(columns=['title', 'link', 'salary', 'country'])
    
    query_vec = vectorizer.transform([user_query.lower()])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    
    recommendations = df.iloc[top_indices][['title', 'link', 'salary', 'country']].copy()
    recommendations['similarity_score'] = sim_scores[top_indices]
    return recommendations

# API endpoint for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query')
        top_n = data.get('top_n', 5)
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        if not isinstance(top_n, int) or top_n <= 0:
            return jsonify({'error': 'top_n must be a positive integer'}), 400
        
        recs = recommend_jobs(query, min(top_n, len(df)))
        if recs.empty:
            return jsonify({'message': 'No recommendations found for the given query'}), 200
        
        response = recs.to_dict(orient='records')
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# UI Route
@app.route('/')
def home():
    html_template = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <title>Job Recommendation System</title>
        <style>
            body {{
                margin: 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f6f9;
                display: flex;
                height: 100vh;
            }}
            .sidebar {{
                width: 250px;
                background-color: #007BFF;
                color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 30px 20px;
            }}
            .sidebar h2 {{
                margin-bottom: 40px;
            }}
            .container {{
                flex: 1;
                padding: 40px;
                overflow-y: auto;
            }}
            h1 {{
                color: #333;
                margin-bottom: 20px;
            }}
            .form-section {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .form-group {{
                margin-bottom: 15px;
            }}
            input[type='text'], input[type='number'] {{
                padding: 10px;
                width: 80%;
                border: 1px solid #ccc;
                border-radius: 6px;
                margin-bottom: 10px;
            }}
            input[type='submit'] {{
                padding: 10px 20px;
                background-color: #007BFF;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }}
            input[type='submit']:hover {{
                background-color: #0056b3;
            }}
            #result {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
            }}
            .job-card {{
                background: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s ease-in-out;
            }}
            .job-card:hover {{
                transform: translateY(-5px);
            }}
            .job-card h3 {{
                margin: 0;
                color: #007BFF;
                font-size: 18px;
            }}
            .job-card p {{
                margin: 5px 0;
                color: #555;
                font-size: 14px;
            }}
            .footer {{
                margin-top: 20px;
                font-size: 0.9em;
                color: #666;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class='sidebar'>
            <h2>Job Recommender</h2>
            <p>Find jobs that match your skills</p>
        </div>
        <div class='container'>
            <h1>Job Recommendation System</h1>
            <div class='form-section'>
                <form id='recommendForm'>
                    <div class='form-group'>
                        <input type='text' id='query' name='query' placeholder='Enter skills (e.g., python developer)' required>
                    </div>
                    <div class='form-group'>
                        <input type='number' id='top_n' name='top_n' placeholder='Number of recommendations (e.g., 5)' min='1' value='5'>
                    </div>
                    <input type='submit' value='Recommend'>
                </form>
            </div>
            <div id='result'></div>
            <div class='footer'>Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <script>
            document.getElementById('recommendForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const query = document.getElementById('query').value;
                const top_n = parseInt(document.getElementById('top_n').value);
                const response = await fetch('/recommend', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{query: query, top_n: top_n}})
                }});
                const result = await response.json();
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '';
                if (response.ok) {{
                    if (result.message) {{
                        resultDiv.innerHTML = `<p>${{result.message}}</p>`;
                    }} else {{
                        result.forEach(job => {{
                            const card = `
                                <div class='job-card'>
                                    <h3>${{job.title || 'N/A'}}</h3>
                                    <p><strong>Country:</strong> ${{job.country || 'Unknown'}}</p>
                                    <p><strong>Salary:</strong> $${{(job.salary || 0).toFixed(2)}}</p>
                                    <p><strong>Similarity:</strong> ${{job.similarity_score.toFixed(2)}}</p>
                                    <a href='${{job.link || '#'}}' target='_blank'>View Job</a>
                                </div>`;
                            resultDiv.innerHTML += card;
                        }});
                    }}
                }} else {{
                    resultDiv.innerHTML = `<p style="color:red;">Error: ${{result.error || 'Unknown error'}}</p>`;
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_template

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
