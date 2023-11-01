from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from cachetools import TTLCache
import mysql.connector

app = Flask(__name__)
CORS(app)

# กำหนดค่าการเชื่อมต่อ MySQL
config = {
    'user': 'root',
    'password': 'root',
    'port': 9906,
    'host': 'localhost',  # หรือที่อยู่ของฐานข้อมูล MySQL
    'database': 'ebook',
    
}


# เชื่อมต่อกับฐานข้อมูล
connection = mysql.connector.connect(**config)
cursor = connection.cursor()

# Load and preprocess data
rating_df = pd.read_csv('/app/users-score-2023.csv')
pt_cache = TTLCache(maxsize=100, ttl=300)

# ดึงข้อมูลจากตาราง book
book_query = "SELECT * FROM book"
cursor.execute(book_query)
book_data = cursor.fetchall()
book_columns = [column[0] for column in cursor.description]
book_df = pd.DataFrame(book_data, columns=book_columns)

rating_df = rating_df.merge(book_df, left_on='Anime Title', right_on='book_name', how="left")

def preprocess_data():
    num_rating_df = rating_df.groupby('book_name').count()['rating'].reset_index()
    num_rating_df.rename(columns={'rating': 'num_ratings'}, inplace=True)

    avg_rating_df = rating_df.groupby('book_name')['rating'].mean().reset_index()
    avg_rating_df.rename(columns={'rating': 'avg_rating'}, inplace=True)

    popular_df = num_rating_df.merge(avg_rating_df, on='book_name')

    popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)

    x = rating_df.groupby('user_id').count()['rating'] > 200
    padhe_likhe_users = x[x].index

    filtered_rating = rating_df[rating_df['user_id'].isin(padhe_likhe_users)]

    y = filtered_rating.groupby('book_name').count()['rating'] >= 50
    famous_books = y[y].index

    final_ratings = filtered_rating[filtered_rating['book_name'].isin(famous_books)]

    pt = final_ratings.pivot_table(index='book_name', columns='user_id', values='rating')
    pt.fillna(0, inplace=True)

    return pt

pt = preprocess_data()
similarity_scores = cosine_similarity(pt)

# Recommendation function
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for i in similar_items:
        item = {}
        temp_df = rating_df[rating_df['book_name'] == pt.index[i[0]]]
        item['book_name'] = temp_df['book_name'].values[0]
        item['book_id'] = temp_df['book_id'].values[0]
        item['book_cover'] = temp_df['book_cover'].values[0]
        item['book_price'] = int(temp_df['book_price'].values[0])
        data.append(item)

    return data

# API endpoint
@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    if not request.is_json:
        return jsonify({'error': 'Invalid JSON'}), 400

    data = request.get_json()
    if 'book_names' not in data:
        return jsonify({'error': 'Missing parameter book_names'}), 400

    book_names = data['book_names']
    recommendations = [recommend(book) for book in book_names]

    response_data = {"recommendations": recommendations}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

