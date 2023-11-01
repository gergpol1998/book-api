import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, jsonify
from flask_cors import CORS 

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

# ดึงข้อมูลจากตาราง book
book_query = "SELECT * FROM book"
cursor.execute(book_query)
book_data = cursor.fetchall()
book_columns = [column[0] for column in cursor.description]
book_df = pd.DataFrame(book_data, columns=book_columns)

# ดึงข้อมูลจากตาราง book_type
book_type_query = "SELECT * FROM book_type"
cursor.execute(book_type_query)
book_type_data = cursor.fetchall()
book_type_columns = [column[0] for column in cursor.description]
book_type_df = pd.DataFrame(book_type_data, columns=book_type_columns)

# ดึงข้อมูลจากตาราง typebook
typebook_query = "SELECT * FROM typebook"
cursor.execute(typebook_query)
typebook_data = cursor.fetchall()
typebook_columns = [column[0] for column in cursor.description]
typebook_df = pd.DataFrame(typebook_data, columns=typebook_columns)

# ดึงข้อมูลจากตาราง typebook
btype_query = "SELECT bt.btype_bookid, GROUP_CONCAT(tb.type_name SEPARATOR ', ') AS Genres FROM book_type bt JOIN typebook tb ON bt.btype_typeid = tb.type_id GROUP BY bt.btype_bookid"
cursor.execute(btype_query)
btype_data = cursor.fetchall()
btype_columns = [column[0] for column in cursor.description]
btype_df = pd.DataFrame(btype_data, columns=btype_columns)

# ปิดการเชื่อมต่อ
cursor.close()
connection.close()

df = pd.merge(book_df, btype_df, left_on='book_id', right_on='btype_bookid', how='left')

df['combined_features'] = df['Genres'] + ' ' + df['book_summary']

# สร้าง TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# แปลงข้อมูลเนื้อหาเป็น TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# คำนวณความคล้ายคลึงระหว่างรายการ
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# ฟังก์ชันการคำนวณหนังสือที่แนะนำ
def recommend_books(titles, cosine_sim=cosine_sim, df=df):
    indices = []
    for title in titles:
        idx = df[df['book_id'] == title].index[0]
        indices.append(idx)
    sim_scores = []
    for idx in indices:
        sim_scores.extend(list(enumerate(cosine_sim[idx])))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # เรียงลำดับและเลือก 5 รายการแนะนำ
    book_indices = [i[0] for i in sim_scores]
    return df[['book_id', 'book_name', 'book_price', 'book_cover']].iloc[book_indices]

# API endpoint for book recommendations
@app.route('/recommendation', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    titles = data['titles']
    recommended_books = recommend_books(titles)
    return jsonify({'recommendations': recommended_books.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
