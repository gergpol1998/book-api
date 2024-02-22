import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# ตั้งค่าการเชื่อมต่อกับฐานข้อมูล MySQL
config = {
    'user': 'root',
    'password': 'root',
    'port': 9906,
    'host': '45.136.238.139',  # ที่อยู่ฐานข้อมูล MySQL
    'database': 'ebook',    
}

# สร้างการเชื่อมต่อกับฐานข้อมูล MySQL
connection = mysql.connector.connect(**config)
cursor = connection.cursor()

# ดึงข้อมูลจากตาราง 'book' ในฐานข้อมูล
book_query = "SELECT * FROM book"
cursor.execute(book_query)
book_data = cursor.fetchall()
book_columns = [column[0] for column in cursor.description]
book_df = pd.DataFrame(book_data, columns=book_columns)

# ดึงข้อมูลจากตาราง 'book_type'
book_type_query = "SELECT * FROM book_type"
cursor.execute(book_type_query)
book_type_data = cursor.fetchall()
book_type_columns = [column[0] for column in cursor.description]
book_type_df = pd.DataFrame(book_type_data, columns=book_type_columns)

# ดึงข้อมูลจากตาราง 'typebook'
typebook_query = "SELECT * FROM typebook"
cursor.execute(typebook_query)
typebook_data = cursor.fetchall()
typebook_columns = [column[0] for column in cursor.description]
typebook_df = pd.DataFrame(typebook_data, columns=typebook_columns)

# ดึงและรวมประเภทของแต่ละหนังสือ
btype_query = "SELECT bt.btype_bookid, GROUP_CONCAT(tb.type_name SEPARATOR ', ') AS Genres FROM book_type bt JOIN typebook tb ON bt.btype_typeid = tb.type_id GROUP BY bt.btype_bookid"
cursor.execute(btype_query)
btype_data = cursor.fetchall()
btype_columns = [column[0] for column in cursor.description]
btype_df = pd.DataFrame(btype_data, columns=btype_columns)

# ปิดการเชื่อมต่อกับฐานข้อมูล
cursor.close()
connection.close()

# รวมข้อมูลหนังสือกับข้อมูลประเภทหนังสือตามรหัสหนังสือ
df = pd.merge(book_df, btype_df, left_on='book_id', right_on='btype_bookid', how='left')

# รวมประเภทหนังสือและสรุปเนื้อหาหนังสือเป็นคุณลักษณะเดียวสำหรับการวิเคราะห์
df['combined_features'] = df['Genres'] + ' ' + df['book_summary']

# สร้าง Vectorizer แบบ TF-IDF เพื่อแปลงข้อมูลข้อความเป็นเวกเตอร์
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# แปลงข้อความคุณลักษณะที่รวมไว้เป็นเวกเตอร์ TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# คำนวณความคล้ายคลึงกันด้วย cosine ระหว่างคู่หนังสือทั้งหมด
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# ฟังก์ชันสำหรับแนะนำหนังสือตามความคล้ายคลึงกันด้วย cosine
def recommend_books(titles, cosine_sim=cosine_sim, df=df):
    indices = []
    # หาดัชนีของหนังสือที่ระบุใน DataFrame
    for title in titles:
        idx = df[df['book_id'] == title].index[0]
        indices.append(idx)

    # ลบดัชนีที่ซ้ำกัน
    unique_indices = list(set(indices))
    
    sim_scores = []
    # รวมคะแนนความคล้ายคลึงกันสำหรับแต่ละดัชนี
    for idx in unique_indices:
        sim_scores.extend(list(enumerate(cosine_sim[idx])))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # ลบหนังสือที่แนะนำไปแล้ว
    recommended_indices = []
    for score in sim_scores:
        if score[0] not in unique_indices and score[0] not in recommended_indices:
            recommended_indices.append(score[0])
    
    # เลือก 5 หนังสือที่แนะนำสูงสุด ยกเว้นหนังสือที่เลือกไปแล้ว
    recommended_indices = recommended_indices[:5]
    book_indices = [i for i in recommended_indices if i not in unique_indices]
    
    # กลับคืนข้อมูลหนังสือที่แนะนำ
    return df[['book_id', 'book_name', 'book_price', 'book_cover']].iloc[book_indices]

# จุดสิ้นสุดของ API สำหรับการดึงข้อมูลแนะนำหนังสือ
@app.route('/recommendation', methods=['POST'])
def get_recommendations():
    # รับข้อมูลจากคำขอ POST
    data = request.get_json()
    titles = data['titles']
    recommended_books = recommend_books(titles)
    return jsonify({'recommendations': recommended_books.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
