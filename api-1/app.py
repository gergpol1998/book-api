from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector

app = Flask(__name__)
CORS(app)

# กำหนดค่าพารามิเตอร์สำหรับการเชื่อมต่อกับ MySQL
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


# ดึงข้อมูลหนังสือจากตาราง 'book' ในฐานข้อมูล
book_query = "SELECT book_id,book_name,book_cover,book_price,bscore_score,bscore_cusid FROM book JOIN bookscore ON bscore_bookid = book_id"
cursor.execute(book_query)
book_data = cursor.fetchall()
book_columns = [column[0] for column in cursor.description]
rating_df = pd.DataFrame(book_data, columns=book_columns)

def preprocess_data():
    # คำนวณจำนวนการให้คะแนนต่อหนังสือ
    num_rating_df = rating_df.groupby('book_name').count()['bscore_score'].reset_index()
    num_rating_df.rename(columns={'bscore_score': 'num_ratings'}, inplace=True)

    # คำนวณคะแนนเฉลี่ยต่อหนังสือ
    avg_rating_df = rating_df.groupby('book_name')['bscore_score'].mean().reset_index()
    avg_rating_df.rename(columns={'bscore_score': 'avg_rating'}, inplace=True)

    # ผสานข้อมูลจำนวนคะแนนและคะแนนเฉลี่ย
    popular_df = num_rating_df.merge(avg_rating_df, on='book_name')

    # กรองหนังสือที่มีความนิยม
    popular_df = popular_df[popular_df['num_ratings'] >= 3].sort_values('avg_rating', ascending=False).head(50)

    # ระบุผู้ใช้ที่ใช้งานอยู่
    x = rating_df.groupby('bscore_cusid').count()['bscore_score'] > 2
    padhe_likhe_users = x[x].index

    # กรองการให้คะแนนจากผู้ใช้ที่ใช้งานอยู่
    filtered_rating = rating_df[rating_df['bscore_cusid'].isin(padhe_likhe_users)]

    # ระบุหนังสือที่มีชื่อเสียง
    y = filtered_rating.groupby('book_name').count()['bscore_score'] >= 5
    famous_books = y[y].index

    # สร้างตารางคะแนนสุดท้าย
    final_ratings = filtered_rating[filtered_rating['book_name'].isin(famous_books)]
    pt = final_ratings.pivot_table(index='book_name', columns='bscore_cusid', values='bscore_score')
    pt.fillna(0, inplace=True)

    return pt

# สร้างตารางประชิดและคำนวณคะแนนความคล้ายคลึงกัน
pt = preprocess_data()
similarity_scores = cosine_similarity(pt)

def random_book_recommendation(n=5):
    # สุ่มเลือกหนังสือจากฐานข้อมูล
    random_books = rating_df.sample(n)
    return random_books.to_dict(orient='records')

def recommend(book_name, already_recommended):
    try:
        # ตรวจสอบว่าหนังสือนี้ได้รับแนะนำไว้แล้วหรือไม่
        if book_name in already_recommended:
            return None

        # หาดัชนีของหนังสือในตาราง
        index = np.where(pt.index == book_name)[0][0]
        # ดึงหนังสือที่มีความคล้ายคลึงกันมากที่สุด
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

        data = []
        for i in similar_items:
            item = {}
            # ตรวจสอบว่าหนังสือที่แนะนำได้รับแนะนำไว้แล้วหรือไม่
            temp_book_name = pt.index[i[0]]
            if temp_book_name not in already_recommended:
                # ดึงรายละเอียดสำหรับแต่ละหนังสือที่แนะนำ
                temp_df = rating_df[rating_df['book_name'] == temp_book_name]
                item['book_name'] = temp_df['book_name'].values[0]
                item['book_id'] = temp_df['book_id'].values[0]
                item['book_cover'] = temp_df['book_cover'].values[0]
                item['book_price'] = int(temp_df['book_price'].values[0])
                data.append(item)

        # เสริมด้วยการแนะนำแบบสุ่มหากไม่พบรายการที่คล้ายคลึงกันเพียงพอ
        if len(data) < 5:
            return random_book_recommendation(5 - len(data)) + data
        return data
    except IndexError:
        # กลับมาแนะนำแบบสุ่มหากไม่พบหนังสือในชุดข้อมูล
        return random_book_recommendation()

# จุดสิ้นสุดของ API สำหรับการดึงข้อมูลแนะนำหนังสือ
@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    # ตรวจสอบให้แน่ใจว่าคำขอเป็นรูปแบบ JSON
    if not request.is_json:
        return jsonify({'error': 'Invalid JSON'}), 400

    data = request.get_json()
    # ตรวจสอบพารามิเตอร์ที่ต้องการ
    if 'book_names' not in data:
        return jsonify({'error': 'Missing parameter book_names'}), 400

    book_names = data['book_names']
    # สร้างการแนะนำสำหรับแต่ละหนังสือ
    recommendations = []
    already_recommended = set()  # เก็บรายการหนังสือที่ได้รับแนะนำไว้แล้ว
    for book in book_names:
        recommendation = recommend(book, already_recommended)
        if recommendation:
            recommendations.append(recommendation)
            # เพิ่มรายการที่ได้รับแนะนำลงในเซต
            already_recommended.update(rec['book_name'] for rec in recommendation)

    response_data = {"recommendations": recommendations}
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
