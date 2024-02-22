from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import openai
import pandas as pd
import mysql.connector

load_dotenv()

# โหลดคีย์ API ของ OpenAI จากตัวแปรสภาพแวดล้อม
openai.api_key = os.getenv('OPEN_API_KEY')

# โหลดชุดข้อมูลอนิเมะเข้าสู่ DataFrame
df = pd.read_csv('../anime-dataset-2023.csv', usecols=['Name', 'Image URL'])

# กำหนดค่าพารามิเตอร์สำหรับการเชื่อมต่อกับ MySQL
config = {
    'user': 'root',
    'password': 'root',
    'port': 9906,
    'host': '103.114.200.105',  # ที่อยู่ฐานข้อมูล MySQL
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

# รวม DataFrame อนิเมะกับ DataFrame หนังสือตามชื่อ
df = df.merge(book_df, left_on='Name', right_on='book_name', how="left")

# ปิดการเชื่อมต่อกับฐานข้อมูล
cursor.close()
connection.close()

app = Flask(__name__)
CORS(app)

def mychatbot(query, df):
    # ใช้ API ChatCompletion ของ OpenAI เพื่อประมวลผลคำถาม
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "คุณเป็นไกด์มังงะ คุณมีความรู้เกี่ยวกับมังงะมากมาย คุณสามารถแนะนำมังงะแบบใดก็ได้"},
            {"role": "user", "content": f"กรุณาตอบคำถามของฉันตามบริบทที่กำหนด \nบริบท: {df}"},
            {"role": "assistant", "content": "ได้เลยครับ!"},
            {"role": "user", "content": f"{query} ตอบเฉพาะชื่อมังงะเท่านั้น ไม่เกิน 10 ชื่อ ไม่ว่าคุณจะถามอะไร กรุณาตอบในรูปแบบเดียวกันนี้"}
        ]
    )
    # ประมวลผลคำตอบจาก OpenAI
    response_content = res['choices'][0]['message']['content']
    manga_list = response_content.split('\n')
    manga_names = [manga.split('. ')[1] for manga in manga_list if '. ' in manga]
    response = "\n".join(manga_names) if manga_names else "ไม่พบคำตอบ"
    return response

@app.route('/chat', methods=['POST'])
def chat():
    # รับคำถามจากคำขอ POST
    data = request.get_json()
    query = data['query']
    response_str = mychatbot(query, df)
    manga_list = response_str.split('\n')

    data_list = []
    # ดึงข้อมูลที่เกี่ยวข้องสำหรับแต่ละมังงะในคำตอบ
    for manga_name in manga_list:
        manga_name = manga_name.strip()
        if manga_name in df['Name'].values:
            relevant_data = df[df['Name'] == manga_name]
            data_list.append({
                "book_id": relevant_data["book_id"].values[0],
                "book_name": relevant_data["Name"].values[0],
                "book_summary": relevant_data["book_summary"].values[0],
                "image_url": relevant_data["Image URL"].values[0]
            })

    # เตรียมและส่งคำตอบ
    response_data = {
        "response": response_str,
        "data": data_list
    }
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
