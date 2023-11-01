from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import openai
import pandas as pd
import mysql.connector

load_dotenv()

openai.api_key = os.getenv('OPEN_API_KEY')

df = pd.read_csv('/app/anime-dataset-2023.csv', usecols=['Name', 'Image URL'])

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

df = df.merge(book_df, left_on='Name', right_on='book_name', how="left")

# ปิดการเชื่อมต่อ
cursor.close()
connection.close()

app = Flask(__name__)
CORS(app)


def mychatbot(query, df):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a manga guide. You have a lot of knowledge about manga, you can recommend any kind of manga."},
            {"role": "user", "content": f"Please answer my queries according to the given context \nContext: {df}"},
            {"role": "assistant", "content": "Okay sure!"},
            {"role": "user", "content": f"{query} Reply with only the name of the manga, with a maximum of 10 names. No matter what you ask, always reply in the same format."}
        ]
    )
    response_content = res['choices'][0]['message']['content']
    manga_list = response_content.split('\n')
    manga_names = [manga.split('. ')[1] for manga in manga_list if '. ' in manga]
    response = "\n".join(manga_names) if manga_names else "No response found."
    return response


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['query']
    response_str = mychatbot(query, df)
    manga_list = response_str.split('\n')

    data_list = []
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

    response_data = {
        "response": response_str,
        "data": data_list
    }
    return jsonify(response_data), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
