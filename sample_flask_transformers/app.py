# app.py
from flask import Flask, render_template, request
from faiss import IndexFlatL2, read_index, write_index
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

app = Flask(__name__)
global model
global shops_index
global df_ramen


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # リクエストパラメータを取得する
    query = request.form.get('query')
        # ここで検索処理を実装する
    embedding = model.encode([query])
    # 類似度を計算する
    search_score, search_index = shops_index.search(embedding,3)
    results = []
    for  i in range(len(search_index[0])):
        results.append(df_ramen.iloc[search_index[0][i]])
    return render_template('search_results.html', results=results)

if __name__ == '__main__':
    # モデルを読み込む
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    # indexファイルを読み込む
    shops_index = read_index("ramen_shops.index")
    # データを読み込む
    df_ramen = pd.read_csv('ramen_shops.csv')
    app.run(debug=True)
