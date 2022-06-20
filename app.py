# 파이썬 Flask 사용법 : https://hleecaster.com/flask-introduction/

import json
import flask
from flask import Flask, render_template, request
import my_text
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("project2.html")


@app.route('/api', methods=['GET'])
def api():
    # URL 매개 변수 추출하기
    q = request.args.get('q', '')
    if q == '':
        return '{"label": "내용을 입력해주세요", "per":0}'
    print("q=", q)
    # 텍스트 카테고리 판별하기
    label, per, no = my_text.check_genre(q)
    # 결과를 JSON으로 출력하기
    return json.dumps({
        "label": label,
        "pre": np.round(per, 2)*100,
        "genre_no": no
    })


if __name__ == "__main__":
    app.run(debug=True)
