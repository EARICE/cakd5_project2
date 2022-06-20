from cgitb import text
from konlpy.tag import Okt
import pickle
import tfidf
import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json

# 텍스트 준비하기 --- ( ※ 1)
text1 = """
대통령이 북한과 관련된 이야기로 한미 정상회담을 준비하고 있습니다.
"""
text2 = """
iPhone과 iPad를 모두 가지고 다니므로 USB를 2개 연결할 수 있는 휴대용 배터리를 선호합니다.
"""
text3 = """
이번 주에는 미세먼지가 많을 것으로 예상되므로 노약자는 외출을 자제하는 것이 좋습니다.
"""

# TF-IDF 읽어 들이기
tfidf.load_dic("dataset/genre-tfidf14.dic")

# Keras 모델 정의, 가중치 데이터 읽어 들이기
nb_classes = 14
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(26167,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy'])
model.load_weights('./dataset/genre-model14.hdf5')


def check_genre(text):
    LABELS = ["정치","외교","행정","투자","경제","생활 ","사건","복지","보건","사회","아시아","미국","유럽","세계"]
    # TF-IDF 벡터로 변환하기
    data = tfidf.calc_text(text)
    # MLP로 예측하기
    pre = model.predict(np.array([data]))[0]
    n = pre.argmax()
    print(LABELS[n], "(", pre[n], ")")
    return LABELS[n], float(pre[n]), int(n)


if __name__ == '__main__':
    check_genre(text1)
    check_genre(text2)
    check_genre(text3)
