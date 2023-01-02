import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np

pd.options.mode.chained_assignment = None

df = pd.read_csv("../datasets/2022-12-17.csv")
df = pd.DataFrame(df)
df.drop("City", inplace=True, axis=1)
desired_width = 320
pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns', 13)


def name_to_num(list1):
    n = 1
    d = dict()
    for q in list1:
        if q not in d:
            d[q] = n
            n += 1
    return d


for i, v in enumerate(df['Building Age']):
    if 'arası' in v:
        df['Building Age'][i] = v.split(' ')[0]
        df['Building Age'][i] = v.split('-')[0]
    elif 'üzeri' in v:
        df['Building Age'][i] = v.split(' ')[0]

for i, v in enumerate(df['Balcony']):
    if v == "Var":
        df['Balcony'][i] = 1
    else:
        df['Balcony'][i] = 0

for i, v in enumerate(df['Floor Location']):
    if v == "Zemin Kat" or "Giriş" in v or v == "Bahçe Katı" or v == 'Bodrum Kat':
        df['Floor Location'][i] = 0
    elif v == "Müstakil" or v == "Yüksek Giriş":
        df['Floor Location'][i] = 1
    elif v == "Çatı Katı":
        df['Floor Location'][i] = df['Number of Floors'][i]
    else:
        df['Floor Location'][i] = v

for i, v in enumerate(df['Heating']):
    if v == "Yok":
        df['Heating'][i] = 0
    elif v == 'Soba' or v == 'Doğalgaz Sobası' or v == 'Kat Kaloriferi':
        df['Heating'][i] = 1
    elif v == 'Elektrikli Radyatör':
        df['Heating'][i] = 2
    elif v == 'Klima':
        df['Heating'][i] = 3
    else:
        df['Heating'][i] = 4

for i, v in enumerate(df['Number of Rooms']):
    if v == 'Stüdyo (1+0)':
        df['Number of Rooms'][i] = 1
    else:
        df['Number of Rooms'][i] = eval(v)

for i, v in enumerate(df['Furnished']):
    if v == "Evet":
        df['Furnished'][i] = 1
    else:
        df['Furnished'][i] = 0

a = set(list(df["District"]))
q_list = name_to_num(a)
for i, v in enumerate(df['District']):
    df['District'][i] = q_list[v]

a = set(list(df["Quarter"]))
q_list = name_to_num(a)
for i, v in enumerate(df['Quarter']):
    df['Quarter'][i] = q_list[v]

X = df.drop(['Price'], axis=1)
y = df['Price']
X_learn, X_test, y_learn, y_test = train_test_split(X, y,random_state=0)
regressor = LinearRegression()
regressor.fit(X_learn, y_learn)

y_pred = regressor.predict(X_test)
for i, v in enumerate(y_pred):
    y_pred[i] = int(v)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})

print(df_preds)

