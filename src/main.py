import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データの読み込み（例: Titanicデータセット）
data = pd.read_csv('../data/titanic.csv')  # 適切なパスに置き換えてください

# 使用するカラムだけに絞る（重要）
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'
data = data[features + [target]]

# 欠損値処理（AgeやEmbarkedなど）
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# カテゴリ変数のエンコード（label encoding）
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 特徴量とターゲットに分離
X = data[features]
y = data[target]

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの訓練
model = lgb.LGBMClassifier(
    device='gpu',
    max_depth=6,
    num_leaves=31,
    min_data_in_leaf=20,
    max_bin=512,  # ← GPUではデフォルト255。上げると分割しやすくなる
    categorical_feature=[1, 6]  # 'Sex', 'Embarked' の列番号 or カラム名でもOK
    )  # CPUのときはdeviceを省略
model.fit(X_train, y_train)

# 予測と評価
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")