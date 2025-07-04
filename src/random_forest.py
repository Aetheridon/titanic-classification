import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("titanic.csv")

def transform_data(df: pd.DataFrame):
    df.drop(columns=["PassengerId", "Cabin", "Name", "Embarked", "Ticket"], inplace=True)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

transform_data(df=df)

y = df["Survived"]
X = df.drop(columns=["Survived"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")