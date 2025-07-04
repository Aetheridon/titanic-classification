import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("titanic.csv")
df['Age'] = df['Age'].fillna(df['Age'].median())

def bar_graph_survival_gender(df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Sex', hue='Survived', palette='pastel')
    plt.title('Survival Counts by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.savefig("gender_survival.png")

def bar_graph_survival_class(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Pclass', hue='Survived', palette='pastel')
    plt.title('Survival Counts by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.savefig("class_survival.png")

def age_distribution_by_survival(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', palette='pastel', bins=30)
    plt.title('Age Distribution by Survival')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.savefig("age_distribution.png")

def survival_by_family_size(df):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='FamilySize', hue='Survived', palette='pastel')
    plt.title('Survival Counts by Family Size')
    plt.xlabel('Family Size')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.savefig("family_survival.png")

bar_graph_survival_gender(df)
bar_graph_survival_class(df)
age_distribution_by_survival(df)
survival_by_family_size(df)