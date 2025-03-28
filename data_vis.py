import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# print(df.head(5))
# print(df.isnull().sum())
# print(df.dtypes)
# print(df.duplicated().sum())

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path, sep=':::', names=['title', 'genre', 'description'], engine="python")
    df_test = pd.read_csv(test_path, sep=':::', names=['id', 'title', 'description'], engine="python")
    return df_train, df_test

def basic_eda(df: pd.DataFrame):
    print("Head of the DataFrame:")
    print(df.head(5))
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)
    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())
    print("\nGenre distribution:")
    print(df['genre'].value_counts())

def plot_genre_distribution(df):
    
    plt.figure(figsize=(20, 10))
    colors = ['#264653']
    ax = sns.countplot(data=df, y='genre', order=df['genre'].value_counts().index, color=colors[0])
    plt.title('Number Of Movies By Genre', fontsize=20)
    plt.ylabel('Genre', fontsize=16)
    plt.xlabel('Number Of Movies', fontsize=16)
    ax.set_xticks([])
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_width())}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2), 
                    ha='center', va='center', 
                    fontsize=12, color='black', 
                    xytext=(20, 0), textcoords='offset points')
    plt.tight_layout()
    plt.show()

def main():
    train_path = "./Genre Classification Dataset/train_data.txt"
    test_path = "./Genre Classification Dataset/train_data.txt"  # Adjust if different
    
    df_train, df_test = load_data(train_path, test_path)
    basic_eda(df_train)
    plot_genre_distribution(df_train)

if __name__ == "__main__":
    main()



