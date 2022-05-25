import pandas as pd
data = "../duolingo.csv"
words = []

print('data processing...')

df = pd.read_csv(data)

df1 = df[df['learning_language'] == 'en']
df1 = df1.reset_index(drop=True)
df1.to_csv('duolingo_en.csv')





