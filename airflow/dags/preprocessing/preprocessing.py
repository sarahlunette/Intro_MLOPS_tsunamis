from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_test(df, column = 'Outcome'):
  X = df.drop(column, axis = 1)
  y = df[column]
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, stratify = y)
  return X_train, X_test_y_train, y_test

def scaling(X_train, X_test):
  sc = StandardScaler()
  X_train_standard = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  return X_train_standard, X_test_standard

# Ça ça complique la compréhension du truc (pas besoin de fonctions dans un preprocessing simple, quand est-ce qu'on met des fonctions,
# comment est-ce qu'on organise ça dans un projet large-scale)
