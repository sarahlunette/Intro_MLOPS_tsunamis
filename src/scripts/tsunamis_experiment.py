import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

score = pd.DataFrame(columns = range(80), index = range(90))
for k in range(1, 90):
  for i in range(1, 80):
    tsne = TSNE(n_components=2,perplexity=i, random_state=42)
    X_tsne = tsne.fit_transform(human_damages_scaled)
    km = KMeans(n_clusters= k)
    centroids = km.fit(X_tsne)
    human_damages_scaled['clustering'] = pd.Series(km.labels_)
    X = human_damages_scaled
    y = y_1

    # Augment data
    X_noisy = pd.concat([X, X.apply(add_noise)], axis = 0).reset_index(drop = True)
    y_noisy = pd.concat([y, y], axis = 0).reset_index(drop = True)

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_noisy, y_noisy, test_size=0.2, random_state=42)

    # Train the regression model
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    score.iloc[k-1, i-1] = r2

best_n = np.argmax(np.array(score.values))
print('Best number of clusters', best_n, 'Best score', score[best_n])