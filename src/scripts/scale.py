from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.manifold import TSNE
def scale(human_damages, houses_damages):
	sc1 = StandardScaler()
	sc2 = StandardScaler()
	y_1 = human_damages['human_damages']
	y_2 = houses_damages['houses_damages']
	human_damages_scaled = pd.DataFrame(sc1.fit_transform(human_damages.drop('human_damages', axis = 1)), columns = human_damages.drop('human_damages', axis = 1).columns)
	houses_damages_scaled = pd.DataFrame(sc2.fit_transform(houses_damages.drop('houses_damages', axis = 1)), columns = houses_damages.drop('houses_damages', axis = 1).columns)
	human_damages_scaled = pd.concat([human_damages_scaled, y_1], axis = 1)
	houses_damages_scaled = pd.concat([houses_damages_scaled, y_2], axis = 1)
	return human_damages_scaled, houses_damages_scaled