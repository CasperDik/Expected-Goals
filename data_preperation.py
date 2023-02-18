import pickle
import pandas as pd

data = pickle.load(open("shots_data_dict.p", "rb"))

# load to dataframe
df = pd.DataFrame.from_dict(data)

# normalize continuous variables --> angle, distance
df[["angle", "distance"]] = (df[["angle", "distance"]] - df[["angle", "distance"]].mean()) / df[["angle", "distance"]].std()

# convert categorical variables to dummy --> shot_technique, play_pattern
df = pd.get_dummies(df, prefix=["shot_technique", "play_pattern"], columns=["shot_technique", "play_pattern"])

df.to_csv("input_data.csv", index=False)
