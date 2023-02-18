import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import pickle
import pandas as pd

data = pd.read_csv("input_data.csv")
subset = data.sample(n=40)
model_inputs = subset.drop(["outcome", "player_name", "x", "y"], axis=1)

model = pickle.load(open("finalized_model.sav", 'rb'))
xg = model.predict_proba(model_inputs)
subset.insert(0, "XG", xg[:, 1], True)

pitch = VerticalPitch(half=True, pad_bottom=-5, pitch_color='#aabb97', line_color='white', stripe_color='#c2d59d', stripe=True)
fig, ax = pitch.draw()
center_goal = (120, 40)

for _, row in subset.iterrows():
    if row["outcome"] == 1:   # if goal
        pitch.arrows(xstart=row["x"], ystart=row["y"], xend=center_goal[0], yend=center_goal[1], ax=ax, width=(0.3 + row["XG"]*4), color="red")
    else:
        pitch.arrows(xstart=row["x"], ystart=row["y"], xend=center_goal[0], yend=center_goal[1], ax=ax, width=(0.3 + row["XG"]*4), color="black")

plt.show()

