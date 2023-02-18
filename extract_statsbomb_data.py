from collections import defaultdict
from mplsoccer import Sbopen
import numpy as np
import pandas as pd
import pickle

parser = Sbopen()

center_goal = (120, 40)

statsbomb_ids = {11: [90, 42, 4, 1, 2, 27, 26, 25, 24, 23, 22, 21, 41, 40, 39, 38, 37],     # la liga - male
                 16: [4, 1, 2, 27, 26, 25, 24, 23, 22, 21, 41, 39, 37, 44],             # premier league - male
                 43: [106, 3],                                                              # world cup
                 55: [43]                                                                   # euro
                 }

match_ids = []
for competition in statsbomb_ids.keys():
    for season in statsbomb_ids[competition]:
        match_df = parser.match(competition_id=competition, season_id=season)
        match_ids.extend(match_df["match_id"].values)

match_df = pd.DataFrame(columns=["match_id"], data=match_ids)

shots_data = defaultdict(list)
for _, row in match_df.iterrows():
    event_data, _, _, _ = parser.event(row["match_id"])
    event_data.sort_values(by=["index"], inplace=True)

    shots = event_data[event_data["type_name"] == "Shot"]

    # goal or not
    outcome = shots["outcome_id"].to_numpy()
    outcome = list(np.where(outcome == 97, 1, 0))
    shots_data["outcome"].extend(outcome)

    shots_data["player_name"].extend(shots["player_name"])

    # angle and distance to goal
    x = shots["x"].to_numpy()
    y = shots["y"].to_numpy()
    shots_data["x"].extend(x)
    shots_data["y"].extend(y)

    distance = np.sqrt((x-center_goal[0])**2 + (y-center_goal[1])**2)       # Euclidean distance
    angle = np.degrees(np.arctan(abs(y-center_goal[1])/abs(x-center_goal[0])))
    shots_data["distance"].extend(distance)
    shots_data["angle"].extend(angle)

    # under_pressure or not
    pressure = shots["under_pressure"].to_numpy()
    pressure = list(np.where(np.isnan(pressure), 0, pressure))
    shots_data["pressure"].extend(pressure)

    # what type of goal e.g. from corner, open play
    play_pattern = shots["play_pattern_id"].to_list()
    shots_data["play_pattern"].extend(play_pattern)

    # what type of shot e.g. volley or normal
    shot_technique = shots["technique_id"].to_list()
    shots_data["shot_technique"].extend(shot_technique)

pickle.dump(shots_data, open("shots_data_dict.p", "wb"))