import os
import platform
import pandas as pd
from glob import glob


def get_keybase_root(dir: str = ""):
    if platform.system().lower() == "windows":
        return os.path.realpath(":k")
    elif platform.system().lower() == "darwin":
        return os.path.realpath("/Volumes/keybase")
    else:
        return os.path.realpath("/keybase")


def get_keybase_team(team_name: str):
    return os.path.join(get_keybase_root(), "team", team_name)


def get_lidc_dataframe(data_path: str):
    sorted_studies = sorted(glob(data_path + "/*"))

    frames = []
    for study in sorted_studies:
        df_temp = pd.read_csv(
            os.path.join(study, "annotations.txt"), sep=": ", header=None
        )
        df_temp.columns = ["File", "Label"]
        df_temp["File"] = df_temp["File"].apply(lambda e: study + str(e) + ".png")

        frames.append(df_temp)
    return pd.concat(frames)
