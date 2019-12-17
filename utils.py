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


def get_lidc_dataframes(data_path: str, num_sectors: int):
    sorted_studies = sorted(glob(data_path + "/*"))

    frames = []
    for study in sorted_studies:
        df_temp = pd.read_csv(
            os.path.join(study, "annotations.txt"), sep=": ", header=None, dtype=str
        )
        df_temp.columns = ["File", "Label"]
        df_temp["File"] = df_temp["File"].apply(lambda e: f"{study}/{e}.png")
        frames.append(df_temp)

    print(slice_sector(frames, num_sectors))

    return slice_sector(frames, num_sectors)


def slice_sector(frames, num_sectors: int):
    frame_size = len(frames[0])
    sector_size = int(frame_size / num_sectors)

    new_frames = []

    for frame in frames:
        new_frames.append(pd.DataFrame(columns=["File", "Label"]))
        for sector_idx in num_sectors:
            new_frames[-1] = frame.iloc([(sector_size * sector_idx):(sector_size * (sector_idx + 1))])

    return new_frames

