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

    return [balance_frame(frame) for frame in slice_sector(frames, num_sectors)]


def balance_frame(frame):
    tumor_sample_df = frame.loc[frame["Label"] != "0"]
    no_nodule_df = frame.loc[frame["Label"] == "0"]

    no_nodule_sample = no_nodule_df.sample(n=len(tumor_sample_df) // 2)

    return pd.concat([tumor_sample_df, no_nodule_sample])


def slice_sector(frames, num_sectors: int):
    frame_size = len(frames[0])
    sector_size = int(frame_size / num_sectors)

    new_frames = []

    for sector_idx in range(num_sectors):
        new_frames.append(pd.DataFrame(columns=["File", "Label"], dtype=object))
        for frame in frames:
            lower = sector_size * sector_idx
            upper = sector_size * (sector_idx + 1)
            new_frames[-1] = new_frames[-1].append(
                frame.iloc[lower:upper], ignore_index=True
            )

    return new_frames

