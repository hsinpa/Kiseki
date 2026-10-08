from pathlib import Path
import pandas as pd


def read_csv(csv_paths: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(Path(f)) for f in csv_paths], ignore_index=True)

def load_csv_folder(folder: str, recursive: bool = False, **read_csv_kwargs) -> pd.DataFrame:
    """Read all .csv files in a folder into a single DataFrame.

    Adds a 'source_file' column with the originating filename.
    """
    pattern = "**/*.csv" if recursive else "*.csv"
    paths = sorted(Path(folder).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No .csv files found in {folder}")

    frames = []
    for p in paths:
        df = pd.read_csv(p, **read_csv_kwargs)
        df["source_file"] = p.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def query_last_within_n_days(df: pd.DataFrame, time_column:str, query_date: pd.Timestamp, n_days: int):
    # Normalize the query date to a tz-aware Timestamp
    query_date = query_date.normalize()   # <-- strip time, snap to 00:00:00
    window_start = query_date - pd.Timedelta(days=n_days)

    mask = (df[time_column] < query_date) & (df[time_column] > window_start)
    result = df.loc[mask]

    if result.empty:
        return None
    # Last row by time (sort first if df isn't guaranteed ordered)
    return result.sort_values(time_column).iloc[-1]
