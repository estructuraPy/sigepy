import pathlib
import pandas as pd
import numpy as np
import json
import os

def get_file_location(file_name: str):
    notebook_location = pathlib.Path().absolute()
    parent_directory = notebook_location.parent
    data_folder = parent_directory / 'tests/test_data'
    file_location = data_folder / file_name
    return file_location

def import_sts_acceleration_txt(file_location: str, label: str) -> pd.DataFrame:
    """
    Processes acceleration data from a txt file exported and returns a Pandas DataFrame.

    Args:
        file_location: The path to the file containing the acceleration data.
        label: The label to return as series. X, Y or Z.

    Returns:
        A Pandas DataFrame containing the time and acceleration data.

    Assumptions:
        - The file exists and is readable.
        - The file encoding is 'latin1'.
        - The file contains at least two lines, with the second line being a header.
        - Each data line has at least 5 fields, with the 2nd, 3rd, 4th and 5th fields representing time, x, y, and z accelerations respectively.
    """
    try:
        with open(file_location, "r", encoding="latin1") as file:
            # Reads lines and skips the header
            lines = [line.strip() for line in file.readlines()[1:]]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_location}") from e
    except Exception as e:
        raise IOError(f"Error reading file: {file_location}") from e

    times = []
    accelerations = []

    for line in lines:
        fields = line.split()
        # Checks if the line has enough fields
        if len(fields) >= 5:
            try:
                time = float(fields[1])
                if label.upper() == "X":
                    acceleration = float(fields[2])
                elif label.upper() == "Y":
                    acceleration = float(fields[3])
                elif label.upper() == "Z":
                    acceleration = float(fields[4])
                else:
                    raise ValueError(
                        "Direction must be X, Y, or Z (case-insensitive)."
                    )
                times.append(time)
                accelerations.append(acceleration * 9.81)
            except ValueError as e:
                print(f"Skipping line due to invalid data: {line}. Error: {e}")  # Errors: Invalid data

    times = np.array(times)
    accelerations = np.array(accelerations)

    return pd.DataFrame({"Time": times, f"{label} Acceleration": accelerations})


def import_cscr_fed(file_location: str, json_location: str) -> pd.DataFrame:
    """
    Imports data from a TXT file, processes it, and returns a Pandas DataFrame.

    Args:
        file_location: Path to the TXT file.
        json_location: Path to the JSON file containing default values.

    Returns:
        A Pandas DataFrame with 'Frequency' and 'FED' columns, or None if an error occurs.

    Assumptions:
        - The TXT file is tab-separated and has two columns: 'T' and 'FED'.
        - The JSON file exists and is accessible.
    """
    try:
        # Convert paths to absolute paths
        json_location = os.path.abspath(json_location)
        file_location = os.path.abspath(file_location)

        # Read the JSON of default values
        with open(json_location, "r", encoding="utf-8") as f:
            default_values = json.load(f)

        # Attempt to read the TXT file with different encodings
        try:
            df = pd.read_csv(
                file_location,
                sep="\t",
                header=None,
                names=["T", "FED"],
                encoding="utf-16",
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                file_location,
                sep="\t",
                header=None,
                names=["T", "FED"],
                encoding="utf-8",
            )

        # Clean and convert the 'T' column to numeric
        df["T"] = pd.to_numeric(df["T"], errors="coerce")

        # Remove rows with NaN values in the 'T' column
        df = df.dropna(subset=["T"])

        # Calculate frequencies from the period 'T'
        df["Frequency"] = 1 / df["T"]

        # Select the columns of interest
        df = df[["Frequency", "FED"]]

        return df
    except FileNotFoundError as e:
        print(
            f"Error: {e}. Check that the JSON file and the TXT file are in the specified path."
        )
        return None