import json
from scipy.io import loadmat
from torch.utils.data import Dataset
import os
import urllib.request as urllib
import patoolib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import torch
from .dataset import BearingDataset

faults = {
    "Healthy": 0,
    "InnerRace": 1,
    "OuterRace": 2,
}

artificial_damage_codes = [
    "KA01",
    "KA03",
    "KA05",
    "KA06",
    "KA07",
    "KA08",
    "KA09",
    "KI01",
    "KI03",
    "KI05",
    "KI07",
    "KI08",
]

real_damage_codes = [
    "KA04",
    "KA15",
    "KA16",
    "KA22",
    "KA30",
    "KB23",
    "KB24",
    "KB27",
    "KI04",
    "KI14",
    "KI16",
    "KI17",
    "KI18",
    "KI21",
]

rotational_speeds = {
    "N15": 1500,
    "N09": 900,
}

load_torques = {
    "M07": 0.7,
    "M01": 0.1,
}

radial_forces_newton = {
    "F04": 400,
    "F10": 1000,
}


class PU_DatasetProcessor:
    def __init__(
        self,
        rdir,
        window_size=2048,
        step_size=2048,
        train_size=0.8,
        artificial_damage=True,
        rotational_speeds=rotational_speeds,
        load_torques=load_torques,
        radial_forces_newton=radial_forces_newton,
        force_reload=False,
        resplit_train_test=False,
        seed=None,
    ):
        """
        Initialize PU Dataset

        Args:
            rdir: Root directory for dataset
            window_size: Size of sliding window
            step_size: Step size for sliding window
            train_size: Proportion of data for training
            artificial_damage: Whether to use artificial damage data
            rotational_speeds: Dictionary of rotational speeds to include
            load_torques: Dictionary of load torques to include
            radial_forces_newton: Dictionary of radial forces to include
            force_reload: If True, reprocess all data even if processed files exist
            resplit_train_test: If True, reprocess train/test split
            seed: Random seed for reproducibility
        """
        self.rdir = rdir
        self.seed = seed
        self.window_size = window_size
        self.step_size = step_size
        self.train_size = train_size
        self.artificial_damage = artificial_damage
        self.rotational_speeds = rotational_speeds
        self.load_torques = load_torques
        self.radial_forces_newton = radial_forces_newton
        self.force_reload = force_reload
        self.resplit_train_test = resplit_train_test
        self.data = None
        self.data_train = None
        self.data_test = None

        self.files_information = {
            "files": {"Healthy": [], "InnerRace": [], "OuterRace": []}
        }

        self.url_list = []
        self.process_files()

    def process_files(self):
        """
        Main method to process dataset files.
        Downloads, processes, and splits data into train/test sets.
        """
        # Check if processed data already exists and force_reload is False
        if not self.force_reload and self._load_existing_processed_data():
            if self.resplit_train_test:
                print("Resplitting train/test...")
                self._train_test_split()
            else:
                print("Loaded existing processed data.")
                return

        print("Processing dataset files...")

        # Download if not exists
        self._download()

        # Get files information
        self._load_or_create_files_info()

        # Create processed folder if not exists
        processed_dir = os.path.join(self.rdir, "processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        # Load and process data
        self._process_and_save_data()

        # Split into train/test
        self._train_test_split()

        # Save files information
        self._save_files_info()

        print("Dataset processing complete.")

    def _load_existing_processed_data(self):
        """
        Load existing processed data if available.

        Returns:
            bool: True if data was successfully loaded, False otherwise
        """
        train_csv_path = os.path.join(self.rdir, "data_train.csv")
        test_csv_path = os.path.join(self.rdir, "data_test.csv")

        if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
            try:
                self.data_train = pd.read_csv(train_csv_path)
                self.data_test = pd.read_csv(test_csv_path)
                self.data = pd.concat(
                    [self.data_train, self.data_test], ignore_index=True
                )
                return True
            except Exception as e:
                print(f"Error loading existing data: {e}")
                return False
        return False

    def _load_or_create_files_info(self):
        """Load existing files information or create new one."""
        files_info_path = os.path.join(self.rdir, "files_information.json")

        if os.path.exists(files_info_path) and not self.force_reload:
            try:
                with open(files_info_path, "r") as f:
                    self.files_information = json.load(f)
                print("Loaded existing files information.")
            except Exception as e:
                print(f"Error loading files information: {e}")
                self._get_files_info()
        else:
            self._get_files_info()

    def _process_and_save_data(self):
        """Process all relevant files and save processed data."""
        file_names = []
        labels = []
        window_counts = []

        for class_name in self.files_information["files"]:
            for i, file in enumerate(self.files_information["files"][class_name]):
                if self._should_process_file(file):
                    file_path, num_windows = self._process_single_file(file, class_name)

                    # Update file information
                    self.files_information["files"][class_name][i].update(
                        {
                            "processed": True,
                            "number_of_windows": num_windows,
                            "processed_file": file_path,
                        }
                    )

                    file_names.append(file_path)
                    labels.append(faults[class_name])
                    window_counts.append(num_windows)

        self.data = pd.DataFrame(
            {"file_name": file_names, "window_counts": window_counts, "label": labels}
        )

    def _should_process_file(self, file):
        """
        Determine if a file should be processed based on configuration.

        Args:
            file: File information dictionary

        Returns:
            bool: True if file should be processed
        """
        # Check if file matches our speed/torque/force criteria
        if not (
            file["rot_speed"] in self.rotational_speeds.keys()
            and file["load_torque"] in self.load_torques.keys()
            and file["radial_force"] in self.radial_forces_newton.keys()
        ):
            return False

        # Check damage type criteria
        if file.get("class_name") == "Healthy":
            return True

        if self.artificial_damage:
            return file["code"] in artificial_damage_codes
        else:
            return file["code"] in real_damage_codes

    def _process_single_file(self, file, class_name):
        """
        Process a single file and return the processed file path and window count.

        Args:
            file: File information dictionary
            class_name: Class name (Healthy, InnerRace, OuterRace)

        Returns:
            tuple: (processed_file_path, number_of_windows)
        """
        # Create class directory if needed
        class_dir = os.path.join(self.rdir, "processed", class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Generate processed file name
        original_file_name = file["file"].split("/")[-1].split(".")[0]
        processed_filename = (
            f"{file['code']}_{file['rot_speed']}_{file['load_torque']}_"
            f"{file['radial_force']}_{file['rep']}_{self.window_size}_{self.step_size}.npy"
        )
        processed_file_path = os.path.join(class_dir, processed_filename)

        # Check if processed file exists and force_reload is False
        if os.path.exists(processed_file_path) and not self.force_reload:
            number_of_windows = np.load(processed_file_path).shape[0]
            print(f"Using existing processed file: {processed_filename}")
        else:
            # Load and process the original file
            print(f"Processing file: {original_file_name}")
            mat_data = loadmat(file["file"])
            data = mat_data[original_file_name]["Y"][0][0][0][6][2].reshape((-1))
            windowed_data = slicer(
                data, self.window_size, self.step_size, return_df=False
            )
            number_of_windows = windowed_data.shape[0]

            # Save processed data
            np.save(processed_file_path, windowed_data)

            # Clean up memory
            del mat_data, data, windowed_data

        return processed_file_path, number_of_windows

    def _train_test_split(self):
        """Split data into training and testing sets."""
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data available for train/test split")

        self.data_train, self.data_test = train_test_split(
            self.data,
            test_size=1 - self.train_size,
            random_state=self.seed,
            stratify=self.data["label"],
        )

        # Save split data
        self.data_train.to_csv(os.path.join(self.rdir, "data_train.csv"), index=False)
        self.data_test.to_csv(os.path.join(self.rdir, "data_test.csv"), index=False)

        print(f"Train set: {len(self.data_train)} samples")
        print(f"Test set: {len(self.data_test)} samples")

    def _save_files_info(self):
        """Save files information to JSON file."""
        files_info_path = os.path.join(self.rdir, "files_information.json")
        with open(files_info_path, "w") as f:
            json.dump(self.files_information, f, indent=2)

    def _download(self):
        """Download dataset files if they don't exist."""
        if not os.path.exists(self.rdir):
            os.makedirs(self.rdir)

        # Load URL list
        metadata_path = "pu_metadata.txt"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            for line in f:
                fault, name, url = line.strip().split()
                self.url_list.append((fault, name, url))

        # Download files
        for fault, name, url in self.url_list:
            self._download_single_dataset(fault, name, url)

    def _download_single_dataset(self, fault, name, url):
        """Download a single dataset if it doesn't exist."""
        fault_dir = os.path.join(self.rdir, fault)
        dataset_dir = os.path.join(fault_dir, name)

        # Check if download is needed
        must_download = False
        if not os.path.exists(fault_dir):
            must_download = True
            os.makedirs(fault_dir)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            must_download = True
        elif len(os.listdir(dataset_dir)) == 0:
            must_download = True

        if must_download:
            archive_path = os.path.join(self.rdir, f"{name}.rar")

            # Download if archive doesn't exist
            if not os.path.exists(archive_path):
                print(f"Downloading {name}...")
                urllib.urlretrieve(url, archive_path)
                print("Download complete.")

            # Extract archive
            print(f"Extracting {name}...")
            patoolib.extract_archive(archive_path, outdir=fault_dir)
            print("Extraction complete.")

            # Clean up archive
            os.remove(archive_path)

    def _get_files_info(self):
        """Scan directories and create files information structure."""
        print("Scanning files and creating information structure...")

        # Reset files information
        self.files_information = {
            "files": {"Healthy": [], "InnerRace": [], "OuterRace": []}
        }

        # Collect metadata
        metadata = {
            "radial_forces": set(),
            "load_torques": set(),
            "rot_speeds": set(),
            "codes": set(),
            "reps": set(),
        }

        # Scan all class directories
        for class_name in self.files_information["files"]:
            class_path = os.path.join(self.rdir, class_name)
            if not os.path.exists(class_path):
                continue

            for folder in os.listdir(class_path):
                folder_path = os.path.join(class_path, folder)
                if not os.path.isdir(folder_path):
                    continue

                for file in os.listdir(folder_path):
                    if file.endswith(".mat"):
                        # Parse filename
                        try:
                            rot_speed, load_torque, radial_force, code, rep = (
                                file.split(".")[0].split("_")
                            )

                            # Update metadata
                            metadata["radial_forces"].add(radial_force)
                            metadata["load_torques"].add(load_torque)
                            metadata["rot_speeds"].add(rot_speed)
                            metadata["codes"].add(code)
                            metadata["reps"].add(int(rep))

                            # Add file information
                            file_info = {
                                "rot_speed": rot_speed,
                                "load_torque": load_torque,
                                "radial_force": radial_force,
                                "code": code,
                                "rep": int(rep),
                                "file": os.path.join(class_path, folder, file),
                                "class_name": class_name,
                            }
                            self.files_information["files"][class_name].append(
                                file_info
                            )

                        except ValueError as e:
                            print(f"Warning: Could not parse filename {file}: {e}")

        # Convert sets to sorted lists
        self.files_information["metadata"] = {
            "radial_forces": sorted(list(metadata["radial_forces"])),
            "load_torques": sorted(list(metadata["load_torques"])),
            "rot_speeds": sorted(list(metadata["rot_speeds"])),
            "codes": sorted(list(metadata["codes"])),
            "reps": sorted(list(metadata["reps"])),
        }

        print(
            f"Found {sum(len(files) for files in self.files_information['files'].values())} files"
        )


def slicer(array, win, step, return_df=True):
    """
    Slice array into overlapping windows.

    Args:
        array: Input array to slice
        win: Window size
        step: Step size between windows
        return_df: If True, return pandas DataFrame, else numpy array

    Returns:
        DataFrame or numpy array of windowed data
    """
    N = array.shape[0]
    windows = []

    m = 0
    while m + win <= N:
        windows.append(array[m : m + win])
        m += step

    if return_df:
        return pd.DataFrame(windows)
    else:
        return np.array(windows)


class PU_Dataset(BearingDataset):
    def __init__(
        self,
        file_path,
        data_raw,
        window_size=2048,
        step_size=2048,
        device="cpu",
    ):
        self.file_path = file_path
        self.window_size = window_size
        self.step_size = step_size
        self.device = device

        self.total_windows = data_raw["window_counts"].sum()

        self.data = data_raw.copy()

        self.index_map = {}

        self.X = np.zeros((0, self.window_size, 1))
        self.y = np.zeros((0))

        self._load_data(data_raw)

    def _presist_data(self):
        np.save(self.file_path + "_X.npy", self.X.cpu().numpy())
        np.save(self.file_path + "_y.npy", self.y.cpu().numpy())

    def _load_data(self, data_raw: pd.DataFrame):
        if os.path.exists(self.file_path):
            self.X = torch.from_numpy(np.load(self.file_path + "_X.npy")).to(
                self.device
            )
            self.y = torch.from_numpy(np.load(self.file_path + "_y.npy")).to(
                self.device
            )
            print(f"Loaded data from {self.file_path}")
            return

        file_names = data_raw["file_name"].tolist()
        labels = data_raw["label"].tolist()
        counts = data_raw["window_counts"].tolist()

        # Determine dtype from the first file to preallocate correctly
        sample = np.load(file_names[0], mmap_mode="r")
        x_dtype = sample.dtype
        del sample

        # Preallocate arrays
        self.X = np.empty((self.total_windows, self.window_size, 1), dtype=x_dtype)
        self.y = np.empty((self.total_windows,), dtype=np.int64)

        # Compute write offsets for each file
        offsets = np.cumsum([0] + counts[:-1])

        def load_and_place(i: int):
            arr = np.load(file_names[i])
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            start = offsets[i]
            end = start + arr.shape[0]
            self.X[start:end, :, :] = arr
            self.y[start:end] = labels[i]

        max_workers = min(32, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(load_and_place, range(len(file_names))))

        self.X = torch.from_numpy(self.X).to(self.device)
        self.y = torch.from_numpy(self.y).to(self.device)
        self.X = self.X.permute(0, 2, 1)

        self._presist_data()

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        return self.X[idx].float(), self.y[idx].long()

    def window_size(self):
        return self.window_size

    def inputs(self):
        return self.X

    def targets(self):
        return self.y

    def classes(self):
        return [
            "Healthy",
            "InnerRace",
            "OuterRace",
        ]


if __name__ == "__main__":
    data_processor = PU_DatasetProcessor(
        rdir="./data/dataset/PU", seed=42, force_reload=False
    )

    # memory usage
    print(
        f"Memory usage: {data_processor.data.memory_usage().sum() / 1024 / 1024:.2f} MB"
    )

    train_dataset = PU_Dataset(
        "./data/dataset/PU/processed/data_train",
        data_processor.data_train,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    test_dataset = PU_Dataset(
        "./data/dataset/PU/processed/data_test",
        data_processor.data_test,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # memory usage
    print(f"Memory usage: {train_dataset.X.nbytes / 1024 / 1024:.2f} MB")
    print(f"Memory usage: {test_dataset.X.nbytes / 1024 / 1024:.2f} MB")
