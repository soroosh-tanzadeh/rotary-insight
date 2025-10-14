from scipy.io import loadmat
from .dataset import BearingDataset
import numpy as np
import os, re
import errno
import urllib.request as urllib
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch

# 12DriveEndFault and 48DriveEndFault are same, difference is in sample rate
exps_idx = {"12DriveEndFault": 0, "48DriveEndFault": 0, "12FanEndFault": 0}

# https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures
faults_idx = {
    "Normal": 0,
    "0.007-Ball": 1,
    "0.014-Ball": 2,
    "0.021-Ball": 3,
    "0.007-InnerRace": 4,
    "0.014-InnerRace": 5,
    "0.021-InnerRace": 6,
    "0.007-OuterRace3": 7,
    "0.007-OuterRace6": 7,
    "0.007-OuterRace12": 7,
    "0.014-OuterRace3": 8,
    "0.014-OuterRace12": 8,
    "0.014-OuterRace6": 8,
    "0.021-OuterRace6": 9,
    "0.021-OuterRace3": 9,
    "0.021-OuterRace12": 9,
}


FOLDER = "cwru-dataset"


def get_class_name(label):
    if label == 0:
        return "Normal"
    for fault in faults_idx:
        if fault == 0:
            continue
        for exp in exps_idx:
            if label == exps_idx[exp] + faults_idx[fault]:
                ## if fault name end with a number like OuterRace12, we need to remove the number
                return (
                    exp
                    + "_"
                    + fault.replace("OuterRace3", "OuterRace")
                    .replace("OuterRace6", "OuterRace")
                    .replace("OuterRace12", "OuterRace")
                )
    return "Unknown"


def filter_key(keys):
    fkeys = []
    for key in keys:
        matchObj = re.match(r"(.*)FE_time", key, re.M | re.I)
        if matchObj:
            fkeys.append(matchObj.group(1))
    return fkeys[0] + "DE_time", fkeys[0] + "FE_time"


def get_class(exp, fault):
    if fault == "Normal":
        return 0
    return exps_idx[exp] + faults_idx[fault]


class CWRU(BearingDataset):
    def __init__(
        self,
        rdir,
        exps=["12DriveEndFault", "48DriveEndFault", "12FanEndFault"],
        rpms=["1797", "1772", "1750", "1730"],
        window_size=2048,
        seed=None,
    ):
        super().__init__()
        self.seed = seed
        for exp in exps:
            if exp not in ("12DriveEndFault", "12FanEndFault", "48DriveEndFault"):
                print("wrong experiment name: {}".format(exp))
                return
        for rpm in rpms:
            if rpm not in ("1797", "1772", "1750", "1730"):
                print("wrong rpm value: {}".format(rpm))
                return

        fmeta = os.path.join(os.path.dirname("__file__"), "metadata.txt")
        all_lines = open(fmeta).readlines()
        infos = []
        for line in all_lines:
            l = line.split()
            if (l[0] in exps or l[0] == "NormalBaseline") and l[1] in rpms:
                if (
                    "Normal" in l[2]
                    or "0.007" in l[2]
                    or "0.014" in l[2]
                    or "0.021" in l[2]
                ):
                    if faults_idx.get(l[2], -1) != -1:
                        infos.append(l)

        self.window_size = window_size  # sequence length
        self.rdir = rdir
        infos = sorted(infos, key=lambda line: get_class(line[0], line[2]))

        self._load_data(rdir, infos)
        # shuffle training and test arrays
        self._shuffle()
        self.all_labels = tuple(
            ((line[0] + line[2]), get_class(line[0], line[2])) for line in infos
        )
        self.classes = sorted(list(set(self.all_labels)), key=lambda label: label[1])
        self.num_classes = len(self.classes)  # number of classes

    def window_size(self):
        self.window_size

    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print("can't create directory '{}''".format(path))
                exit(1)

    def _download(self, fpath, link, retryCount=1):
        print(link + " Downloading to: '{}'".format(fpath))
        try:
            urllib.urlretrieve(link, fpath)
        except:
            if retryCount > 0:
                os.remove(fpath)
                self._download(fpath, link, retryCount - 1)
            else:
                print("can't download file '{}'".format(link))
                exit(1)

    def _load_data(self, rdir, infos):
        if self._load_preprocessed():
            return

        X = np.zeros((0, self.window_size, 2))
        y = []

        for idx, info in enumerate(infos):
            # Directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + ".mat")

            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip("\n"), 10)

            mat_dict = loadmat(fpath)
            key1, key2 = filter_key(mat_dict.keys())
            time_series = np.hstack((mat_dict[key1], mat_dict[key2]))

            n_samples = (
                len(time_series) - (len(time_series) % self.window_size)
            ) / self.window_size

            # Process training data
            clips = np.zeros((0, 2))
            for cut in shuffle(list(range(int(n_samples)))):
                clips = np.vstack(
                    (
                        clips,
                        time_series[
                            cut * self.window_size : (cut + 1) * self.window_size
                        ],
                    )
                )
            clips = clips.reshape(-1, self.window_size, 2)
            X = np.vstack((X, clips))
            y.extend([get_class(info[0], info[2]) for _ in range(len(clips))])

        X = X.reshape(-1, 2, self.window_size)

        self.X = X
        self.y = np.array(y)

        self.presist()

    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _load_preprocessed(self):
        if not os.path.exists(
            os.path.join(self.rdir, FOLDER, "X.npy")
        ) or not os.path.exists(os.path.join(self.rdir, FOLDER, "y.npy")):
            return False
        print("Loading preprocessed data...")
        self.X = np.load(os.path.join(self.rdir, FOLDER, "X.npy"))
        self.y = np.load(os.path.join(self.rdir, FOLDER, "y.npy"))
        return True

    def presist(self):
        self.ensure_dir(os.path.join(self.rdir, FOLDER))
        np.save(os.path.join(self.rdir, "test", "X.npy"), self.X)
        np.save(os.path.join(self.rdir, "test", "y.npy"), self.y)

    def _shuffle(self):
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = np.array(tuple(self.y[i] for i in indices))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            torch.tensor(self.X[index]).float(),
            torch.tensor(self.y[index]).long(),
        )

    def inputs(self):
        return self.X

    def targets(self):
        return self.y

    def classes(self):
        return [
            "Normal",
            "0.007-Ball",
            "0.014-Ball",
            "0.021-Ball",
            "0.007-InnerRace",
            "0.014-InnerRace",
            "0.021-InnerRace",
            "0.007-OuterRace",
            "0.014-OuterRace",
            "0.021-OuterRace",
        ]
