import numpy as np
import pickle


class LabelNormalizer:
    """
    Label normalizer for downstream regressors.

    Supported modes:
        - zscore
        - robust
        - minmax
        - none
    """

    def __init__(self, mode: str = "zscore"):
        self.mode = mode
        self.mean = None
        self.std = None
        self.median = None
        self.iqr = None
        self.y_min = None
        self.y_max = None

    # -------------------------
    # Fit (train only)
    # -------------------------
    def fit(self, y: np.ndarray) -> None:
        y = y.astype(float)

        if self.mode == "zscore":
            self.mean = float(np.mean(y))
            self.std = float(np.std(y) + 1e-8)

        elif self.mode == "robust":
            self.median = float(np.median(y))
            q1 = np.percentile(y, 25)
            q3 = np.percentile(y, 75)
            self.iqr = float(q3 - q1 + 1e-8)

        elif self.mode == "minmax":
            self.y_min = float(np.min(y))
            self.y_max = float(np.max(y))

        elif self.mode == "none":
            pass

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    # -------------------------
    # Transform / Inverse
    # -------------------------
    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.mode == "zscore":
            return (y - self.mean) / self.std

        elif self.mode == "robust":
            return (y - self.median) / self.iqr

        elif self.mode == "minmax":
            return (y - self.y_min) / (self.y_max - self.y_min + 1e-8)

        elif self.mode == "none":
            return y

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.mode == "zscore":
            return z * self.std + self.mean

        elif self.mode == "robust":
            return z * self.iqr + self.median

        elif self.mode == "minmax":
            return z * (self.y_max - self.y_min) + self.y_min

        elif self.mode == "none":
            return z

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = LabelNormalizer(d["mode"])
        obj.__dict__.update(d)
        return obj
