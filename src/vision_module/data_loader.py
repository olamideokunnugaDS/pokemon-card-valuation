"""Dataset class for PSA card image loading."""

from pathlib import Path
from typing import Callable, Dict, Optional

from PIL import Image
from torch.utils.data import Dataset


class CardDataset(Dataset):
    """PSA card image dataset.

    Supports two target modes, corresponding to the two training stages:
      - 'card_name': 7-class identity classification (Stage 1)
      - 'grade': 3-class condition classification (Stage 2)
    """

    def __init__(
        self,
        dataframe,
        target_col: str,
        label_map: Dict,
        project_root: Path,
        transform: Optional[Callable] = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.target_col = target_col
        self.label_map = label_map
        self.project_root = Path(project_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.project_root / row["local_image_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_map[row[self.target_col]]

        return image, label, idx  # idx is kept for embedding extraction
