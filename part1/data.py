from tokenizer import tokenize_text
import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub

SPLITS = ("train", "dev", "test")


def load_data(path, name, drop_multi_labels=False):
    df = pd.read_csv(
        os.path.join(path, f"{name}.tsv"),
        sep="\t",
        header=None,
        names=["text", "labels", "id"],
    )
    df["labels"] = df["labels"].str.split(",").apply(lambda ls: [int(l) for l in ls])
    if drop_multi_labels:
        df = df[df["labels"].str.len() == 1].reset_index(drop=True)
        df["label"] = df["labels"].str[0]
    return df


class GoEmotions:
    """GoEmotions splits and statistics

    drop_multi_labels: if True -> drop comments that have more than one label
        and add a "label" column with the single int label
    merge_train_dev: if True -> concatenate dev into train and remove the dev split
        (useful when doing our own cross-validation splits on the training data)
    """
    def __init__(self, drop_multi_labels=False, merge_train_dev=False):
        self.path = os.path.join(
            kagglehub.dataset_download("debarshichanda/goemotions"), "data"
        )
        self.drop_multi_labels = drop_multi_labels
        self.splits = {
            name: load_data(self.path, name, drop_multi_labels) for name in SPLITS
        }
        self.merge_train_dev = merge_train_dev
        if merge_train_dev:
            self.splits["train"] = pd.concat(
                [self.splits["train"], self.splits.pop("dev")], ignore_index=True
            )
        with open(os.path.join(self.path, "emotions.txt")) as f:
            self.emotions = f.read().split()

    @property
    def train(self):
        return self.splits["train"]

    @property
    def dev(self):
        if self.merge_train_dev:
            raise AttributeError("dev was merged into train (merge_train_dev=True)")
        return self.splits["dev"]

    @property
    def test(self):
        return self.splits["test"]

    @property
    def summary(self):
        """Statistics per split"""
        rows = {}
        for name, df in self.splits.items():
            n_tokens = tokenize_text(df)["text_tokens"].str.len()
            rows[name] = {
                "Total comments": len(df),
                "Average n tokens": n_tokens.mean(),
                "Median n tokens": n_tokens.median(),
                "Max tokens": n_tokens.max(),
            }
        return pd.DataFrame(rows).T.round(2)

    @property
    def label_counts(self):
        """Number of comments per label per split (sorted by train)"""
        counts = (
            pd.DataFrame(
                {
                    name: pd.Series(Counter(l for ls in df["labels"] for l in ls))
                    for name, df in self.splits.items()
                }
            )
            .fillna(0)
            .astype(int)
        )
        counts.index = [self.emotions[i] for i in counts.index]
        return counts.sort_values("train", ascending=False)

    @property
    def label_percent(self):
        """Percentage of comments for each label per split"""
        sizes = pd.Series({name: len(df) for name, df in self.splits.items()})
        return (self.label_counts / sizes * 100).round(2)

    def plot_label_dist(self, split="train"):
        """Sorted horizontal bar chart of percentage of comments per label"""
        percentage = self.label_percent[split].sort_values()
        _, ax = plt.subplots(figsize=(8, 8))
        ax.barh(percentage.index, percentage.values, height=0.8)
        ax.set_xlabel(f"Percentage of {split} examples")
        ax.set_title(f"Label distribution in GoEmotions ({split})")
        ax.margins(y=0.01)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        for name in percentage.index:
            ax.text(
                percentage[name],
                name,
                f"{percentage[name]:.1f}%",
            )
        plt.plot()
