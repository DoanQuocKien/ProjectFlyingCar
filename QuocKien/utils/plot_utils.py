"""Plotting helpers for training history CSVs.

Functions:
- plot_loss_curve(csv_path or DataFrame or list): plots training and validation loss.
- plot_accuracy_curve(csv_path or DataFrame or list): plots validation (and training if present) accuracy.

Designed to be imported into notebooks: from utils.plot_utils import plot_loss_curve, plot_accuracy_curve
"""
from typing import Union, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PathLike = Union[str, pd.DataFrame]


def _load_history(item: PathLike) -> pd.DataFrame:
    if isinstance(item, pd.DataFrame):
        return item.copy()
    return pd.read_csv(item)


def _ensure_epochs(df: pd.DataFrame) -> pd.DataFrame:
    if 'epoch' in df.columns:
        return df
    # try to infer epoch index from row numbers
    df = df.reset_index().rename(columns={'index': 'epoch'})
    df['epoch'] = df['epoch'] + 1
    return df


def plot_loss_curve(sources: Union[PathLike, List[PathLike]], title: str = None, ax=None):
    """Plot training and validation loss curves.

    Sources may be a single CSV path/DataFrame or a list of them. CSVs commonly contain
    columns like 'train_loss', 'loss', 'val_loss', or similar. The function heuristically
    finds sensible columns to plot.
    """
    if not isinstance(sources, list):
        sources = [sources]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    for src in sources:
        df = _load_history(src)
        df = _ensure_epochs(df)

        # find loss columns
        train_loss_cols = [c for c in df.columns if 'train' in c.lower() and 'loss' in c.lower()]
        val_loss_cols = [c for c in df.columns if ('val' in c.lower() or 'validation' in c.lower()) and 'loss' in c.lower()]
        # fallbacks
        if not train_loss_cols and 'loss' in df.columns:
            train_loss_cols = ['loss']
        if not val_loss_cols:
            # sometimes 'val_loss' absent; try 'val_loss' variations
            possible = [c for c in df.columns if 'val' in c.lower() and 'loss' in c.lower()]
            val_loss_cols = possible

        label_prefix = getattr(src, 'name', str(src))

        if train_loss_cols:
            for c in train_loss_cols:
                ax.plot(df['epoch'], df[c], label=f'{label_prefix} {c}')
        if val_loss_cols:
            for c in val_loss_cols:
                ax.plot(df['epoch'], df[c], '--', label=f'{label_prefix} {c}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Loss curves')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')
    plt.tight_layout()
    plt.show()
    return fig


def plot_accuracy_curve(sources: Union[PathLike, List[PathLike]], title: str = None, ax=None):
    """Plot accuracy curves (validation accuracy preferred).

    Looks for columns such as 'val_accuracy', 'val_acc', 'accuracy', 'acc', 'train_accuracy'.
    """
    if not isinstance(sources, list):
        sources = [sources]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    for src in sources:
        df = _load_history(src)
        df = _ensure_epochs(df)

        acc_cols = [c for c in df.columns if 'acc' in c.lower() or 'accuracy' in c.lower()]
        # prefer validation columns
        val_cols = [c for c in acc_cols if 'val' in c.lower() or 'validation' in c.lower()]
        train_cols = [c for c in acc_cols if c not in val_cols]

        label_prefix = getattr(src, 'name', str(src))

        if val_cols:
            for c in val_cols:
                ax.plot(df['epoch'], df[c], label=f'{label_prefix} {c}')
        if train_cols:
            for c in train_cols:
                ax.plot(df['epoch'], df[c], '--', label=f'{label_prefix} {c}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Accuracy curves')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == '__main__':
    print('plot_utils: helper functions to plot training CSV histories')
