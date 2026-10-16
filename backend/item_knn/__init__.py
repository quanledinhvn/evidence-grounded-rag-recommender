"""Standalone item-based collaborative filtering."""

from backend.item_knn.bundle import ItemKNNBundle, ItemKNNConfig, train_item_knn_bundle


__all__ = ["ItemKNNBundle", "ItemKNNConfig", "train_item_knn_bundle"]
