from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Sampler

from data.dataset import PatchDataset, BinarizedMemoryDataset


class BalancedBatchSampler(Sampler[list]):
    def __init__(
        self,
        labels: list,
        batch_size: int,
        samples_per_class: int,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive")
        if batch_size % samples_per_class != 0:
            raise ValueError("batch_size must be divisible by samples_per_class")
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last
        self._labels = labels
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            self.label_to_indices.setdefault(label, []).append(idx)
        self.num_classes = batch_size // samples_per_class
        self.classes = list(self.label_to_indices.keys())

    def __iter__(self):
        indices = []
        if not self.classes:
            return iter(indices)
        for _ in range(len(self)):
            selected = torch.randperm(len(self.classes))[: self.num_classes].tolist()
            for class_idx in selected:
                label = self.classes[class_idx]
                label_indices = self.label_to_indices[label]
                choices = torch.randint(0, len(label_indices), (self.samples_per_class,))
                indices.extend([label_indices[i] for i in choices.tolist()])
            if len(indices) == self.batch_size:
                yield indices
                indices = []
        if indices and not self.drop_last:
            yield indices

    def __len__(self) -> int:
        if not self.classes:
            return 0
        total_samples = len(self._labels)
        per_batch = self.batch_size
        return total_samples // per_batch


def summarize_dataset(name: str, dataset: PatchDataset) -> None:
    labels = [label for _, _, label in dataset.samples]
    counts = Counter(labels)
    num_classes = len(dataset.class_to_idx) if dataset.class_to_idx else len(counts)
    total_samples = len(dataset)
    if counts:
        min_count = min(counts.values())
        max_count = max(counts.values())
        mean_count = total_samples / max(1, num_classes)
    else:
        min_count = max_count = mean_count = 0.0
    print(
        f"{name} stats | samples {total_samples} | "
        f"classes {num_classes} | "
        f"samples/class min {min_count} max {max_count} mean {mean_count:.2f}"
    )


def _subset_dataset_by_writer_fraction(dataset: PatchDataset, fraction: float, seed: int) -> None:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("train_writer_fraction must be in the interval (0, 1].")
    if fraction >= 1.0 or not dataset.samples:
        return

    labels = sorted({label for _, _, label in dataset.samples})
    if not labels:
        return

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    shuffled = torch.randperm(len(labels), generator=generator).tolist()
    keep_count = max(1, round(len(labels) * fraction))
    keep_old_labels = {labels[idx] for idx in shuffled[:keep_count]}

    if dataset.class_to_idx:
        idx_to_name = {idx: name for name, idx in dataset.class_to_idx.items()}
        kept_names = sorted(idx_to_name[label] for label in keep_old_labels)
        remap = {
            dataset.class_to_idx[name]: new_idx
            for new_idx, name in enumerate(kept_names)
        }
        dataset.class_to_idx = {name: new_idx for new_idx, name in enumerate(kept_names)}
    else:
        kept_labels = sorted(keep_old_labels)
        remap = {old_label: new_idx for new_idx, old_label in enumerate(kept_labels)}

    dataset.samples = [
        (image_path, rel_path, remap[label])
        for image_path, rel_path, label in dataset.samples
        if label in remap
    ]
    print(
        "Train writer subset | "
        f"fraction {fraction:.4g} | writers {len(remap)}/{len(labels)} | "
        f"samples {len(dataset.samples)} | seed {seed}"
    )


def build_datasets(
    data_cfg,
    model_cfg,
    train_transform,
    page_transform,
    eval_only: bool,
):
    train_dataset = None
    val_dataset = None
    val_class_split_char = data_cfg.class_split_char
    val_class_id_index = data_cfg.class_id_index
    if data_cfg.val_class_split_char:
        val_class_split_char = data_cfg.val_class_split_char
    if data_cfg.val_class_id_index >= 0:
        val_class_id_index = data_cfg.val_class_id_index
    dataset_cls_train = PatchDataset
    dataset_cls_val = PatchDataset
    if data_cfg.binarized_in_memory and not data_cfg.train_use_precomputed_patches:
        dataset_cls_train = BinarizedMemoryDataset
    if data_cfg.full_image_input and data_cfg.train_use_precomputed_patches:
        raise ValueError("full_image_input cannot be combined with train_use_precomputed_patches.")
    if data_cfg.full_image_input and data_cfg.val_use_precomputed_patches:
        raise ValueError("full_image_input cannot be combined with val_use_precomputed_patches.")
    dataset_kwargs = {"debug": data_cfg.debug_dataset}
    train_dataset_kwargs = dict(dataset_kwargs)
    if dataset_cls_train is BinarizedMemoryDataset:
        train_dataset_kwargs["cache_on_gpu"] = data_cfg.binarized_cache_on_gpu
        train_dataset_kwargs["cache_workers"] = data_cfg.binarized_cache_workers
    train_page_transform = page_transform
    # Match packed-patch behavior for in-memory mode: keep patch-level augmentation,
    # skip expensive page-level random crop on full images.
    if dataset_cls_train is BinarizedMemoryDataset and data_cfg.augment_patches:
        train_page_transform = None

    if data_cfg.train_csv and data_cfg.val_csv:
        train_common_kwargs = dict(
            csv_path=data_cfg.train_csv,
            image_root=data_cfg.train_root,
            mask_root=data_cfg.train_mask_root,
            num_patches=model_cfg.num_patches,
            patch_size=model_cfg.patch_size,
            transform=train_transform,
            page_transform=train_page_transform,
            augment_patches=data_cfg.augment_patches,
            augment_otsu=data_cfg.augment_otsu,
            class_split_char=data_cfg.class_split_char,
            class_id_index=data_cfg.class_id_index,
            class_id_use_folder=data_cfg.class_id_use_folder,
            image_ext=data_cfg.image_ext,
            mask_ext=data_cfg.mask_ext,
            full_image_input=data_cfg.full_image_input,
            full_image_height=data_cfg.full_image_height,
            full_image_width=data_cfg.full_image_width,
            full_image_pad_to_size=data_cfg.full_image_pad_to_size,
            full_image_resize_longest_side_first=data_cfg.full_image_resize_longest_side_first,
            sequence_patch_augment_in_dataset=data_cfg.sequence_patch_augment_in_dataset,
            sequence_patch_split_per_sample=data_cfg.sequence_patch_split_per_sample,
            sequence_patch_crop_size=data_cfg.sequence_patch_crop_size,
            strong_patch_augment=data_cfg.strong_patch_augment,
            **train_dataset_kwargs,
        )
        if dataset_cls_train is PatchDataset:
            train_common_kwargs.update(
                {
                    "precomputed_patches_root": data_cfg.train_patches_root
                    if data_cfg.train_use_precomputed_patches
                    else "",
                    "patch_ext": data_cfg.patch_ext,
                    "debug_first_batch": data_cfg.debug_first_batch,
                    "cache_packed_patches_in_memory": data_cfg.cache_packed_patches_in_memory,
                }
            )
        train_dataset = dataset_cls_train(**train_common_kwargs)
        _subset_dataset_by_writer_fraction(
            train_dataset,
            data_cfg.train_writer_fraction,
            data_cfg.train_writer_subset_seed,
        )
        val_dataset = dataset_cls_val(
            csv_path=data_cfg.val_csv,
            image_root=data_cfg.val_root,
            mask_root=data_cfg.val_mask_root,
            num_patches=data_cfg.num_patches_eval,
            patch_size=model_cfg.patch_size,
            transform=None,
            augment_otsu=data_cfg.augment_otsu,
            val_gray=data_cfg.val_gray,
            precomputed_patches_root=data_cfg.val_patches_root if data_cfg.val_use_precomputed_patches else "",
            class_split_char=val_class_split_char,
            class_id_index=val_class_id_index,
                class_id_use_folder=data_cfg.class_id_use_folder,
                image_ext=data_cfg.image_ext,
                mask_ext=data_cfg.mask_ext,
                full_image_input=data_cfg.full_image_input,
                full_image_height=data_cfg.full_image_height,
                full_image_width=data_cfg.full_image_width,
                full_image_pad_to_size=data_cfg.full_image_pad_to_size,
                full_image_resize_longest_side_first=data_cfg.full_image_resize_longest_side_first,
                sequence_patch_augment_in_dataset=False,
                sequence_patch_split_per_sample=1,
                sequence_patch_crop_size=data_cfg.sequence_patch_crop_size,
                strong_patch_augment=False,
                patch_ext=data_cfg.patch_ext,
                cache_packed_patches_in_memory=data_cfg.cache_packed_patches_in_memory,
                **{**dataset_kwargs, "debug_first_batch": data_cfg.debug_first_batch},
            )
    elif data_cfg.train_root and data_cfg.val_root:
        train_common_kwargs = dict(
            csv_path=None,
            image_root=data_cfg.train_root,
            mask_root=data_cfg.train_mask_root,
            num_patches=model_cfg.num_patches,
            patch_size=model_cfg.patch_size,
            transform=train_transform,
            page_transform=train_page_transform,
            augment_patches=data_cfg.augment_patches,
            augment_otsu=data_cfg.augment_otsu,
            class_split_char=data_cfg.class_split_char,
            class_id_index=data_cfg.class_id_index,
            class_id_use_folder=data_cfg.class_id_use_folder,
            image_ext=data_cfg.image_ext,
            mask_ext=data_cfg.mask_ext,
            full_image_input=data_cfg.full_image_input,
            full_image_height=data_cfg.full_image_height,
            full_image_width=data_cfg.full_image_width,
            full_image_pad_to_size=data_cfg.full_image_pad_to_size,
            full_image_resize_longest_side_first=data_cfg.full_image_resize_longest_side_first,
            sequence_patch_augment_in_dataset=data_cfg.sequence_patch_augment_in_dataset,
            sequence_patch_split_per_sample=data_cfg.sequence_patch_split_per_sample,
            sequence_patch_crop_size=data_cfg.sequence_patch_crop_size,
            strong_patch_augment=data_cfg.strong_patch_augment,
            **train_dataset_kwargs,
        )
        if dataset_cls_train is PatchDataset:
            train_common_kwargs.update(
                {
                    "precomputed_patches_root": data_cfg.train_patches_root
                    if data_cfg.train_use_precomputed_patches
                    else "",
                    "patch_ext": data_cfg.patch_ext,
                    "debug_first_batch": data_cfg.debug_first_batch,
                    "cache_packed_patches_in_memory": data_cfg.cache_packed_patches_in_memory,
                }
            )
        train_dataset = dataset_cls_train(**train_common_kwargs)
        _subset_dataset_by_writer_fraction(
            train_dataset,
            data_cfg.train_writer_fraction,
            data_cfg.train_writer_subset_seed,
        )
        val_dataset = dataset_cls_val(
            csv_path=None,
            image_root=data_cfg.val_root,
            mask_root=data_cfg.val_mask_root,
            num_patches=data_cfg.num_patches_eval,
            patch_size=model_cfg.patch_size,
            transform=None,
            augment_otsu=data_cfg.augment_otsu,
            val_gray=data_cfg.val_gray,
            precomputed_patches_root=data_cfg.val_patches_root if data_cfg.val_use_precomputed_patches else "",
            class_split_char=val_class_split_char,
            class_id_index=val_class_id_index,
                class_id_use_folder=data_cfg.class_id_use_folder,
                image_ext=data_cfg.image_ext,
                mask_ext=data_cfg.mask_ext,
                full_image_input=data_cfg.full_image_input,
                full_image_height=data_cfg.full_image_height,
                full_image_width=data_cfg.full_image_width,
                full_image_pad_to_size=data_cfg.full_image_pad_to_size,
                full_image_resize_longest_side_first=data_cfg.full_image_resize_longest_side_first,
                sequence_patch_augment_in_dataset=False,
                sequence_patch_split_per_sample=1,
                sequence_patch_crop_size=data_cfg.sequence_patch_crop_size,
                strong_patch_augment=False,
                patch_ext=data_cfg.patch_ext,
                cache_packed_patches_in_memory=data_cfg.cache_packed_patches_in_memory,
                **{**dataset_kwargs, "debug_first_batch": data_cfg.debug_first_batch},
            )
    elif eval_only and data_cfg.val_root:
        val_dataset = dataset_cls_val(
            csv_path=None,
            image_root=data_cfg.val_root,
            mask_root=data_cfg.val_mask_root,
            num_patches=data_cfg.num_patches_eval,
            patch_size=model_cfg.patch_size,
            transform=None,
            augment_otsu=data_cfg.augment_otsu,
            val_gray=data_cfg.val_gray,
            precomputed_patches_root=data_cfg.val_patches_root if data_cfg.val_use_precomputed_patches else "",
            class_split_char=val_class_split_char,
            class_id_index=val_class_id_index,
            class_id_use_folder=data_cfg.class_id_use_folder,
            image_ext=data_cfg.image_ext,
            mask_ext=data_cfg.mask_ext,
            full_image_input=data_cfg.full_image_input,
            full_image_height=data_cfg.full_image_height,
            full_image_width=data_cfg.full_image_width,
            full_image_pad_to_size=data_cfg.full_image_pad_to_size,
            full_image_resize_longest_side_first=data_cfg.full_image_resize_longest_side_first,
            sequence_patch_augment_in_dataset=False,
            sequence_patch_split_per_sample=1,
            sequence_patch_crop_size=data_cfg.sequence_patch_crop_size,
            strong_patch_augment=False,
            patch_ext=data_cfg.patch_ext,
            cache_packed_patches_in_memory=data_cfg.cache_packed_patches_in_memory,
            **{**dataset_kwargs, "debug_first_batch": data_cfg.debug_first_batch},
        )
    elif eval_only and data_cfg.val_csv:
        val_dataset = dataset_cls_val(
            csv_path=data_cfg.val_csv,
            image_root=data_cfg.val_root,
            mask_root=data_cfg.val_mask_root,
            num_patches=data_cfg.num_patches_eval,
            patch_size=model_cfg.patch_size,
            transform=None,
            augment_otsu=data_cfg.augment_otsu,
            val_gray=data_cfg.val_gray,
            precomputed_patches_root=data_cfg.val_patches_root if data_cfg.val_use_precomputed_patches else "",
            class_split_char=val_class_split_char,
            class_id_index=val_class_id_index,
            class_id_use_folder=data_cfg.class_id_use_folder,
            image_ext=data_cfg.image_ext,
            mask_ext=data_cfg.mask_ext,
            full_image_input=data_cfg.full_image_input,
            full_image_height=data_cfg.full_image_height,
            full_image_width=data_cfg.full_image_width,
            full_image_pad_to_size=data_cfg.full_image_pad_to_size,
            full_image_resize_longest_side_first=data_cfg.full_image_resize_longest_side_first,
            sequence_patch_augment_in_dataset=False,
            sequence_patch_split_per_sample=1,
            sequence_patch_crop_size=data_cfg.sequence_patch_crop_size,
            strong_patch_augment=False,
            patch_ext=data_cfg.patch_ext,
            cache_packed_patches_in_memory=data_cfg.cache_packed_patches_in_memory,
            **{**dataset_kwargs, "debug_first_batch": data_cfg.debug_first_batch},
        )
    return train_dataset, val_dataset


def build_loaders(
    data_cfg,
    train_dataset: Optional[PatchDataset],
    val_dataset: PatchDataset,
) -> Tuple[Optional[DataLoader], DataLoader]:
    train_loader = None
    if (
        isinstance(train_dataset, BinarizedMemoryDataset)
        and getattr(train_dataset, "cache_on_gpu", False)
        and data_cfg.num_workers > 0
    ):
        print("Warning: binarized_cache_on_gpu requires num_workers=0; forcing num_workers=0.")
        data_cfg = data_cfg.__class__(**{**data_cfg.__dict__, "num_workers": 0})
    if train_dataset is not None:
        loader_kwargs = {
            "num_workers": data_cfg.num_workers,
            "pin_memory": data_cfg.pin_memory,
        }
        if data_cfg.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4
        if data_cfg.balanced_batch:
            labels = [label for _, _, label in train_dataset.samples]
            batch_sampler = BalancedBatchSampler(
                labels=labels,
                batch_size=data_cfg.batch_size,
                samples_per_class=data_cfg.samples_per_class,
                drop_last=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=data_cfg.batch_size,
                shuffle=True,
                **loader_kwargs,
            )
    eval_batch_size = data_cfg.batch_size
    if data_cfg.eval_batch_size and data_cfg.eval_batch_size > 0:
        eval_batch_size = data_cfg.eval_batch_size
    val_loader_kwargs = {
        "num_workers": data_cfg.num_workers,
        "pin_memory": data_cfg.pin_memory,
    }
    if data_cfg.num_workers > 0:
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = 4
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **val_loader_kwargs,
    )
    return train_loader, val_loader
