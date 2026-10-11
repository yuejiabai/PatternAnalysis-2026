
import json
import random
import re
from pathlib import Path
from collections import defaultdict, Counter

from torchvision import datasets, transforms
from torch.utils.data import Subset


DATA_ROOT = Path("/home/groups/comp3710/ADNI/AD_NC")
METADATA_PATH = Path(
    "/home/groups/comp3710/ADNI/meta_data_with_label.json"
)

SEED = 42
VAL_RATIO = 0.2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def load_patient_map():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    patient_map = {}

    for image_id, record in metadata.items():
        raw_path = str(record.get("raw", ""))
        match = re.search(r"\d{3}_S_\d{4}", raw_path)

        if match:
            patient_map[str(image_id)] = match.group(0)

    return patient_map


def get_image_id(path):
    return Path(path).stem.split("_")[0]


def load_datasets():
    full_train = datasets.ImageFolder(
        DATA_ROOT / "train",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        DATA_ROOT / "test",
        transform=transform
    )

    assert full_train.class_to_idx == test_dataset.class_to_idx

    patient_map = load_patient_map()

    def get_patient(path):
        image_id = get_image_id(path)

        if image_id not in patient_map:
            raise ValueError(
                f"Missing patient ID for image {image_id}"
            )

        return patient_map[image_id]

    # Identify all patients in the official test set.
    test_patients = {
        get_patient(path)
        for path, _ in test_dataset.samples
    }

    # Exclude official test patients from training.
    patient_groups = defaultdict(list)
    excluded_images = 0
    excluded_patients = set()

    for index, (path, label) in enumerate(full_train.samples):
        patient_id = get_patient(path)

        if patient_id in test_patients:
            excluded_images += 1
            excluded_patients.add(patient_id)
            continue

        patient_groups[patient_id].append(index)

    # Stratify patient groups by their available class labels.
    strata = defaultdict(list)

    for patient_id, indices in patient_groups.items():
        labels = tuple(sorted({
            full_train.targets[i]
            for i in indices
        }))
        strata[labels].append(patient_id)

    rng = random.Random(SEED)
    train_patients = set()
    val_patients = set()

    for label_group, patients in sorted(strata.items()):
        patients = sorted(patients)
        rng.shuffle(patients)

        if len(patients) > 1:
            n_val = max(
                1,
                min(
                    len(patients) - 1,
                    round(len(patients) * VAL_RATIO)
                )
            )
        else:
            n_val = 0

        val_patients.update(patients[:n_val])
        train_patients.update(patients[n_val:])

    train_indices = []
    val_indices = []

    for patient_id, indices in patient_groups.items():
        if patient_id in val_patients:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    assert not (train_patients & val_patients)
    assert not (train_patients & test_patients)
    assert not (val_patients & test_patients)

    train_labels = Counter(
        full_train.targets[i] for i in train_indices
    )
    val_labels = Counter(
        full_train.targets[i] for i in val_indices
    )

    expected_classes = set(range(len(full_train.classes)))

    if set(train_labels) != expected_classes:
        raise ValueError("Training set is missing a class")

    if set(val_labels) != expected_classes:
        raise ValueError("Validation set is missing a class")

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)

    print("Excluded overlapping patients:",
          len(excluded_patients))
    print("Excluded training images:", excluded_images)
    print("Training patients:", len(train_patients))
    print("Validation patients:", len(val_patients))
    print("Testing patients:", len(test_patients))
    print("Train class counts:", dict(train_labels))
    print("Val class counts:", dict(val_labels))
    print("Patient-level split verified.")

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = load_datasets()

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Testing images:", len(test_dataset))
