import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIGURATION ----------------

INPUT_CSV = "../data/review_clean.csv"
OUTPUT_DIR = "../data"
MAX_LENGTH = 256
TEST_SIZE = 0.2
RANDOM_STATE = 42
AUGMENTATION_TARGET_LABEL = 2  # Index 2 corresponds to 3-Star reviews (the target ceiling)

# ---------------- AUGMENTATION SETUP (Synonym Replacement Only) ----------------

# A. Synonym Replacement (SR)
aug_sr = naw.SynonymAug(
    aug_src='wordnet',
    aug_max=3,  # Max 3 synonym replacements per sentence
    aug_p=0.2  # Replace about 20% of words in the sentence
)


def prepare_and_save_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load and Prepare Data
    df = pd.read_csv(INPUT_CSV, encoding="latin1")
    df["Rating"] = df["Rating"].astype(int)
    df["label"] = df["Rating"] - 1

    texts = df["Review_Text_bert"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels,
    )

    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    val_df = pd.DataFrame({'text': X_val, 'label': y_val})

    # Get the target count (3-Star ceiling)
    train_counts = train_df['label'].value_counts().sort_index()
    target_count = train_counts.loc[AUGMENTATION_TARGET_LABEL]

    print(f"\n--- Initial Training Distribution ---")
    print(train_counts)
    print(f"\nTarget count for minority classes: {target_count} (3-Star ceiling)")

    X_aug_final_combined = train_df['text'].tolist()
    y_aug_final_combined = train_df['label'].tolist()

    # 1. Iterate over minority classes (1-Star/Label 0 and 2-Star/Label 1)
    for label_idx in [0, 1]:
        current_count = train_counts.loc[label_idx]
        minority_texts = train_df[train_df['label'] == label_idx]['text'].tolist()
        num_needed = target_count - current_count

        if num_needed <= 0:
            continue

        print(f"\nAugmenting {label_idx + 1}-Star reviews (Need: {num_needed} samples via Synonym Replacement)")

        X_new_augmented = []

        # Calculate how many times we need to cycle through the original data to hit the target
        # Use ceiling division to ensure we get enough repetitions
        repeat_factor = (num_needed + current_count - 1) // current_count

        # Use a list of indices to cycle through the data multiple times
        indices_to_augment = np.tile(np.arange(current_count), repeat_factor)

        # Shuffle indices to ensure diversity in augmentation order
        np.random.shuffle(indices_to_augment)

        # Take only the exact number of samples needed for augmentation
        indices_to_augment = indices_to_augment[:num_needed]

        for i in tqdm(indices_to_augment, total=num_needed, desc=f"Generating {num_needed} samples"):
            original_text = minority_texts[i]

            try:
                # Generate exactly 1 augmented variant per iteration
                new_text = aug_sr.augment(original_text, n=1)[0]
                X_new_augmented.append(new_text)
            except Exception:
                # If augmentation fails for this sample, try to generate a duplicate instead
                X_new_augmented.append(original_text)

                # Combine the new augmented data with the main lists
        y_new_augmented = [label_idx] * len(X_new_augmented)
        X_aug_final_combined.extend(X_new_augmented)
        y_aug_final_combined.extend(y_new_augmented)

    # 2. Final Training Dataset Assembly
    final_train_df = pd.DataFrame({'text': X_aug_final_combined, 'label': y_aug_final_combined})


    print(f"\n--- Final Training Distribution (Balanced) ---")
    print(final_train_df['label'].value_counts().sort_index())
    print(f"Total Final Training Size: {len(final_train_df)}")

    # 3. Save to CSV Files
    final_train_df.to_csv(os.path.join(OUTPUT_DIR, "train_augmented.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)

    print("\nData preparation complete!")
    print(f"Augmented training data saved to: {os.path.join(OUTPUT_DIR, 'train_augmented.csv')}")
    print(f"Original validation data saved to: {os.path.join(OUTPUT_DIR, 'val.csv')}")


if __name__ == "__main__":
    prepare_and_save_data()
