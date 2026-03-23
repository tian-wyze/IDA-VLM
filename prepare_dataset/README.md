# Prepare the dataset

1. [Not used] Similarity-based gallery

Using the `wyze_person_v2_cross_clothes` for example, we first split the dataset into train and test sets, ensuring no identity overlapping.

```bash
cd prepare_dataset/
python split_train_test.py

# calculate the embeddings and cosine simialrities between query and gallery images
python calculate_embed_sim.py

# format the test file, where k gallery images (> threshold) are defined for each query image
python prepare_gallery.py test 5 0.5

# visualize some gallery examples for sanity check
python visualize_gallery.py
```

2. Household-based gallery

```bash
python prepare_gallery.py

python visualize_case.py
```