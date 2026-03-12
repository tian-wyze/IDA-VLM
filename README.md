# IDA-VLM
Tian's work on IDA-VLM


## Updates

Create WYZE in-house IDA-VLM benchmark.


## Benchmarks

WYZE in-house data

| Dataset | Identities (total) | Query Images | Gallery Images | Train Split (identity/imgs) | Test Split (identity/imgs)
|---------|-----------|--------|-----------------|--------------|-----------------|
| wyze_person_v1 (cross-camera, same clothes)| 382 | 3,745 | 5,164 | | | |
| wyze_person_v2 (cross-camera, same clothes) | | | | | | |
| wyze_person_v2 (cross-camera, cross clothes) | | | | | | |



## Prepare the dataset

Using the `wyze_person_v2_cross_clothes` for example, we first split the dataset into train and test sets, ensuring no identity overlapping.

```bash
cd prepare_dataset/
python  split_train_test.py

# calculate the embeddings and cosine simialrities between query and gallery images
python calculate_embed_sim.py

# format the test file, where k gallery images (> threshold) are defined for each query image
python prepare_gallery.py test 0.8

# visualize some gallery examples for sanity check

```


1. Access data in the google storage bucket

```bash
# verify you have access
gcloud auth list

# locate the data
gsutil ls
gsutil ls gs://wyze-ai-team-data/

# check local VM disk
df -h

# check the size of the data folder to be downloaded
gsutil du -sh gs://wyze-ai-team-data/wyze_person_v1/
gsutil du -sh gs://wyze-ai-team-data/wyze_person_v2/

# donwload the data folder to local VM
cd /home/tian.liu/data/
gsutil -m cp -r gs://wyze-ai-team-data/wyze_person_v1/ .
gsutil -m cp -r gs://wyze-ai-team-data/wyze_person_v2/ .
```

