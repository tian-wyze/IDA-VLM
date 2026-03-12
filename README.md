# IDA-VLM
Tian's work on IDA-VLM


## Updates

Create WYZE in-house IDA-VLM benchmark.


## Env

```bash
conda create -n ida-vlm python=3.10
conda clean --all

pip install pandas tqdm boto3 matplotlib
# in case nvidia driver was swept due to GCP kernel update, reinstall drivers
sudo /opt/deeplearning/install-driver.sh

# install torch and xformers of compatible versions
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.28.post3
```


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
python split_train_test.py

# calculate the embeddings and cosine simialrities between query and gallery images
python calculate_embed_sim.py

# format the test file, where k gallery images (> threshold) are defined for each query image
python prepare_gallery.py test 5 0.5

# visualize some gallery examples for sanity check
python visualize_gallery.py
```

## Test VLM

```bash

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

# to increase VM disk space, first update the disk space on GCP portal
# then grow the partition and stretch the file system
sudo growpart /dev/sda 1
sudo resize2fs /dev/sda1
```

