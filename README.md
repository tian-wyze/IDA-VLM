# IDA-VLM
Building Identity-Aware VLM.


## Updates

* Mar 19, 2026: tested InternVL-3.5-8B
* Mar 17, 2026: created WYZE in-house Person ReID VLM benchmarks.


## Env



## Benchmarks

WYZE in-house data

| Dataset | Identities (total) | Query Images | Gallery Images | Train Split (identity/imgs) | Test Split (identity/imgs)
|---------|-----------|--------|-----------------|--------------|-----------------|
| wyze_person_v1 (cross-camera, same clothes)| 382 | 3,745 | 5,164 | | | |
| wyze_person_v2 (cross-camera, same clothes) | | | | | | |
| wyze_person_v2 (cross-camera, cross clothes) | | | | | | |





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

To download videos from AWS S3 bucketsL

```bash
# install awscli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o  "awscliv2.zip" && unzip awscliv2.zip && sudo ./aws/install --update

# refresh terminal command cache
hash -r

# autheticate in web
aws configure sso

# set the env variable
export AWS_PROFILE=AWSPowerUserAccess-447056034859

# run the downloading script
cd /home/tian.liu/IDA-VLM/download_videos
python download_video.py

# extract full frames from the videos
python extract_frames.py

```