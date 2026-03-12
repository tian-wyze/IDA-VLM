import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import pandas as pd


def assume_role():
    sts_client = boto3.client('sts')
    response = sts_client.assume_role(
        RoleArn='arn:aws:iam::596358926690:role/ai-dev-feedbackvideo-read-only-role',
        RoleSessionName='ai-dev-feedbackvideo-read-only-role-session'
    )
    return response['Credentials']


def get_s3_bucket():
    credentials = assume_role()
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    bucket = s3_resource.Bucket('wyze-feedback-video-service-596358926690-us-west-2')
    return bucket


def download(bucket, video_path, location):
    video_name = video_path.split('/')[-1]
    bucket.download_file(video_path, f'{location}/{video_name}')


def download_feedback_video_multi_thread(bucket, video_list, location):
    with ThreadPoolExecutor(max_workers=30) as executor:
        executor.map(lambda video_path: download(bucket, video_path, location), video_list)


def get_video_names_from_csv(csv_dir):
    video_names = []
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files[:]:
        print(csv_file)
        file_path = os.path.join(csv_dir, csv_file)
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            video_names.extend(df['FILE_PATH'].tolist())
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    
    print('length of video list:', len(video_names))

    return video_names


video_list = get_video_names_from_csv('./')
bucket = get_s3_bucket()
import pdb; pdb.set_trace()

folder_to_save = './videos_wyze_person_v2_cross_clothes'
if not os.path.exists(folder_to_save):
    os.makedirs(folder_to_save)

download_feedback_video_multi_thread(bucket, video_list, folder_to_save)
