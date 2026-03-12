
def get_frame_indexes(video_file_path, target_fps=2, min_frames=40):
    """
    Calculate frame indexes to extract.
    - Extract at target_fps (2 frames per second)
    - If video < 20 seconds, extract min_frames (40) evenly distributed
    """
    video = cv2.VideoCapture(video_file_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    if fps == 0 or total_frames == 0:
        return []

    duration = total_frames / fps  # Duration in seconds

    if duration >= 20:
        # Extract at target_fps (2 frames per second)
        frame_interval = int(fps / target_fps)
        frame_indexes = list(range(0, total_frames, frame_interval))
    else:
        # Video is less than 20 seconds, extract min_frames evenly distributed
        if total_frames < min_frames:
            # If total frames less than min_frames, use all frames
            frame_indexes = list(range(total_frames))
        else:
            # Distribute min_frames evenly across the video
            frame_interval = total_frames / min_frames
            frame_indexes = [int(i * frame_interval) for i in range(min_frames)]

    return frame_indexes

def extract_frames_from_video(video_metadata):
    """Extract frames from a single video"""
    try:
        user_id = video_metadata['user_id']
        event_id = video_metadata['event_id']
        device_model = video_metadata['device_model']

        video_file_path = join(VIDEO_DIRECTORY, str(user_id), f"{event_id}.mp4")

        if not isfile(video_file_path):
            print(f"Video not found: {video_file_path}")
            return

        # Create user-specific frames directory
        user_frames_dir = join(FRAMES_OUTPUT_DIRECTORY, str(user_id))
        if not exists(user_frames_dir):
            os.makedirs(user_frames_dir)

        # Get frame indexes to extract
        frame_indexes = get_frame_indexes(video_file_path)

        if not frame_indexes:
            print(f"Could not determine frame indexes for {event_id}")
            return

        # Open video and extract frames
        cap = cv2.VideoCapture(video_file_path)

        for idx in frame_indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                print(f"Failed to read frame {idx} from {event_id}")
                continue

            # Apply device-specific transformations
            if device_model == 'WYZEDB3':
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                h, w, _ = frame.shape
                new_w = int(h / 9 * 16)
                pad = int((new_w - w) / 2)
                frame = cv2.copyMakeBorder(frame, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                frame = cv2.resize(frame, (1920, 1080))
                # Save single frame
                image_filename = f"{event_id}_{str(idx).zfill(6)}.jpg"
                output_path = join(user_frames_dir, image_filename)
                cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            elif device_model == 'GW_DUO':
                # Split frame into top and bottom halves
                h, w, _ = frame.shape
                frame_up = frame[:h//2]
                frame_down = frame[h//2:]
                print(f"DUO {event_id}: frame_up {frame_up.shape}, frame_down {frame_down.shape}")

                image_filename_up = f"{event_id}_{str(idx).zfill(6)}_up.jpg"
                image_filename_down = f"{event_id}_{str(idx).zfill(6)}_down.jpg"
                output_path_up = join(user_frames_dir, image_filename_up)
                output_path_down = join(user_frames_dir, image_filename_down)

                cv2.imwrite(output_path_up, frame_up, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite(output_path_down, frame_down, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            elif device_model == 'GW_DBD':
                # Split frame at 1200 pixels
                frame_up = frame[:1200]
                frame_down = frame[1200:]
                print(f"DBD {event_id}: frame_up {frame_up.shape}, frame_down {frame_down.shape}")

                image_filename_up = f"{event_id}_{str(idx).zfill(6)}_up.jpg"
                image_filename_down = f"{event_id}_{str(idx).zfill(6)}_down.jpg"
                output_path_up = join(user_frames_dir, image_filename_up)
                output_path_down = join(user_frames_dir, image_filename_down)

                cv2.imwrite(output_path_up, frame_up, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite(output_path_down, frame_down, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            else:
                # Default: save frame as-is
                image_filename = f"{event_id}_{str(idx).zfill(6)}.jpg"
                output_path = join(user_frames_dir, image_filename)
                cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        cap.release()
 