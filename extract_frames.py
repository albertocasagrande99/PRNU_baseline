import cv2
import os
import numpy as np

def extract_frames(video_path, output_folder, device, i):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get some video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    frame_count = i
    while frame_count<(i+50):
        # Read the next frame
        ret, frame = video.read()

        if not ret:
            break

        # Save the frame as an image file
        frame_path = os.path.join(output_folder, device + f"_Nat_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

        # Display progress
        print(f"Extracting frame {frame_count}/{total_frames}")

    # Release the video file
    video.release()

    print("Frame extraction completed.")

fingerprint_devices = os.listdir("video_fingerprints/")
fingerprint_devices = sorted(np.unique(fingerprint_devices))
    
fingerprint_devices.remove('.DS_Store')

for device in fingerprint_devices:
    video_path = "test/Dataset/" + device + "/Videos/Flat/" + device + "_Flat_Move_0.mp4"
    output_folder = "video_fingerprints/" + device + "/Flat"
    extract_frames(video_path, output_folder, device, 50)
    video_path = "test/Dataset/" + device + "/Videos/Flat/" + device + "_Flat_Move_0.mov"
    extract_frames(video_path, output_folder, device, 50)
    video_path = "test/Dataset/" + device + "/Videos/Flat/" + device + "_Flat_Move_0.MOV"
    extract_frames(video_path, output_folder, device, 50)

'''indoor = device + "/Videos/Indoor/"
indoor_videos = os.listdir("test/Dataset/" + indoor)
#indoor_videos.remove(".DS_Store")
output_folder = "video_fingerprints/" + device + "/Nat"
i=0
for video in indoor_videos:
    video_path = "test/Dataset/" + indoor + video
    extract_frames(video_path, output_folder, device, i)
    i = i + 10

outdoor = device + "/Videos/Outdoor/"
outdoor_videos = os.listdir("test/Dataset/" + outdoor)
#outdoor_videos.remove(".DS_Store")
output_folder = "video_fingerprints/" + device + "/Nat"
i=40
for video in outdoor_videos:
    video_path = "test/Dataset/" + outdoor + video
    extract_frames(video_path, output_folder, device, i)
    i = i + 10'''