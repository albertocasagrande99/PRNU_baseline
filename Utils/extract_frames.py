import cv2
import os
import numpy as np
import random
import shutil

def extract_frames(video_path, output_folder, device, i):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get some video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate frame interval for equally spaced frames
    frame_interval = 7  # Extract every 20th frame

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    #while frame_count<(i+50):    #for flat videos
    #while frame_count<(i+30):     #for natural videos
    while True:
        # Read the next frame
        ret, frame = video.read()

        if not ret:
            break
            
        # Save the frame as an image file
        #frame_path = os.path.join(output_folder, device + f"_Nat_{frame_count}.jpg")
        #cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Save the frame as an image file
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, device + f"_Nat_{i}.jpg")
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            i+=1

        frame_count += 1

        # Display progress
        print(f"Extracting frame {frame_count}/{total_frames}")

    # Release the video file
    video.release()

    print("Frame extraction completed.")

fingerprint_devices = os.listdir("test/Videos/")
fingerprint_devices = sorted(np.unique(fingerprint_devices))
    
fingerprint_devices.remove('.DS_Store')

#Extract flat frames

for device in fingerprint_devices:
    video_path = "test/Videos/" + device + "/Videos/Flat/" + device + "_Flat_Move_0.mp4"
    output_folder = "test/Videos/" + device + "/Videos/Flat/Frames"
    extract_frames(video_path, output_folder, device, 0)
    video_path = "test/Videos/" + device + "/Videos/Flat/" + device + "_Flat_Move_0.mov"
    output_folder = "test/Videos/" + device + "/Videos/Flat/Frames"
    extract_frames(video_path, output_folder, device, 0)
    video_path = "test/Videos/" + device + "/Videos/Flat/" + device + "_Flat_Move_0.MOV"
    output_folder = "test/Videos/" + device + "/Videos/Flat/Frames"
    extract_frames(video_path, output_folder, device, 0)


for device in fingerprint_devices:
    video_path = "test/Videos/" + device + "/Videos/Flat/" + device + "_Flat_Still_0.mp4"
    output_folder = "test/Videos/" + device + "/Videos/Flat/Frames"
    extract_frames(video_path, output_folder, device, 50)
    video_path = "test/Videos/" + device + "/Videos/Flat/" + device + "_Flat_Still_0.mov"
    output_folder = "test/Videos/" + device + "/Videos/Flat/Frames"
    extract_frames(video_path, output_folder, device, 50)
    video_path = "test/Videos/" + device + "/Videos/Flat/" + device + "_Flat_Still_0.MOV"
    output_folder = "test/Videos/" + device + "/Videos/Flat/Frames"
    extract_frames(video_path, output_folder, device, 50)


#Extract indoor frames

for device in fingerprint_devices:
    indoor = device + "/Videos/Indoor/"
    indoor_videos = os.listdir("test/Videos/" + indoor)
    #indoor_videos.remove(".DS_Store")
    i=0
    for video in indoor_videos:
        output_folder = "test/Videos/" + device + "/Videos/IframesIndoor/" + video
        video_path = "test/Videos/" + indoor + video
        #extract_frames(video_path, output_folder, device, i)
        os.makedirs(output_folder, exist_ok=True)
        os.system("ffmpeg -i test/Videos/" + indoor + video + " -vf \"select='eq(pict_type\,I)'\" -vsync vfr -qmin 1 -qscale:v 1 test/Videos/" + device + "/Videos/IframesIndoor/" + video + "/output_%03d.jpg")
        i = i + 40

#Extract outdoor frames
for device in fingerprint_devices:
    '''
    i=600
    indoor = device + "/Videos/IframesIndoor/"
    indoor_videos = os.listdir("test/Videos/" + indoor)
    for video in indoor_videos:
        if video != '.DS_Store':
            folder = "test/Videos/" + device + "/Videos/IframesIndoor/" + video
            frames = sorted(filter( lambda x: os.path.isfile(os.path.join(folder, x)),os.listdir(folder)))
            # Iterate through each file
            for filename in frames:
                if filename.lower().endswith('.jpg'):  # Check if the file is a JPEG image
                    old_path = os.path.join(folder, filename)
                    new_filename = device + "_Nat_" + str(i) + ".jpg"
                    new_path = os.path.join(folder, new_filename)
                    # Rename the file
                    os.rename(old_path, new_path)
                    i=i+1
    '''
    outdoor = device + "/Videos/Outdoor/"
    outdoor_videos = os.listdir("test/Videos/" + outdoor)
    #outdoor_videos.remove(".DS_Store")
    #output_folder = "test/Videos/" + device + "/Videos/Nat_frames5"
    i=80
    for video in outdoor_videos:
        output_folder = "test/Videos/" + device + "/Videos/IframesOutdoor/" + video
        video_path = "test/Videos/" + outdoor + video
        #extract_frames(video_path, output_folder, device, i)
        os.makedirs(output_folder, exist_ok=True)
        os.system("ffmpeg -i test/Videos/" + outdoor + video + " -vf \"select='eq(pict_type\,I)'\" -vsync vfr -qmin 1 -qscale:v 1 test/Videos/" + device + "/Videos/IframesOutdoor/" + video + "/output_%03d.jpg")
        i = i + 40

#Create train and test splits
for device in fingerprint_devices:
    # Set the paths for your source folder and destination folders
    source_folder = 'test/Videos/' + device + '/Videos/Nat_frames5'
    train_folder = 'test/Videos/' + device + '/Videos/Nat_frames5/Train'
    test_folder = 'test/Videos/' + device + '/Videos/Nat_frames5/Test'

    # List all image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Shuffle the image files
    random.shuffle(image_files)

    # Calculate the number of images for training and testing
    num_train = 100

    # Create destination folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Move images to the train folder
    for image_file in image_files[:num_train]:
        source_path = os.path.join(source_folder, image_file)
        dest_path = os.path.join(train_folder, image_file)
        shutil.move(source_path, dest_path)
    
    # Move images to the test folder
    for image_file in image_files[num_train:]:
        source_path = os.path.join(source_folder, image_file)
        dest_path = os.path.join(test_folder, image_file)
        shutil.move(source_path, dest_path)