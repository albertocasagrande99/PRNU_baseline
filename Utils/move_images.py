import os
import random
import shutil
import numpy as np

def adjust_destination_folder(destination_folder):
    destination_images = os.listdir(destination_folder)
    source_images = os.listdir(source_folder)
    i=0
    while (i < len(source_images) and len(destination_images)>0):
        image_to_delete = random.choice(destination_images)
        os.remove(os.path.join(destination_folder, image_to_delete))
        destination_images.remove(image_to_delete)
        i+=1

def move_all_images(source_folder, destination_folder):
    adjust_destination_folder(destination_folder)
    
    source_images = os.listdir(source_folder)
    destination_images = os.listdir(destination_folder)
    
    for image in source_images:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.move(source_path, destination_path)
    
    print(f"All images moved from {source_folder} to {destination_folder}.")
    
if __name__ == "__main__":
    fingerprint_devices = os.listdir("test/Videos/")
    fingerprint_devices = sorted(np.unique(fingerprint_devices))     
    fingerprint_devices.remove('.DS_Store')
    for device in fingerprint_devices:
        indoor = device + "/Videos/IframesOutdoor/"
        indoor_videos = os.listdir("test/Videos/" + indoor)
        test_videos = os.listdir("test/Videos/" + device + "/Videos/VideoLevel+/Test/")
        for video in indoor_videos:
            if video != '.DS_Store':
                source_folder = 'test/Videos/' + device + '/Videos/IframesOutdoor/' + video
                destination_folder = "test/Videos/" + device + "/Videos/VideoLevel+/Train/"
                move_all_images(source_folder, destination_folder)
                os.rmdir(source_folder)
        shutil.rmtree('test/Videos/' + device + '/Videos/IframesOutdoor/')
        shutil.rmtree('test/Videos/' + device + '/Videos/IframesIndoor/')
        
