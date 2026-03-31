import cv2
import os
import sys


def images_to_video(image_folder, video_name, fps=30):
    # from Carstens ICRS framework
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    images.sort()

    if not images:
        print("No PNG images found in the specified folder.")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    print(f"Video '{video_name}' created successfully.")