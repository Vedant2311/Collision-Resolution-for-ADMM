#source : https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python

import cv2
import os

image_folder = ("images/")
video_name = ("animations/collision_scipy10x10.avi")
frame = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 20, (width,height))

for img_id in range(len(os.listdir(image_folder))):
    video.write(cv2.imread(os.path.join(image_folder, "image" + str(img_id) + ".png")))

cv2.destroyAllWindows()
video.release()