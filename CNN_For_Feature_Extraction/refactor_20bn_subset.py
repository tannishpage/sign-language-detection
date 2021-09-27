import pandas as pd
import os

df = pd.read_csv("/home/tannishpage/Documents/Sign_Language_Detection/archive/Validation.csv")

FRAMES = "/home/tannishpage/Documents/Sign_Language_Detection/archive/Validation"

def copy_all_frames(from_dir, to, attachment):
    files = os.listdir(from_dir)
    for file in files:
        old_file = open(os.path.join(from_dir, file), 'rb')
        new_file = open(os.path.join(to, attachment + "_" + file), 'wb')
        new_file.write(old_file.read())
        old_file.close()
        new_file.close()
        os.remove(os.path.join(from_dir, file))
    os.rmdir(from_dir)

for video in zip(df['video_id'], df['label']):
    video_id = video[0]
    video_label = video[1]

    if not (os.path.exists(os.path.join(FRAMES, video_label))):
        os.mkdir(os.path.join(FRAMES, video_label))

    copy_all_frames(os.path.join(FRAMES, str(video_id)), os.path.join(FRAMES, video_label), str(video_id))
