"""
    This script extracts CNN features for the given video frames,
    MHI images, or frame differencing images.

    Note that for optical flow data, the script generate_CNN_features_from_flow_data.py
    should be used instead.
"""
import sys
import os
import glob
import argparse
import pytictoc
import numpy as np
from keras import preprocessing
from keras.applications.vgg16 import preprocess_input
import common
from create_neural_net_model import create_cnn_model
import cv2

def get_groundtruth_data(groundtruth_file):
    gt = {}
    have_groundtruth_data = False
    if len(groundtruth_file) > 0:
        try:
            # open and load the groundtruth data
            print('Loading groundtruth data...')
            with open(groundtruth_file, 'r') as gt_file:
                gt_lines = gt_file.readlines()
            for gtl in gt_lines:
                gtf = gtl.rstrip().split(' ')
                if len(gtf) == 3:                   # our groundtruth file has 3 items per line (video ID, frame ID, class label)
                    gt[(gtf[0], int(gtf[1]))] = gtf[2]
            print('ok\n')
            have_groundtruth_data = True
        except:
            pass
    return gt

def generate_CNN_features_opencv(input_path, cnn_model, output_path, groundtruth_file, video_formats, image_data_shape):
    gt = get_groundtruth_data(groundtruth_file)
    have_groundtruth_data = False
    if (len(gt) > 0):
        have_groundtruth_data = True
    videos = [vid for vid in os.listdir(input_path) if vid.endswith(video_formats)]
    tt = pytictoc.TicToc()

    for i, video in enumerate(videos):
        capture = cv2.VideoCapture(os.path.join(input_path, video))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_output_folder = os.path.join(output_path, video.split(".")[0])
        print('processing video %d of %d:  %s: ' % (i+1, len(videos), video))
        if not os.path.exists(video_output_folder):
            tt.tic()
            os.makedirs(video_output_folder)
            for frame_id in range(0, total_frames):
                result, frame = capture.read()
                if not result:
                    print("\tOpenCV read more frames then there are in the video", end='')
                    break
                skip_frame = False
                try:
                    skip_frame = True if have_groundtruth_data and gt[(video.split(".")[0], frame_id)] == '?' else False
                except Exception as e:
                    print(e) # safest option is not to skip the frame

                if skip_frame:
                    print("x", end='', flush=True)
                else:

                    # Note that we don't scale the pixel values because VGG16 was not trained with normalised pixel values!
                    # Instead we use the pre-processing function that comes specifically with the VGG16
                    resized_frame = cv2.resize(frame, image_data_shape[0:2], interpolation=cv2.INTER_AREA)
                    X = preprocess_input(resized_frame)

                    X = np.expand_dims(X, axis=0)       # package as a batch of size 1, by adding an extra dimension

                    # generate the CNN features for this batch
                    #print(".", end='', flush=True)
                    sys.stdout.write("\r{:.2f}%".format( (frame_id/total_frames) * 100))
                    X_cnn = cnn_model.predict_on_batch(X)

                    # save to disk
                    output_file = os.path.join(video_output_folder, str(frame_id) + '.npy')
                    np.savez(open(output_file, 'wb'), X=X_cnn)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            print()
            tt.toc()



def generate_CNN_features(input_path, input_file_mask, cnn_model, output_path, groundtruth_file=""):
    # groundtruth data?
    gt = {}
    have_groundtruth_data = False
    if len(groundtruth_file) > 0:
        try:
            # open and load the groundtruth data
            print('Loading groundtruth data...')
            with open(groundtruth_file, 'r') as gt_file:
                gt_lines = gt_file.readlines()
            for gtl in gt_lines:
                gtf = gtl.rstrip().split(' ')
                if len(gtf) == 3:                   # our groundtruth file has 3 items per line (video ID, frame ID, class label)
                    gt[(gtf[0], int(gtf[1]))] = gtf[2]
            print('ok\n')
            have_groundtruth_data = True
        except:
            pass

    tt = pytictoc.TicToc()

    # get all the video folders

    ## TODO: replace this code to use opencv instead of the extracted frames
    ##       because this takes a lot more hdd space
    video_folders = os.listdir(input_path)
    video_folders.sort(key=common.natural_sort_key)

    for (i, video_i) in enumerate(video_folders):
        print('processing video %d of %d:  %s' % (i+1, len(video_folders), video_i))

        # create the output folder; if it exists, then CNN features have already been produced - skip the video
        video_i_output_folder = os.path.join(output_path, video_i)
        if not os.path.exists(video_i_output_folder):
            tt.tic()
            os.makedirs(video_i_output_folder)

            # get the list of extracted frames for this video
            video_i_images = glob.glob(os.path.join(input_path, video_i, input_file_mask))
            video_i_images.sort(key=common.natural_sort_key)       # ensure images are in the correct order to preserve temporal sequence
            length = len(video_i_images)
            assert(len(video_i_images) > 0), "video %s has no frames!!!" % video_i

            # for each video frame...
            for i, image_j in enumerate(video_i_images):
                frame_id = int(os.path.splitext(os.path.basename(image_j))[0])
                skip_frame = False
                try:
                    skip_frame = True if have_groundtruth_data and gt[(video_i, frame_id)] == '?' else False
                except:
                    pass    # safest option is not to skip the frame

                if skip_frame:
                    print("x", end='', flush=True)
                else:
                    # load the image and convert to numpy 3D array
                    img = np.array(preprocessing.image.load_img(image_j))

                    # Note that we don't scale the pixel values because VGG16 was not trained with normalised pixel values!
                    # Instead we use the pre-processing function that comes specifically with the VGG16
                    X = preprocess_input(img)

                    X = np.expand_dims(X, axis=0)       # package as a batch of size 1, by adding an extra dimension

                    # generate the CNN features for this batch
                    sys.stdout.write("\r[{}{}] {:.2f}%".format('='*int((((i + 1)/length)*100)/2), '.' *(50 - int((((i + 1)/length)*100)/2)), ((i + 1)/length)*100))
                    sys.stdout.flush()
                    X_cnn = cnn_model.predict_on_batch(X)

                    # save to disk
                    output_file = os.path.join(video_i_output_folder, os.path.splitext(os.path.basename(image_j))[0] + '.npy')
                    np.savez(open(output_file, 'wb'), X=X_cnn)
            print(" ", end='')
            tt.toc()
        print('\n\n')

        """if msvcrt.kbhit():  # if a key is pressed
            key = msvcrt.getch()
            if key == b'q' or key == b'Q':
                print('User termination')
                return"""

    print('\n\nReady')


if __name__ == "__main__":
    argparser  =argparse.ArgumentParser()
    argparser.add_argument("--input", help="Path to the input parent folder containing the video frames of the downloaded YouTube videos", default="")
    argparser.add_argument("--mask", help="The file mask to use for the video frames of the downloaded YouTube videos", default="*.jpg")
    argparser.add_argument("--output", help="Path to the output folder where the the CNN features will be extracted to", default="")
    argparser.add_argument("--imwidth", help="Video frame width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Video frame height (in pixels)", default=224)
    argparser.add_argument("--fc1_layer", help="Include the first fully-connected layer (fc1) of the CNN", default=True)
    argparser.add_argument("--groundtruth", help="If groundtruth is available, then we load the file in order to only process video frames which have been labelled.", default="")
    argparser.add_argument("--opencv_mode", help="Use the video files directly", default=False)
    argparser.add_argument("--video_formats", help="What are the formats of the videos used? (Only applicable if you use --opencv_mode=True) Example: --video_formats \"mp4 mkv\", or --video_formats mp4")
    args = argparser.parse_args()

    if not args.input or not args.output:
        argparser.print_help()
        exit()

    image_data_shape = (args.imwidth, args.imheight, 3)   # width, height, channels
    model = create_cnn_model(image_data_shape, include_fc1_layer=args.fc1_layer)

    if (args.opencv_mode):
        generate_CNN_features_opencv(input_path=args.input, cnn_model=model, output_path=args.output, groundtruth_file=args.groundtruth, video_formats=tuple(args.video_formats.split(" ")), image_data_shape=image_data_shape)
    else:
        generate_CNN_features(input_path=args.input, input_file_mask=args.mask, cnn_model=model, output_path=args.output, groundtruth_file=args.groundtruth)
