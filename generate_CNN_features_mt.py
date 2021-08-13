"""
    This script performs the same task as generate_CNN_features.py, except that
    it uses multiple threads to parallelise the operation of loading the data.
"""
import sys
import os
import glob
import cv2
import argparse
import functools
import pytictoc
import numpy as np
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from keras import preprocessing
from keras.applications.vgg16 import preprocess_input
import common
from create_neural_net_model import create_cnn_model


def _process_video(video_list_item, input_path, input_file_mask, output_path, have_groundtruth_data, gt, cnn_model):
    i, video = video_list_item

    tt = pytictoc.TicToc()

    # create the output folder; if it exists, then CNN features have already been produced - skip the video
    capture = cv2.VideoCapture(os.path.join(input_path, video))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_output_folder = os.path.join(output_path, video.split(".")[0])
    print('processing video %d:  %s. Total Frames: %d' % (i, video, total_frames))
    if not os.path.exists(video):
        tt.tic()
        os.makedirs(video_output_folder)
        for frame_id in range(0, total_frames):
            result, frame = capture.read()
            skip_frame = False
            """try:
                skip_frame = True if have_groundtruth_data and gt[(video.split(".")[0], frame_id)] == '?' else False
            except Exception as e:
                print(e) # safest option is not to skip the frame"""

            if skip_frame:
                print("x", end='', flush=True)
            else:

                # Note that we don't scale the pixel values because VGG16 was not trained with normalised pixel values!
                # Instead we use the pre-processing function that comes specifically with the VGG16
                resized_frame = cv2.resize(frame, image_data_shape[0:2], interpolation=cv2.INTER_AREA)
                X = preprocess_input(resized_frame)

                X = np.expand_dims(X, axis=0)       # package as a batch of size 1, by adding an extra dimension

                # generate the CNN features for this batch
                #print(".", end='', flush=True) # TODO: Change the way we see progress. Maybe use a progress bar or something nicer and neater to represent the progress
                X_cnn = cnn_model.predict_on_batch(X)
                #print("{}:{}/{}".format(video, frame_id, total_frames))
                # save to disk
                output_file = os.path.join(video_output_folder, str(frame_id) + '.npy')
                np.savez(open(output_file, 'wb'), X=X_cnn)
        tt.toc()
        print("------ Finished Processing video {} ------".format(i))
        return True
    return False

def generate_CNN_features_mt_opencv(input_path, cnn_model, output_path, groundtruth_file, video_formats, image_data_shape):
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

    # the following line compiles the predict function. In multi thread setting, you have to manually call this function to compile
    # predict in advance, otherwise the predict function will not be compiled until you run it the first time, which will be
    # problematic when many threading calling it at once.
    cnn_model.make_predict_function()

    videos = list(enumerate([vid for vid in os.listdir(input_path) if vid.endswith(video_formats)]))
    print('Processing %d videos' % len(videos))

    # prepare for multiprocessing
    pool = ThreadPool(4)
    fn = functools.partial(_process_video, input_path=input_path, input_file_mask="", output_path=output_path,
            have_groundtruth_data=have_groundtruth_data, gt=gt, cnn_model=cnn_model)
    res = pool.map(fn, videos)

    print('\n\nReady')


def generate_CNN_features_mt(input_path, input_file_mask, cnn_model, output_path, groundtruth_file=""):
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

    # the following line compiles the predict function. In multi thread setting, you have to manually call this function to compile
    # predict in advance, otherwise the predict function will not be compiled until you run it the first time, which will be
    # problematic when many threading calling it at once.
    cnn_model.make_predict_function()

    # get all the video folders
    video_folders = os.listdir(input_path)
    video_folders.sort(key=common.natural_sort_key)

    video_list = list(enumerate(video_folders))
    print('Processing %d videos' % len(video_list))

    # prepare for multiprocessing
    pool = ThreadPool()
    fn = functools.partial(_process_video, input_path=input_path, input_file_mask=input_file_mask, output_path=output_path,
            have_groundtruth_data=have_groundtruth_data, gt=gt, cnn_model=cnn_model)
    res = pool.map(fn, video_list)

    print('\n\nReady')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
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
        generate_CNN_features_mt_opencv(input_path=args.input, cnn_model=model, output_path=args.output, groundtruth_file=args.groundtruth, video_formats=tuple(args.video_formats.split(" ")), image_data_shape=image_data_shape)
    else:
        generate_CNN_features_mt(input_path=args.input, input_file_mask=args.mask, cnn_model=model, output_path=args.output, groundtruth_file=args.groundtruth)
