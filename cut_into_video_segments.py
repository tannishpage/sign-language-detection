"""
    This script splits a video into a number of video segments of fixed temporal duration.
    If groundtruth is available, then this script will ensure that each video segment
    contains frames having the same class label.
"""
import os
import glob
import argparse
import common
import cv2

# pylint: disable=no-member


def cut_into_video_segments(input_path, input_file_mask, output_path, video_segment_length, groundtruth_file, path_to_videos="", seg_len_time=0):
    # open and load the groundtruth data
    print('Loading groundtruth data...')
    gt = {}
    with open(groundtruth_file, 'r') as gt_file:
        gt_lines = gt_file.readlines()
    for gtl in gt_lines:
        gtf = gtl.rstrip().split(' ')
        if len(gtf) == 3:                   # our groundtruth file has 3 items per line (video ID, frame ID, class label)
            gt[(gtf[0], int(gtf[1]))] = gtf[2]
    print('ok\n')

    # create the output folder, if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get all the video folders
    video_folders = os.listdir(input_path)
    video_folders.sort(key=common.natural_sort_key)

    if path_to_videos:
        if seg_len_time > 0:
            folders_to_videos = {x.split(".")[0]:os.path.join(path_to_videos, x) for x in os.listdir(path_to_videos)}

    for (i, video_i) in enumerate(video_folders):
        print('#', i, ': cutting ', video_i, end=' ')

        # for each video, get the list of its extracted frames
        video_i_images = glob.glob(os.path.join(input_path, video_i, input_file_mask))
        video_i_images.sort(key=common.natural_sort_key)       # ensure images are in the correct order to preserve temporal sequence
        assert(len(video_i_images) > 0), "video %s has no frames!!!" % video_i
        if path_to_videos:
            if seg_len_time > 0:
                video = cv2.VideoCapture(folders_to_videos[video_i])
                video_segment_length = video.get(cv2.CAP_PROP_FPS) * seg_len_time
                video.release()
        segment = []
        segment_k = 0
        for image_j in video_i_images:
            frame_id = int(os.path.splitext(os.path.basename(image_j))[0])

            # groundtruth available?
            gt_label = '?'
            try:
                gt_label = gt[(video_i, frame_id)]
            except:
                pass

            segment.append((image_j, gt_label))

            # have we gathered a full video segment?
            if len(segment) >= video_segment_length:
                video_segment_id = video_i + '__%d' % segment_k

                # save to file
                with open(os.path.join(output_path, video_segment_id + ".txt"), "w") as f:
                    for l in segment:
                        f.write(l[0] + ' ' + l[1] + '\n')

                segment_k += 1
                segment = []

        # non-complete video segment?
        if len(segment) > 0 and len(segment) < video_segment_length:
            while len(segment) < video_segment_length:
                segment.append(segment[-1])         # repeat the last image so as to ensure all video segments are of the same length

            video_segment_id = video_i + '__%d' % segment_k

            # save to file
            with open(os.path.join(output_path, video_segment_id + ".txt"), "w") as f:
                for l in segment:
                    f.write(l[0] + ' ' + l[1] + '\n')

        print('   into %d video segments' % segment_k)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", help="Path to the input parent folder containing the video frames of the downloaded YouTube videos", default="")
    argparser.add_argument("--mask", help="The file mask to use for the video frames of the downloaded YouTube videos", default="*.jpg")
    argparser.add_argument("--output", help="Path to the output file containing the list of video segments", default="")
    argparser.add_argument("--len", help="Length of the video segments (in number of frames)", default=100)
    argparser.add_argument("--gt", help="Path to the groundtruth file", default="")
    argparser.add_argument("--path_to_videos", help="Path to the original videos", default="") # If someone passes len=0, we can use the original video's frame rate to determine the segment length
    argparser.add_argument("--seg_len_time", help="Length of the video segments (in seconds)", default=0) # Should be checked only if --path_to_videos is passed and --len=0
    args = argparser.parse_args()

    if not args.input or not args.output:
        argparser.print_help()
        exit()

    if args.path_to_videos:
        if int(args.seg_len_time) > 0:
            cut_into_video_segments(input_path=args.input, input_file_mask=args.mask, output_path=args.output, video_segment_length=int(args.len), groundtruth_file=args.gt, path_to_videos=args.path_to_videos, seg_len_time=int(args.seg_len_time))
    else:
        cut_into_video_segments(input_path=args.input, input_file_mask=args.mask, output_path=args.output, video_segment_length=int(args.len), groundtruth_file=args.gt)
