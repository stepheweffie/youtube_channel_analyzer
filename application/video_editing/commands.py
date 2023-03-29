import argparse
from editor import VideoEditor
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import copy
import shutil

# Create the argparse object and define the arguments
parser = argparse.ArgumentParser(description='Edit a video file')
parser.add_argument('input_file', type=str, help='Input video file')
parser.add_argument('output_file', type=str, help='Output video file')
parser.add_argument('--preprocess', type=str, help='Takes two vido files')
parser.add_argument('--concatenate', type=str, help='Conjoin two video files')
parser.add_argument('--resize', type=str, help='Resize the video (widthxheight)')
parser.add_argument('--trim', type=str, help='Trim the video (start-end)')
parser.add_argument('--speedup', type=float, help='Speed up the video (factor)')
parser.add_argument('--slowmotion', type=float, help='Slow down the video (factor)')

# Parse the arguments and perform the video edits
args = parser.parse_args()

editor = VideoEditor(args.input_file)


def rename_input_file(input_file, new_filename):
    input_dir, input_filename = os.path.split(input_file)
    new_input_file = os.path.join(input_dir, new_filename)
    os.rename(input_file, new_input_file)
    return new_input_file


def preprocess_videos(input_files):
    # Rename the input files to "video1.mp4" and "video2.mp4"
    new_input_files = []
    for i, input_file in enumerate(input_files):
        new_input_file = rename_input_file(input_file, f'video{i + 1}.mp4')
        new_input_files.append(new_input_file)
    print('Preprocessing Complete')
    return new_input_files


if args.preprocess:
    video_array = []
    while len(video_array) < 2:
        get_input_file = input("Please provide the name of a video: ")
        get_input = copy.deepcopy(get_input_file)
        video_array.append(get_input)
    new_input = preprocess_videos(video_array)
    editors = [VideoEditor(new_input_file) for new_input_file in new_input]
    concatenated_clip = concatenate_videoclips([editor.clip for editor in editors])
else:
    editor = VideoEditor(args.input_file)
    # Perform other video edits and write the edited video to a file
    editor.write(args.output_file)


if args.concatenate:
    editor1 = VideoEditor('/video1.mp4')
    editor2 = VideoEditor('/video2.mp4')
    editor1.concatenate(editor2)
    editor1.write('/concatenated_video.mp4')

if args.resize:
    width, height = map(int, args.resize.split('x'))
    editor.resize(width, height)

if args.trim:
    start, end = map(float, args.trim.split('-'))
    editor.trim(start, end)

if args.speedup:
    editor.speedup(args.speedup)

if args.slowmotion:
    editor.slowmotion(args.slowmotion)

editor.write(args.output_file)
