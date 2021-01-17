import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from ISR.models import RDN

def health_check():
    gpus = tf.config.list_physical_devices("GPU")
    print('Found {cnt} GPUs available.'.format(cnt=len(gpus)))
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print('All GPU set to allow memory growth.')

def read_vidcap(filename):
    vidcap = cv.VideoCapture(os.path.join('.', 'data', filename))
    return vidcap

def read_frames(vidcap, max_frame=500):
    success, frame, frames = True, None, []
    while success and len(frames) < max_frame:
        success, frame = vidcap.read()
        frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    return success, frames

def output_video_clip(frames, output_location, clip_id):
    filename = os.path.join(output_location, 'clip_{id}.mp4'.format(id=clip_id))
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    writer = cv.VideoWriter(filename, fourcc, 20, (2560, 1440))
    for frame in frames:
        writer.write(frame)
    writer.release()

def main():
    target_video_filename = 'sample.mp4'
    clips_output_location = os.path.join('.', 'output')
    vidcap = read_vidcap(target_video_filename)
    success = True
    clip_id = 0
    rdn = RDN(weights='noise-cancel')
    while success:
        print('Generate clip #{id}'.format(id=clip_id))
        success, frames = read_frames(vidcap, max_frame=100)
        sr_frames = []
        for frame_id, frame in enumerate(frames):
            print('Process frame #{id}'.format(id=frame_id))
            sr_frame = rdn.predict(frame, by_patch_of_size=50)
            sr_frames.append(sr_frame)
        output_video_clip(sr_frames, clips_output_location, clip_id)
        clip_id += 1
    vidcap.release()
    print('All Done!')


if __name__ == '__main__':
    main()