import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2 as cv
import numpy as np
from ISR.models import RDN
from PIL import Image


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


def output_video_clip(frames, output_location, output_frame_rate,
                      clip_filename, output_resolution):
    filename = os.path.join(output_location, clip_filename)
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    writer = cv.VideoWriter(filename, fourcc, output_frame_rate,
                            output_resolution)
    for frame in frames:
        writer.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
    writer.release()


def main():
    clip_frame_cnt = 24 * 12
    model_id = 'psnr-small'
    target_video_filename = 'sample.mp4'
    inference_batch_size = 32
    output_frame_rate = 30
    output_width = 2560
    output_height = 1440
    output_resolution = (output_width, output_height)
    side_by_side_output_resolution = (output_width, output_height * 2)
    clips_output_location = os.path.join('.', 'output')
    vidcap = read_vidcap(target_video_filename)
    success = True
    clip_id = 0
    rdn = RDN(weights=model_id)
    while success:
        print('Generate clip #{id}'.format(id=clip_id))
        success, frames = read_frames(vidcap, max_frame=clip_frame_cnt)
        sr_frames = []
        baseline_frames = []
        side_by_side_frames = []
        for frame_id, frame in enumerate(frames):
            print('Process frame #{id}/{total}'.format(id=frame_id, total=len(frames)))
            sr_frame = rdn.predict(frame,
                                   by_patch_of_size=inference_batch_size)
            baseline_frame = np.array(Image.fromarray(frame).resize(output_resolution))
            side_by_side_frame = np.vstack([sr_frame, baseline_frame])
            sr_frames.append(sr_frame)
            baseline_frames.append(baseline_frame)
            side_by_side_frames.append(side_by_side_frame)
        output_video_clip(frames=sr_frames,
                          output_location=clips_output_location,
                          output_frame_rate=output_frame_rate,
                          clip_filename='sr_clip_{id}.mp4'.format(id=clip_id),
                          output_resolution=output_resolution)
        output_video_clip(
            frames=baseline_frames,
            output_location=clips_output_location,
            output_frame_rate=output_frame_rate,
            clip_filename='baseline_clip_{id}.mp4'.format(id=clip_id),
            output_resolution=output_resolution)
        output_video_clip(
            frames=side_by_side_frames,
            output_location=clips_output_location,
            output_frame_rate=output_frame_rate,
            clip_filename='side_by_side_clip_{id}.mp4'.format(id=clip_id),
            output_resolution=side_by_side_output_resolution)
        clip_id += 1
    vidcap.release()
    print('All Done!')


if __name__ == '__main__':
    main()
