import cv2
import numpy as np


def get_frames(vid_file, title):
    """Converts a video to a series of images and saves to a specified location."""
    cam = cv2.VideoCapture(vid_file)
    frames = list()
    ret = True
    while(ret):
        ret, frame = cam.read()
        if not ret:
            break
        frames.append(frame)

    cam.release()
    cv2.destroyAllWindows()
    return frames


def convert_to_vid(frames, video_file, fps=25, frame_size=(1280, 720)):
    """Converts a series of images to video and saves to a specified location."""
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file, codec, float(fps), frame_size, isColor=True)

    for frame in frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def ChromaKey(fg, bg, keyColor, threshold=200):
    """Replace keyColor in fg with bg."""
    color_diff = fg.astype(np.float) - keyColor.astype(np.float)
    color_dist = np.sqrt(np.sum(color_diff * color_diff, axis=2))
    color_pixels = (color_dist > threshold).nonzero()

    transform_map = np.zeros(fg.shape, dtype=np.uint8)
    transform_map[color_pixels] = 1
    inverse_map = 1 - transform_map

    transform_map = transform_map * fg
    inverse_map = inverse_map * bg
    final_image = transform_map + inverse_map

    return final_image


def compose(fg_vid, bg_vid, keyColor, output_vid):
    print("Getting frames from FG video")
    fg_frames = get_frames(fg_vid, "Foreground video")
    print("Getting frames from BG video")
    bg_frames = get_frames(bg_vid, "Background video")

    bg_len = len(bg_frames)
    output = list()

    print("Chroma keying two videos")
    for i, f in enumerate(fg_frames):
        out = ChromaKey(f, bg_frames[i % bg_len], keyColor)
        output.append(out)

    print("Saving video at", output_vid)
    convert_to_vid(output, output_vid)


if __name__ == "__main__":
    output_vid = "output/chroma.avi"
    fg_vid = "data/fg.mp4"
    bg_vid = "data/bg.mp4"
    keyColor = np.array([0, 255, 0], dtype=np.uint8)

    compose(fg_vid, bg_vid, keyColor, output_vid)
