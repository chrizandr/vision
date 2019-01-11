import cv2
import os


def convert_to_vid(frames_folder, video_file, fps=25, frame_size=(1280, 720)):
    """Converts a series of images to video and saves to a specified location."""
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file, codec, float(fps), frame_size, isColor=True)

    frames = os.listdir(frames_folder)
    filetype = "." + frames[0].split(".")[1]

    frames = [x.split(".")[0] for x in frames]
    frames.sort(key=int)
    frames = [x + filetype for x in frames]

    for f in frames:
        frame = cv2.imread(os.path.join(frames_folder, f))
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    frames_folder = "data/frames/"
    video_file = "output/output.avi"
    fps = 10
    frame_size = (1280, 720)

    convert_to_vid(frames_folder, video_file, fps=fps, frame_size=frame_size)
