import cv2
import os


def convert_to_img(vid_file, output_folder):
    """Converts a video to a series of images and saves to a specified location."""
    cam = cv2.VideoCapture(vid_file)
    counter = 0
    ret = True
    while(ret):
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(output_folder, str(counter) + ".png"), frame)
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "data/sample.mp4"
    output_folder = "output/vid_to_img/"
    convert_to_img(video_file, output_folder)
