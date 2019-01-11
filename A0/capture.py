import cv2
import os


def capture_video(output_folder):
    """Captures video as frames from camera and saves to a specified location."""
    cap = cv2.VideoCapture(0)
    counter = 0

    while(True):
        ret, frame = cap.read()

        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(output_folder, str(counter) + ".png"), frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    output_folder = "output/capture/"

    capture_video(output_folder)
