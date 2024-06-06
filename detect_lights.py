"""
Test script for detecting bright areas on a video feed.
This script uses the OpenCV computer vision library.
https://opencv.org/
https://pyimagesearch.com/start-here/
"""

__date__ = "2024-04-17"
__author__ = "Geno Navarro"
__all__ = [
    "get_file",
    "get_cameras",
]
import cv2
import numpy
import imutils
from imutils import contours as imutils_contours
from pygrabber.dshow_graph import FilterGraph
from skimage import measure  # for connected component analysis
from datetime import datetime
from datetime import timedelta
from dateutil import tz
from suntime import Sun
from tkinter import filedialog
import argparse
import os
import sys

TIMEZONE = "US/Central"
SAVE_DETECTED_EVERY_TSEC = 5
SIZE_OF_BRIGHT_REGION_PIXELS = 100
TIMESTAMP_TEXT_COLOR = (255, 255, 255)


def set_cli_flags() -> int:
    """
    Add command line options when running the script. To show available options, use the command:
        python detect_lights.py -h

    Parameters
    ----------
    None

    Returns
    -------
    cli_timeout_hr: int
        Number of hours to run the script, values 0-168
    """
    parser = argparse.ArgumentParser(
        description="Test script for detecting bright areas on a video feed."
    )
    parser.add_argument(
        "-nt",
        "--notimeout",
        action="store_true",
        help="Use this option to endlessly run the script until a manual exit.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        help="Number of hours to run the script, values 0-168. Leave empty to run the script until the next sunrise.",
    )
    args = parser.parse_args()
    cli_notimeout = args.notimeout
    cli_timeout_hr = args.timeout
    cli_timeout_hr = 0 if cli_timeout_hr == None else cli_timeout_hr
    if cli_timeout_hr < 0 or cli_timeout_hr > 168:
        cli_timeout_hr = -1
    if cli_notimeout:
        cli_timeout_hr = None
    return cli_timeout_hr


def get_file(file_path: str = "") -> str:
    """
    Allows the user to select the file to be analyzed by showing a file explorer pop-up.

    Parameters
    ----------
    file_path: str
        The path and filename for the file.
        If empty, a file explorer dialog box will pop-up for file selection.

    Returns
    -------
        The path and filename for the file.
    """
    if file_path == "":
        print("\nChoose file to read...")
        file_path = filedialog.askopenfilename()
    print(file_path)
    return file_path


def preprocess_image(orig_image: numpy.ndarray, threshold=200) -> numpy.ndarray:
    """
    Convert a cv2 image to a binary image.

    Parameters
    ----------
    orig_image: numpy.ndarray
        The cv2 image to analyze.
    threshold: int
        The pixel intensities lower than this value are replaced as 0 (black),
        while pixel intensities higher than this value are replaced as 255 (white),

    Returns
    -------
        The binary cv2 image with thresholding applied.
    """
    grayscale = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (11, 11), 0)
    binary_image = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]
    binary_image = cv2.erode(binary_image, None, iterations=2)
    binary_image = cv2.dilate(binary_image, None, iterations=4)
    return binary_image


def detect_regions(orig_image: numpy.ndarray, binary_image: numpy.ndarray) -> int:
    """
    Labels and encircles the detected bright regions on an image.

    Parameters
    ----------
    orig_image: numpy.ndarray
        The original cv2 image
    binary_image: numpy.ndarray
        The binary image of orig_image.

    Returns
    -------
        The total count of bright regions on the binary image.
    """
    # We can filter out small pixels/noise by analyzing only the connected components
    labels = measure.label(binary_image)

    contour_count = 0
    mask = numpy.zeros(binary_image.shape, dtype="uint8")

    for label in numpy.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the number of pixels
        label_mask = numpy.zeros(binary_image.shape, dtype="uint8")
        label_mask[labels == label] = 255
        pixel_count = cv2.countNonZero(label_mask)

        # if the number of pixels in the component is sufficiently large,
        # then add it to our mask of "large blobs"
        if pixel_count > SIZE_OF_BRIGHT_REGION_PIXELS:
            mask = cv2.add(mask, label_mask)

            # highlight the bright spot
            contours = cv2.findContours(
                mask.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contours = imutils.grab_contours(contours)
            contours = imutils_contours.sort_contours(contours)[0]
            for contour_index, each_contour in enumerate(contours):
                contour_count += 1
                # draw the bright spot on the original image
                (x, y, w, h) = cv2.boundingRect(each_contour)
                ((cX, cY), radius) = cv2.minEnclosingCircle(each_contour)
                cv2.circle(orig_image, (int(cX), int(cY)), int(radius), (0, 255, 0), 3)
                cv2.putText(
                    orig_image,
                    f"#{contour_index + 1}",
                    (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2,
                )
    return contour_count


def add_time_on_image(orig_image: numpy.ndarray, additional_str="") -> datetime:
    """
    Show date and time on an image.

    Parameters
    ----------
    orig_image: numpy.ndarray
        The cv2 image that will have the timestamp
    additional_str: str
        Additional messages to display below the timestamp

    Returns
    -------
        The timestamp as a datetime object
    """
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H-%M-%S")
    cv2.putText(
        orig_image,
        f"{timestamp_str}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        TIMESTAMP_TEXT_COLOR,
    )
    if additional_str != "":
        cv2.putText(
            orig_image,
            f"{additional_str}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            TIMESTAMP_TEXT_COLOR,
        )
    return timestamp


def save_image(image: numpy.ndarray, timestamp: datetime) -> bool:
    """
    Save an image with timestamp as the filename

    Parameters
    ----------
    image: numpy.ndarray
        The cv2 image to save
    timestamp: datetime
        Datetime used as the filename

    Returns
    -------
        True if the image was saved, False otherwise
    """
    timestamp_str = timestamp.strftime("%Y-%m-%d %H-%M-%S")
    isImageSaved = cv2.imwrite(f"{timestamp_str}.png", image)
    print(f"Image {timestamp_str}.png {'saved' if (isImageSaved) else 'NOT saved'}")
    return isImageSaved


def get_cameras() -> dict:
    """
    Returns the list of camera devices as a Dictionary type

    Parameters
    ----------
    None

    Returns
    -------
        {0: device_name0, 1 : device_name1, 2 : device_name2, ...}
    """
    available_cameras = {}
    devices = FilterGraph().get_input_devices()
    print("Listing available cameras...")
    for device_port, device_name in enumerate(devices):
        print(f"Device {device_port}: {device_name}")
        available_cameras[device_port] = device_name
    return available_cameras


def connect_to_video() -> cv2.VideoCapture:
    """
    Asks the user to select a camera to connect to.

    Parameters
    ----------
    None

    Returns
    -------
        VideoCapture object
    """
    capture = cv2.VideoCapture()
    available_cameras = get_cameras()
    total_device_count = len(available_cameras)
    while True:
        print("Enter 'f' to choose a video file")
        print("Enter 'q' to quit this program")
        selection = input(f"Or choose a camera [0-{total_device_count - 1}]: ")
        if selection == "q":
            exit(0)
        elif selection == "f":
            vfile = get_file()
            capture = cv2.VideoCapture(vfile)
            break
        else:
            try:
                selection = int(selection)
            except ValueError:
                continue
            else:
                if 0 <= selection < total_device_count:
                    print(f"Connecting to {available_cameras[selection]}...")
                    capture = cv2.VideoCapture(selection, cv2.CAP_DSHOW)
                    break
    return capture


def get_next_sunrise() -> datetime:
    """
    Get the sunrise time of the next day

    Parameters
    ----------
    None

    Returns
    -------
        The time of sunrise as a datetime object.
    """
    _location_lat = 41.87536
    _location_lon = -87.63870
    _sun = Sun(_location_lat, _location_lon)
    _at_date = datetime.now() + timedelta(days=1)
    return _sun.get_sunrise_time(_at_date, tz.gettz(TIMEZONE))


def configure_timeout(runtime_hours: int) -> datetime | None:
    """
    Set the date when the program should stop.

    Parameters
    ----------
    runtime_hours: int
        Number of hours to run.
        Set to -1 for the next sunrise.
        Set to None to run endlessly.

    Returns
    -------
        The date when to stop as a datetime object. None if set to run endlessly.
    """
    if runtime_hours == None:
        print(f"This program will stop continuously until a manual exit.")
        return None

    _date = datetime.now().astimezone(tz.gettz(TIMEZONE))
    if runtime_hours > 0:
        _date = _date + timedelta(hours=runtime_hours)
    else:
        _date = get_next_sunrise()
    time_str = _date.strftime("%Y-%m-%d %H:%M:%S")
    print(f"This program will stop on {time_str}")
    return _date


def detect_lights(runtime_hours: int):
    """
    Detect bright lights on a video feed.

    Parameters
    ----------
    runtime_hours: int
        Number of hours to run.
        Set to -1 for the next sunrise.
        Set to None to run endlessly.

    Returns
    -------
    None
    """
    hit_counter = 0
    save_counter = 0
    next_save_time = datetime.now().astimezone(tz.gettz(TIMEZONE))
    timeout_date = configure_timeout(runtime_hours)

    video = connect_to_video()
    print("\nPress q while on the videostream to exit\n")

    while video.isOpened():
        is_reading, frame = video.read()
        if not is_reading:
            continue

        binary_image = preprocess_image(frame, threshold=230)
        bright_region_count = detect_regions(frame, binary_image)
        hit_counter += bright_region_count

        _timestamp = add_time_on_image(frame, f"bright regions detected: {hit_counter}")
        timestamp = _timestamp.astimezone(tz.gettz(TIMEZONE))

        if (bright_region_count > 0) and (next_save_time < timestamp):
            if save_image(frame, timestamp):
                save_counter += 1
            next_save_time = timestamp + timedelta(seconds=SAVE_DETECTED_EVERY_TSEC)
        else:
            print(f"No bright regions detected", end="\r", flush=True)

        cv2.imshow("frame", frame)

        if timeout_date and (timeout_date < timestamp):
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"\nTotal of {save_counter} images were saved in {os.getcwd()}\n")
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if sys.version_info < (3, 11):
        raise RuntimeError("This package requres Python 3.11+")
    cli_timeout_hr = set_cli_flags()
    _str = (
        f"This program will detect bright lights on a video feed."
        f" Detection will work best on low-light conditions."
        f" Bright areas on the video will be counted and highlighted,"
        f" and a snapshot of the video feed will be saved with the timestamp as the filename on:\n"
        f" {os.getcwd()}\n"
        f"To show available command line options, use the command: python detect_lights.py -h\n"
    )
    print(_str)
    detect_lights(cli_timeout_hr)
