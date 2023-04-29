"""
Main script.
Create panoramas from a set of images.
"""

import argparse
import logging
import os
import time
from typing import List

import cv2
import numpy as np

from src.images import Image
from src.matching import MultiImageMatches, PairMatch, build_homographies, find_connected_components
from src.rendering import multi_band_blending, set_gain_compensations, simple_blending


def read_video_frames(video_path):
    frames = []
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames

def add_zero_padding(image, right_padding, bottom_padding):
    # Get the height and width of the input image
    height, width = image.shape[:2]

    # Add zero padding to the right and bottom of the image
    padded_image = cv2.copyMakeBorder(image,
                                      top=0,
                                      bottom=bottom_padding,
                                      left=0,
                                      right=right_padding,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])
    return padded_image

def stitch_frames(frames):
    stitched_image = frames[0]
    step = 20
    images = []
    for i in range(step, len(frames),step):
        images.append(frames[i])

    logging.info("Computing features with SIFT...")
    images = [Image(str(ii), args.get("size"), img) for ii, img in enumerate(images)]
    for image in images:
        image.compute_features()

    logging.info("Matching images with features...")
    matcher = MultiImageMatches(images)
    pair_matches: List[PairMatch] = matcher.get_pair_matches()
    pair_matches.sort(key=lambda pair_match: len(pair_match.matches), reverse=True)

    logging.info("Finding connected components...")
    connected_components = find_connected_components(pair_matches)

    logging.info("Found %d connected components", len(connected_components))
    logging.info("Building homographies...")
    build_homographies(connected_components, pair_matches)

    time.sleep(0.1)

    logging.info("Computing gain compensations...")
    for connected_component in connected_components:
        component_matches = [
            pair_match for pair_match in pair_matches if pair_match.image_a in connected_components[0]
        ]

        set_gain_compensations(
            connected_components[0],
            component_matches,
            sigma_n=args["gain_sigma_n"],
            sigma_g=args["gain_sigma_g"],
        )

    time.sleep(0.1)

    for image in images:
        image.image = (image.image * image.gain[np.newaxis, np.newaxis, :]).astype(np.uint8)

    results = []

    if args["multi_band_blending"]:
        logging.info("Applying multi-band blending...")
        for connected_component in connected_components:
            results.append(
                multi_band_blending(
                    connected_component, num_bands=args["num_bands"], sigma=args["mbb_sigma"]
                )
            )

    else:
        logging.info("Applying simple blending...")
        for connected_component in connected_components:
            results.append(simple_blending(connected_component))

    logging.info("Saving results to %s", os.path.join(args["data_dir"], "results"))
    os.makedirs(os.path.join(args["data_dir"], "results"), exist_ok=True)
    for j, result in enumerate(results):
        cv2.imwrite(os.path.join(args["data_dir"], "results", f"pano_{i}_{j}.jpg"), result)
        cv2.imwrite(os.path.join(args["data_dir"], "results", f"pano_out.jpg"), result)
        stitched_image = result

    return results

def save_stitched_image(stitched_image, output_path):
    cv2.imwrite(output_path, stitched_image)


def main(args):
    if args["verbose"]:
        logging.basicConfig(level=logging.INFO)

    logging.info("Extract keyframes from ...")
    video_frames = read_video_frames(args["data_dir"]+'/'+args["input_vid_file"])
    stitched_image = stitch_frames(video_frames)
    ##save_stitched_image(stitched_image, args["output_img_file"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create panoramas from a set of images. \
                    All the images must be in the same directory. \
                    Multiple panoramas can be created at once"
    )
    parser.add_argument("--data_dir", type=str, default='/DL_data_super_ssd/EFPD_samples/n65', help="input data path")
    parser.add_argument("--input_vid_file", type=str, default='pig_vid.mp4', help="input video file path")
    parser.add_argument("--output_img_file", type=str, default='output.png',  help="output image file path")
    parser.add_argument(
        "-mbb",
        "--multi-band-blending",
        action="store_true",
        default=False,
        help="use multi-band blending instead of simple blending",
    )
    parser.add_argument("--size", type=int, help="maximum dimension to resize the images to")
    parser.add_argument(
        "--num-bands", type=int, default=5, help="number of bands for multi-band blending"
    )
    parser.add_argument("--mbb-sigma", type=float, default=1, help="sigma for multi-band blending")

    parser.add_argument("--gain-sigma-n", type=float, default=10, help="sigma_n for gain compensation")
    parser.add_argument("--gain-sigma-g", type=float, default=0.1, help="sigma_g for gain compensation")

    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")

    args = vars(parser.parse_args())
    main(args)