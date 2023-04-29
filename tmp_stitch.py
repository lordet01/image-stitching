import cv2
import numpy as np
import logging
from typing import List
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

def stitch_frames(frames):
    stitched_image = frames[0]
    step = 10
    for i in range(step, len(frames),step):

        logging.info("Computing features with SIFT...")
        images = [stitched_image, frames[i]]
        images = [Image(None, args.get("size"), img) for img in images]
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
            for i, result in enumerate(results):
                cv2.imwrite(os.path.join(args["data_dir"], "results", f"pano_{i}.jpg"), result)

    return stitched_image


def save_stitched_image(stitched_image, output_path):
    cv2.imwrite(output_path, stitched_image)

def main(input_video, output_image):
    video_frames = read_video_frames(input_video)
    stitched_image = stitch_frames(video_frames)
    save_stitched_image(stitched_image, output_image)

if __name__ == "__main__":
    input_video = "input.mp4"
    output_image = "output.png"
    main(input_video, output_image)