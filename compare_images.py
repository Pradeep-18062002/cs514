"""
Compare two images using perceptual hashing (pHash).

Usage:
    python compare_images.py <image1> <image2> [--threshold N]

The script computes the pHash of each image and reports the Hamming distance
between the two hashes. A distance of 0 means identical images; values up to
the threshold (default 10) are treated as "same or near-duplicate".
"""

import argparse
import sys
from pathlib import Path

import imagehash
from PIL import Image


ALGORITHMS = {
    "phash": imagehash.phash,
    "ahash": imagehash.average_hash,
    "dhash": imagehash.dhash,
    "whash": imagehash.whash,
}


def load_image(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():
        sys.exit(f"Error: file not found — {path}")
    try:
        return Image.open(p).convert("RGB")
    except Exception as e:
        sys.exit(f"Error opening {path}: {e}")


def compare(image1_path: str, image2_path: str, algorithm: str = "phash", threshold: int = 10) -> None:
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    hash_fn = ALGORITHMS[algorithm]
    h1 = hash_fn(img1)
    h2 = hash_fn(img2)

    distance = h1 - h2  # Hamming distance between the two hashes

    print(f"Algorithm  : {algorithm}")
    print(f"Image 1    : {image1_path}")
    print(f"  hash     : {h1}")
    print(f"Image 2    : {image2_path}")
    print(f"  hash     : {h2}")
    print(f"Hamming distance : {distance}")
    print(f"Threshold        : {threshold}")

    if distance == 0:
        verdict = "IDENTICAL — exact perceptual match"
    elif distance <= threshold:
        verdict = f"SIMILAR — likely the same image (distance {distance} <= {threshold})"
    else:
        verdict = f"DIFFERENT — images are distinct (distance {distance} > {threshold})"

    print(f"\nVerdict: {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Compare two images using perceptual hashing.")
    parser.add_argument("image1", help="Path to the first image")
    parser.add_argument("image2", help="Path to the second image")
    parser.add_argument(
        "--algorithm",
        choices=list(ALGORITHMS.keys()),
        default="phash",
        help="Hashing algorithm to use (default: phash)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Max Hamming distance to consider images the same (default: 10)",
    )
    args = parser.parse_args()
    compare(args.image1, args.image2, algorithm=args.algorithm, threshold=args.threshold)


if __name__ == "__main__":
    main()
