import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main(args):
    # 1. Load the features
    print(f"[INFO] Loading features from '{args.features_path}'...")
    data = np.load(args.features_path, allow_pickle=True)

    available_keys = list(data.keys())

    if args.feature_key not in data:
        print(f"[ERROR] Key '{args.feature_key}' not found.")
        print(f"[INFO] Available keys are: {available_keys}")
        return

    features = data[args.feature_key]
    paths = data["paths"]

    print(f"[INFO] Loaded '{args.feature_key}' with shape {features.shape}")

    # 2. Extract pseudo-labels for coloring
    has_labels = False
    labels = []
    unique_labels = []

    if not args.no_colors and "paths" in data:
        try:
            # Use the parent directory name as the class label
            labels = [
                p.split(os.sep)[0] if os.sep in p else p.split("/")[0] for p in paths
            ]
            unique_labels = sorted(list(set(labels)))  # Sorted for consistent coloring

            if len(unique_labels) > 1:
                has_labels = True
                print(
                    f"[INFO] Found {len(unique_labels)} distinct classes. Coloring all of them."
                )
            else:
                print(
                    f"[INFO] Found {len(unique_labels)} classes. Defaulting to single-color plot."
                )
        except Exception as e:
            print(f"[WARNING] Could not parse labels from paths: {e}")

    # 3. Run t-SNE Dimensionality Reduction
    print(
        f"[INFO] Running t-SNE (perplexity={args.perplexity}). This might take a moment..."
    )
    tsne = TSNE(
        n_components=2, perplexity=args.perplexity, random_state=args.seed, init="pca"
    )
    features_2d = tsne.fit_transform(features)
    print(f"[INFO] t-SNE completed. Output shape: {features_2d.shape}")

    # 4. Plotting
    # Make the figure quite wide to accommodate a massive legend
    plt.figure(figsize=(16, 10))

    if has_labels:
        # 'turbo' or 'hsv' are great for many classes, even if not perfectly contrasting
        cmap = plt.get_cmap("turbo")

        for i, label in enumerate(unique_labels):
            idx = [j for j, l in enumerate(labels) if l == label]

            # Spread the colors evenly across the colormap
            color = cmap(i / max(1, len(unique_labels) - 1))

            plt.scatter(
                features_2d[idx, 0],
                features_2d[idx, 1],
                label=label,
                color=color,
                alpha=0.7,
                s=15,
                edgecolors="none",
            )

        # Format the legend dynamically to handle up to 100+ items without clipping
        num_classes = len(unique_labels)
        cols = max(1, num_classes // 20)  # E.g., 100 classes / 20 = 5 columns
        font_size = "small" if num_classes <= 40 else "x-small"

        plt.legend(
            title="Classes",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            ncol=cols,
            fontsize=font_size,
        )
    else:
        plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            alpha=0.7,
            s=15,
            c="royalblue",
            edgecolors="none",
        )

    plt.title(f"t-SNE Visualization: {args.feature_key}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)

    # Adjust layout so the massive legend isn't cut off
    plt.tight_layout()

    # 5. Save or Show
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"[INFO] Plot saved successfully to '{args.output}'")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize extracted multimodal features using t-SNE."
    )

    parser.add_argument(
        "--features-path",
        type=str,
        default="output\\features\\corel10k_clip_multimodal_features.npz",
        help="Path to the .npz file containing the extracted features.",
    )
    parser.add_argument(
        "--feature-key",
        type=str,
        default="image_features",
        help="Which feature array to visualize (e.g., 'image_features', 'text_features', 'fused_concat', 'fused_avg').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tsne_visualization.png",
        help="Path to save the output plot image. Leave empty to just show the plot.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="The perplexity parameter for t-SNE. Usually between 5 and 50.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--no-colors",
        action="store_true",
        help="Disable attempting to color points based on directory names in paths.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
