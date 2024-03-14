'''
The following code is adapted from: https://github.com/uzh-rpg/rpg_vision-based_slam/blob/main/scripts/python/create_colmap_project_uzhfpv_dataset.py
Origianl author: Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
'''


import argparse
import yaml
import os
import requests

from utils import colmap_params
from utils import util_colmap


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgdir", dest="imgdir", help="image directory", required="true"
    )
    parser.add_argument("--param", dest="param",
                        help="parameters", required="true")
    parser.add_argument(
        "--outdir", dest="outdir", help="output directory", required="true"
    )
    return parser.parse_args()


def yamlparser(filepath):
    with open(filepath) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def run():
    args = get_arguments()
    imgdir = args.imgdir
    colmap_dir = args.outdir
    camparams = yamlparser(args.param)["camera"]
    if not os.path.exists(colmap_dir):
        os.makedirs(colmap_dir)

    feature_extractor_params = colmap_params.getFeatureExtractorParams()
    feature_extractor_params["database_path"] = os.path.join(
        colmap_dir, "database.db")
    feature_extractor_params["image_path"] = imgdir
    feature_extractor_params["ImageReader"]["camera_model"] = "OPENCV"

    camera_params_str = "{},{},{},{},{:.9f},{:.9f},{:.9f},{:.9f}".format(
        camparams["fx"],
        camparams["fy"],
        camparams["cx"],
        camparams["cy"],
        camparams["k1"],
        camparams["k2"],
        camparams["p1"],
        camparams["p2"],
    )
    feature_extractor_params["ImageReader"]["camera_params"] = camera_params_str

    feature_extractor_config_fn = os.path.join(
        colmap_dir, "feature_extractor_config.ini"
    )
    util_colmap.write_feature_extractor_config(
        feature_extractor_params, feature_extractor_config_fn
    )

    sequential_matcher_params = colmap_params.getSequentialMatcherParams()
    sequential_matcher_params["database_path"] = os.path.join(
        colmap_dir, "database.db")
    sequential_matcher_params["SequentialMatching"]["vocab_tree_path"] = os.path.join(
        colmap_dir, "vocab_tree_flickr100K_words256K.bin"
    )

    sequential_matcher_config_fn = os.path.join(
        colmap_dir, "sequential_matcher_config.ini"
    )
    util_colmap.write_sequential_matcher_config(
        sequential_matcher_params, sequential_matcher_config_fn
    )

    mapper_params = colmap_params.getMapperParams()
    mapper_params["database_path"] = os.path.join(colmap_dir, "database.db")
    mapper_params["image_path"] = imgdir
    mapper_params["output_path"] = os.path.join(colmap_dir, "output")

    mapper_config_fn = os.path.join(colmap_dir, "mapper_config.ini")
    util_colmap.write_mapper_config(mapper_params, mapper_config_fn)
    if "vocab_tree_flickr100K_words256K.bin" not in os.listdir(colmap_dir):
        print("\nDownloading pre-trained vocabulary tree for loop closing ...")
        url = "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin"
        r = requests.get(url, allow_redirects=True)
        open(
            os.path.join(
                colmap_dir, "vocab_tree_flickr100K_words256K.bin"), "wb"
        ).write(r.content)

    dir_colmap_output = os.path.join(colmap_dir, "output")
    if not os.path.exists(dir_colmap_output):
        os.makedirs(dir_colmap_output)

    print("\nA new COLMAP project: %s was successfully created.\n" % colmap_dir)
    print("To run SfM, execute sequentially the following commands")
    print(
        "\tcolmap database_creator --database_path {}".format(
            os.path.join(colmap_dir, "database.db")
        )
    )
    print(
        "\tcolmap feature_extractor --project_path {}".format(
            feature_extractor_config_fn
        )
    )
    print(
        "\tcolmap sequential_matcher --project_path {}".format(
            sequential_matcher_config_fn
        )
    )
    print("\tcolmap mapper --project_path {}".format(mapper_config_fn))

    print("To print statistics after SfM is completed, run")
    print(
        "\tcolmap model_analyzer --path {}".format(
            os.path.join(dir_colmap_output, "0"))
    )


if __name__ == "__main__":
    run()
