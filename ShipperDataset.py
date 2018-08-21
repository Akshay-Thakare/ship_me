import os
import sys
import matplotlib
import json
import datetime
import numpy as np
import pandas as pd
import skimage.io
from imgaug import augmenters as iaa

from ShipperMaskExtractor import ShipperMaskExtractor

# Root directory of the project
ROOT_DIR = "/Users/ast/temp/ml_self_learn/sample_ship"

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class ShipperDataset(utils.Dataset):

    def init(self, ROOT_DIR):
        # Training segmentation csv --- Run length encoded
        self.masks = pd.read_csv(os.path.join(ROOT_DIR, "train_ship_segmentations.csv"))

        # Results directory
        # Save submission files here
        self.RESULTS_DIR = os.path.join(ROOT_DIR, "results")

    def load_ships(self, dataset_dir, subset):
        # Add classes. We have only one class.
        # Naming the dataset shipper, and the class shipper
        self.add_class("ships", 1, "ships")

        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add images
        for image_id in [file for file in os.listdir(ROOT_DIR+"/"+subset) if os.path.isfile(os.path.join(ROOT_DIR, subset, file))]:
            self.add_image(source="ships", image_id=image_id, path=os.path.join(ROOT_DIR, subset, image_id))

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        masks = pd.read_csv(os.path.join(ROOT_DIR, "train_ship_segmentations.csv"))
        image_id = self.image_info[image_id]["id"]
        mask = ShipperMaskExtractor().masks_as_image(masks.query('ImageId==\"'+image_id+'\"')['EncodedPixels'])
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = [data for data in self.image_info if data["id"]==image_id]
        return info[0]["path"]
