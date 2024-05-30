import os
import sys
import json
import datetime
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.client import device_lib
import skimage.draw
from PIL.ImageDraw import ImageDraw, Image
import PIL

from mrcnn import utils
from mrcnn.visualize import display_instances, display_top_masks

class MTrackingDataset(utils.Dataset):
    def load_data(self, annotations_json, images_dir):
        print(annotations_json)
        json_file = open(annotations_json)
        coco_json = json.load(json_file)
        json_file.close()
        print(type(coco_json))

        source_name = "coco_like"
        for category in coco_json["categories"]:
            class_id = category["id"]
            class_name = category["name"]
            if class_id < 1:
                print("error")
                return
            self.add_class(source_name, class_id, class_name)

        annotations = {}
        for annotation in coco_json["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        seen_images = {}
        for image in coco_json["images"]:
            image_id = image["id"]
            if image_id in seen_images:
                print("warning")
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image["file_name"]
                    image_width = image["width"]
                    image_height = image["height"]
                except KeyError as key:
                    print(key)

                image_path = os.path.abspath(os.path.join(images_dir,image_file_name))
                image_annotations = annotations[image_id]

                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info["annotations"]
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation["category_id"]
            mask = Image.new('1',(image_info['width'], image_info['height']))

            mask_draw = PIL.ImageDraw.ImageDraw(mask,'1')
            for segmentation in annotation["segmentation"]:
                mask_draw.polygon(segmentation,fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

def train(model):
    dataset_train = MTrackingDataset()
    dataset_train.load_data(args.dataset, "Dataset/train")
    dataset_train.prepare()

    dataset_val = MTrackingDataset()
    dataset_val.load_data(args.dataset, "Dataset/train")
    dataset_val.prepare()

    print("TRAINING NETWORK HEADS")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')