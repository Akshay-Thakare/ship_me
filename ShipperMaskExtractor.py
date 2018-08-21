import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MASK_FILE_PATH = os.path.join("/Users/ast/temp/ml_self_learn/sample_ship", "train_ship_segmentations.csv")

# class ShipperMaskExtractor:

def rle_decode(mask_rle, shape=(768,768)):
    '''
    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    :return: numpy array, 1 - mask, 0 - background
    '''

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.int16)

    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)

    return np.expand_dims(all_masks, -1)

masks = pd.read_csv(MASK_FILE_PATH)
# print(masks.shape[0], 'masks found')
# print(masks['ImageId'].value_counts().shape[0])
# print(masks.head())

img_0 = masks_as_image(masks.query('ImageId=="0a7f650ee.jpg"')['EncodedPixels'])
print(np.sum(img_0, axis=(0,1)))
# plt.imshow(img_0[:,:,0])
# plt.show()