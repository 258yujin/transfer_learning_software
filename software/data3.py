from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import logging
import random
import skimage.io
import skimage.transform
import tensorflow as tf
from skimage import morphology, measure



class myAugmentation(object):
    """
	A class used to augmentate image
	Firstly, read train image and label seperately, and then merge them together for the next process
	Secondly, use keras preprocessing to augmentate image
	Finally, seperate augmentated image apart into train image and label
	"""

    def __init__(self, train_path="train", label_path="label", merge_path="merge", aug_merge_path="aug_merge",
                 aug_train_path="aug_train", aug_label_path="aug_label", img_type="tif"):

        """
		Using glob to get all .img_type form path
		"""

        self.train_imgs = glob.glob(train_path + "/*." + img_type)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)
        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)
        self.datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')#'''数据增强（shear_range: 剪切强度 zoom_range放缩操作）'''

    def Augmentation(self):

        """
		Start augmentation.....
		"""
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0
        for i in range(len(trains)):
            img_t = load_img(path_train + "/" + str(i) + "." + imgtype)
            img_l = load_img(path_label + "/" + str(i) + "." + imgtype)
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)
            x_t[:, :, 2] = x_l[:, :, 0]
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge + "/" + str(i) + "." + imgtype)
            img = x_t
            img = img.reshape((1,) + img.shape)#'''增加一维'''
            savedir = path_aug_merge + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            self.doAugmentate(img, savedir, str(i))

    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):

        """
		augmentate one image
		"""
        datagen = self.datagen
        i = 0
        for batch in datagen.flow(img,
                                  batch_size=batch_size,
                                  save_to_dir=save_to_dir,
                                  save_prefix=save_prefix,
                                  save_format=save_format):
            i += 1
            if i > imgnum:
                break

    def splitMerge(self):

        """
		split merged image apart
		"""
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_label = self.aug_label_path
        for i in range(self.slices):
            path = path_merge + "/" + str(i)
            train_imgs = glob.glob(path + "/*." + self.img_type)
            savedir = path_train + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            savedir = path_label + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            for imgname in train_imgs:
                midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)]
                img = cv2.imread(imgname)
                img_train = img[:, :, 2]  # cv2 read image rgb->bgr
                img_label = img[:, :, 0]
                cv2.imwrite(path_train + "/" + str(i) + "/" + midname + "_train" + "." + self.img_type, img_train)
                cv2.imwrite(path_label + "/" + str(i) + "/" + midname + "_label" + "." + self.img_type, img_label)

    def splitTransform(self):

        """
		split perspective transform images
		"""
        # path_merge = "transform"
        # path_train = "transform/data/"
        # path_label = "transform/label/"
        path_merge = "deform/deform_norm2"
        path_train = "deform/train/"
        path_label = "deform/label/"
        train_imgs = glob.glob(path_merge + "/*." + self.img_type)
        for imgname in train_imgs:
            midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)]
            img = cv2.imread(imgname)
            img_train = img[:, :, 2]  # cv2 read image rgb->bgr
            img_label = img[:, :, 0]
            cv2.imwrite(path_train + midname + "." + self.img_type, img_train)
            cv2.imwrite(path_label + midname + "." + self.img_type, img_label)


class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path=".\\data1\\train dataC\\data",
                 label_path=".\\data1\\train dataC\\label", test_path=".\\data1\\val dataC\\data", npy_path="./npydata",
                 img_type="png"):

        """
		
		"""

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self):
        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)
        imgs = glob.glob(self.data_path + "/*." + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            img = load_img(self.data_path + "\\" + midname, grayscale=True)
            label = load_img(self.label_path + "\\" + midname, grayscale=True)
            img = img_to_array(img)
            label = img_to_array(label)
            # new add,in order to reduce rows and cols
            # img = img[0:1248, 0:1248, :]
            # label = label[0:1248, 0:1248, :]
            # img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            # label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            # img = np.array([img])
            # label = np.array([label])
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)
        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.float32)
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            img = load_img(self.test_path + "\\" + midname, grayscale=True)
            img = img_to_array(img).astype('uint8')
            # img = skimage.transform.resize(img, (1248, 1248,1), mode='wrap') * 255
            img = img[0:1248,0:1248,:]
            img = (img - 128) / 34
            # new add,in order to reduce rows and cols
            # img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            # img = np.array([img])
            imgdatas[i] = img
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        # mean = imgs_train.mean(axis = 0)
        # imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        # imgs_test = imgs_test.astype('float32')
        # mean = imgs_test.mean(axis = 0)
        # imgs_test -= mean
        return imgs_test


def load_image_gt(dataset, image_id, augment=False):
    # Load image and mask
    image = skimage.io.imread(os.path.join(dataset,'data\\'+str(image_id).zfill(4)+'.tif'),0)
    image1 = skimage.io.imread(os.path.join(dataset,'data1\\'+str(image_id).zfill(4)+'.tif'),0)#'''RGB   cv2是gbr'''

    images = np.zeros((image.shape[0], image.shape[1], 2))
    # images = np.zeros((image1.shape[0], image1.shape[1], 1))
    images[:, :, 0] = image
    images[:, :, 1] = image1
    image = images
    image = cv2.resize(image, (512, 512))
    # image = image.reshape(512,512,1)
    # image = skimage.transform.resize(image,(512,512),mode='wrap')

    mask = skimage.io.imread(os.path.join(dataset,'label\\'+str(image_id).zfill(4)+'.tif'))

    mask = cv2.resize(mask, (512, 512))
    mask = mask > 0

    # if mask.ndim == 3:
    #     mask = mask[:,:,0:1]
    image = image.astype('float32')
    mask = mask.astype('float32')

    # Random horizontal flips.
    if augment:
        if random.randint(0, 3) == 0:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        else:
            if random.randint(0, 2) == 0:
                image = np.flipud(image)
                mask = np.flipud(mask)

            else:
                image = image
                mask = mask


        if random.randint(0, 4) == 0:
            image = np.rot90(image, 2)
            mask = np.rot90(mask, 2)

        else:
            if random.randint(0, 3) == 1:
                image = np.rot90(image, 1)
                mask = np.rot90(mask, 1)

            else:
                if random.randint(0, 2) == 1:
                    image = np.rot90(image, 2)
                    mask = np.rot90(mask, 2)

                else:
                    image = image
                    mask = mask
     ## 数据扩充： 加噪声
        if np.random.random() < 0.3:
            muti_noise = np.random.normal(1, 0.001, (image.shape[0], image.shape[1], image.shape[2]))
            image *= muti_noise
        if np.random.random() < 0.3:
            add_noise = np.random.normal(0, 0.01, (image.shape[0], image.shape[1], image.shape[2]))
            image += add_noise


        if np.random.randint(2, 5) == 3:
            data = image.copy()
            old_min = data.min()
            old_max = data.max()
            scale = np.random.normal(0.5, 0.1)
            center = np.random.normal(1.2, 0.2)
            data = scale * (data - old_min) + 0.5 * scale * center * (old_max - old_min) + old_min
            # image = np.concatenate((data, data, data), axis=2)
            image = data

    return image, mask

# input_size = (512, 512, 1)
def data_generator(dataset, length, shuffle=True, augment=True,
                   batch_size=1):
    b = 0  # batch item index
    image_index = -1
    # image_ids = np.copy(dataset.image_ids)
    image_ids = np.arange(1, length+1, step=1)
    error_count = 0
    image_id = 0
    restart = True
    if shuffle:
        np.random.shuffle(image_ids)
    while restart:
        for i in range(len(image_ids)):
            image_id = image_ids[i]
            image, gt_masks, = load_image_gt(dataset, image_id, augment=augment)
            mean = np.mean(image)
            std = np.std(image)
            image -= mean
            image /= std
            if b == 0:
                batch_images = np.zeros(
                    (batch_size, image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
                # batch_gt_masks = np.zeros(
                #     (batch_size, image.shape[0], image.shape[1], 2))
                batch_gt_masks = np.zeros(
                    (batch_size, image.shape[0], image.shape[1], 1))

            # If more instances than fits in the array, sub-sample from them.
            # Add to batch
            batch_images[b, :, :, :] = image
            # gt_masks/=255
            batch_gt_masks[b, :, :, 0] = gt_masks

            # batch_gt_masks[b, :, :, 0] = gt_masks/255
            # batch_gt_masks[b,:,:,0] = 1-gt_masks
            # cv2.imshow('mask_train',(batch_gt_masks[b,:,:,0]*255).astype('uint8'))
            # cv2.waitKey(0)
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = batch_images
                # outputs = [batch_gt_masks,batch_gt_masks,batch_gt_masks,batch_gt_masks,batch_gt_masks,batch_gt_masks]
                outputs = batch_gt_masks

                yield inputs, outputs

                # start a new batch
                b = 0




    # Keras requires a generator to run indefinately.
    # while True:
    #     try:
    #         # Increment index to pick next image. Shuffle if at the start of an epoch.
    #         image_index = (image_index + 1) % len(image_ids)
    #
    #
    #         # Get GT bounding boxes and masks for image.
    #         image_id = image_ids[image_index]
    #         # image_big = skimage.io.imread(os.path.join(dataset,'data\\'+str(image_id).zfill(4)+'.tif'))
    #         # # image_big = image_big.reshape(image_big.shape[0],image_big.shape[1],1)
    #         # gt_masks_big = skimage.io.imread(os.path.join(dataset,'label\\'+str(image_id).zfill(4)+'.tif'))
    #         # # gt_masks_big = gt_masks_big.reshape(gt_masks_big.shape[0],gt_masks_big.shape[1],1)
    #         # gt_masks_big = gt_masks_big > 0
    #         # image_big = image_big.astype('float32')
    #         # gt_masks_big = gt_masks_big.astype('float32')
    #         # mean = np.mean(image_big)
    #         # std = np.std(image_big)
    #         # image_big -= mean
    #         # image_big /= std
    #
    #         # print('image_id:'+str(image_id))
    #         # x_start = int(np.random.uniform(1, image_big.shape[0] - input_size[0], 1))
    #         # y_start = int(np.random.uniform(1, image_big.shape[1] - input_size[1], 1))
    #         # image_clip = image_big[x_start:x_start+input_size[0], y_start:y_start+input_size[1]]
    #         # gt_masks_clip = gt_masks_big[x_start:x_start+input_size[0], y_start:y_start+input_size[1]]
    #         image, gt_masks = load_image_gt(dataset,image_id,augment=augment)
    #
    #         # image = tf.convert_to_tensor(image)
    #         # gt_masks = tf.convert_to_tensor(gt_masks)
    #
    #         # data preprocess
    #         # image1 = morphology.remove_small_objects(image < 1, min_size=1000, connectivity=2, in_place=False)
    #         #
    #         # NANZero = image[np.where(image1 == 0)]
    #         mean = np.mean(image)
    #         std = np.std(image)
    #         # # std = 20
    #         image -= mean
    #         image /= std
    #
    #         # image /= 255.
    #
    #         # Init batch arrays  批处理
    #         if b == 0:
    #             batch_images = np.zeros(
    #                 (batch_size, image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
    #             # batch_gt_masks = np.zeros(
    #             #     (batch_size, image.shape[0], image.shape[1], 2))
    #             batch_gt_masks = np.zeros(
    #                  (batch_size, image.shape[0], image.shape[1], 1))
    #
    #         # If more instances than fits in the array, sub-sample from them.
    #         # Add to batch
    #         batch_images[b, :, :, :] = image
    #         # gt_masks/=255
    #         batch_gt_masks[b, :, :, 0] = gt_masks
    #         # batch_gt_masks[b, :, :, 0] = gt_masks/255
    #         # batch_gt_masks[b,:,:,0] = 1-gt_masks
    #         # cv2.imshow('mask_train',(batch_gt_masks[b,:,:,0]*255).astype('uint8'))
    #         # cv2.waitKey(0)
    #         b += 1
    #
    #         # Batch full?
    #         if b >= batch_size:
    #             inputs = [batch_images]
    #             outputs = [batch_gt_masks]
    #             yield inputs, outputs
    #
    #             # start a new batch
    #             b = 0
    #     except (GeneratorExit, KeyboardInterrupt):
    #         raise
    #     except:
    #         # Log it and skip the image
    #         logging.exception("Error processing image {}".format(
    #             image_id))
    #         error_count += 1
    #         if error_count > 5:
    #             raise


if __name__ == "__main__":
    # aug = myAugmentation()
    # aug.Augmentation()
    # aug.splitMerge()
    # aug.splitTransform()
    mydata = dataProcess(512, 512)
    # mydata.create_train_data()
    mydata.create_test_data()
# imgs_train,imgs_mask_train = mydata.load_train_data()
# print imgs_train.shape,imgs_mask_train.shape
