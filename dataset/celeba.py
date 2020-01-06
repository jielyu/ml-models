import os, random
import pandas as pd
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import matplotlib
import matplotlib.pyplot as plt

def check_path(path_name):
    if not os.path.exists(path_name):
        raise ValueError('not exist path:{}'.format(path_name))


def get_and_split_dataset(dataset_dir):
    # get path and check existence
    attr_path = os.path.join(dataset_dir, 'Anno', 'list_attr_celeba.txt')
    check_path(attr_path)
    partition_path = os.path.join(dataset_dir, 'Eval', 'list_eval_partition.txt')
    check_path(partition_path)
    img_dir = os.path.join(dataset_dir, 'Img', 'img_align_celeba') 
    check_path(img_dir)
    # read partition info
    part_df = pd.read_csv(partition_path, sep='\s+', skiprows=0, header=None)
    part_df.columns = ['filename', 'partition']
    part_df = part_df.set_index('filename')
    #part_df.head(5)
    # read attr info
    attr_df = pd.read_csv(attr_path, sep='\s+', skiprows=1)
    attr_df[attr_df == -1] = 0
    #attr_df.head()
    # merge partition and attribution data frame
    df = attr_df.merge(part_df, left_index=True, right_index=True)
    # split into train, val, test partition
    train_df = df.loc[df['partition'] == 0].drop(['partition'], axis=1)
    val_df = df.loc[df['partition'] == 1].drop(['partition'], axis=1)
    test_df = df.loc[df['partition'] == 2].drop(['partition'], axis=1)
    train_df, val_df, test_df = train_df[['Male']], val_df[['Male']], test_df[['Male']]
    return img_dir, train_df, val_df, test_df

def center_crop_index(origin_len, crop_len):
    tmp = int(origin_len/2.0 - crop_len/2.0)
    c_start = tmp if tmp >= 0 else 0
    tmp = c_start + crop_len
    c_end = tmp if tmp <= origin_len else origin_len
    return c_start, c_end

def center_crop(img, crop_size):
    h_start, h_end = center_crop_index(img.shape[0], crop_size[1])
    c_start, c_end = center_crop_index(img.shape[1], crop_size[0])
    return img[h_start:h_end, c_start:c_end,...]


class Dataset:

    def __init__(self, img_dir, df, batch_size=32, target_shape=(128, 128, 3), shuffle=False):
        self.img_dir = img_dir
        self.img_names = df.index.values
        self.labels = df.values
        self.attrs = df.columns.tolist()
        self.batch_size =batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle

    def get_dataset_size(self):
        return len(self.img_names)

    def get_cate_num(self):
        return 2

    def get_attr_num(self):
        return len(self.attrs)

    def get_target_shape(self):
        return self.target_shape

    def generate(self, epoch_stop=False):
        # compute number of batches in an epoch
        num_samples = self.get_dataset_size()
        all_idx = list(range(num_samples))
        num_batch = num_samples // self.batch_size
        if num_batch * self.batch_size < num_samples:
            num_batch += 1
        # generate samples batch by batch
        tgt_size = (self.target_shape[1], self.target_shape[0])
        crop_size = (178, 178)
        while True:
            if self.shuffle is True:
                random.shuffle(all_idx)
            for i in range(num_batch):
                # compute start and end index on all_idx
                start_idx = i * self.batch_size
                tmp_idx = (i+1) * self.batch_size
                end_idx = tmp_idx if tmp_idx <= num_samples else num_samples
                # get a batch of samples
                batch_images = []
                batch_labels = []
                for idx in range(start_idx, end_idx):
                    index = all_idx[idx]
                    # get image path
                    img_name = self.img_names[index]
                    img_path = os.path.join(self.img_dir, img_name)
                    # read image from file by opencv with BGR channels
                    img = cv2.imread(img_path)
                    # center crop
                    img = center_crop(img, crop_size)
                    # resize to uniform size
                    img = cv2.resize(img, tgt_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # normalize
                    img = img / 255.0
                    batch_images.append(img)
                    # get label info
                    batch_labels.append(self.labels[index, ...])
                images = np.array(batch_images, dtype=np.float32)
                labels = np.array(batch_labels)
                # one-hot encoding
                oh_labels = np.zeros((labels.shape[0], labels.shape[1], self.get_cate_num()), dtype=np.float32)
                for i in range(oh_labels.shape[0]):
                    for j in range(oh_labels.shape[1]):
                        oh_labels[i, j, labels[i, j]] = 1
                yield images, np.squeeze(oh_labels)
            if epoch_stop is True:
                break

def create_dataset(dataset_dir, batch_size=32, target_shape=(128,128,3)):
    img_dir, train_df, val_df, test_df = get_and_split_dataset(dataset_dir)
    trainset = Dataset(img_dir, train_df, batch_size, target_shape, True)
    valset = Dataset(img_dir, val_df, batch_size, target_shape, False)
    testset = Dataset(img_dir, test_df, batch_size, target_shape, False)
    return trainset, valset, testset


def main():
    # create dataset
    dataset_dir = os.path.expanduser('~/Database/Dataset/CelebA-dataset')
    trainset, valset, testset = create_dataset(dataset_dir)

    # check generator
    cnt = 0
    for images, labels in trainset.generate(epoch_stop=True):
        cnt += 1
        print(cnt, images.shape, labels.shape)
        print(images.dtype, labels.dtype, labels[0,...])
        plt.imshow(images[0, ...])
        plt.show()
        break

if __name__ == '__main__':
    main()
