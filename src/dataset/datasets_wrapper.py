# encoding: utf-8

from functools import wraps
import torch
from torch.utils.data.dataset import Dataset as torchDataset
from torch.utils.data.dataloader import DataLoader as torchDataLoader
from dataset.samplers import YoloBatchSampler


class Dataset(torchDataset):
    """This class is a subclass of the base :class:`torch.utils.data.Dataset`,
    that enables on the fly resizing of the ``input_dim``.

    Args:
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
    """

    def __init__(self, input_dimension, mosaic=True):
        super().__init__()
        self.__input_dim = input_dimension[:2]
        self.enable_mosaic = mosaic

    @property
    def input_dim(self):
        """
        Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth
        for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim
        return self.__input_dim

    @staticmethod
    def mosaic_getitem(getitem_fn):
        """
        Decorator method that needs to be used around the ``__getitem__`` method. |br|
        This decorator enables the closing mosaic

        Example:
            >>> class CustomSet(ln.data.Dataset):
            ...     def __len__(self):
            ...         return 10
            ...     @ln.data.Dataset.mosaic_getitem
            ...     def __getitem__(self, index):
            ...         return self.enable_mosaic
        """

        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                self.enable_mosaic = index[0]
                index = index[1]

            ret_val = getitem_fn(self, index)

            return ret_val

        return wrapper


class DataLoader(torchDataLoader):
    """
    Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if len(args) > 5:
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        elif len(args) > 3:
            shuffle = args[2]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        else:
            if "shuffle" in kwargs:
                shuffle = kwargs["shuffle"]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                    # sampler = torch.utils.data.DistributedSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = YoloBatchSampler(
                sampler,
                self.batch_size,
                self.drop_last,
                input_dimension=self.dataset.input_dim,
            )
            # batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations =

        self.batch_sampler = batch_sampler

        self.__initialized = True

    def close_mosaic(self):
        self.batch_sampler.mosaic = False
