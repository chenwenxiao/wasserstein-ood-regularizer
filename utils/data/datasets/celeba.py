import os
from typing import *

import mltk
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from mltk.data import MapperDataStream

from ..types import *
from .base import *
from .utils import *

__all__ = [
    'CelebA'
]


class BaseCelebA(DataSet):
    """Base class for loading the original CelebA and CelebA-HQ dataset."""

    root_dir: str
    split_ids: Dict[str, np.ndarray]

    def __init__(self,
                 root_dir: str,
                 val_split: Optional[float] = None):
        """
        Construct a new CelebA dataset.

        Args:
            root_dir: The root directory of the CelebA dataset
                (containing the standard images and the patched HQ images).
            val_split: If specified, will merge the original train
                and validation sets, shuffle them, and re-partition
                the train and validation sets according to this portion.
        """
        root_dir = os.path.abspath(root_dir)
        partition_file = os.path.join(
            root_dir, 'Eval/list_eval_partition.txt')
        df = pd.read_csv(partition_file,
                         sep=' ', header=None,
                         names=['file_name', 'set_id'],
                         dtype={'file_name': str, 'set_id': int},
                         engine='c')

        train_ids = np.asarray(df[df['set_id'] == 0]['file_name'])
        val_ids = np.asarray(df[df['set_id'] == 1]['file_name'])
        test_ids = np.asarray(df[df['set_id'] == 2]['file_name'])
        assert (len(train_ids) == 162770)
        assert (len(val_ids) == 19867)
        assert (len(test_ids) == 19962)

        if val_split is not None:
            if val_split > 0.:
                train_ids, val_ids = mltk.utils.split_numpy_array(
                    np.concatenate([train_ids, val_ids]),
                    portion=val_split,
                )
            else:
                train_ids = np.concatenate([train_ids, val_ids])
                val_ids = None

        split_ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}
        if split_ids['val'] is None:
            split_ids.pop('val')

        splits = {k: SplitInfo(data_count=len(v)) for k, v in split_ids.items()}
        slots = {
            'id': ArrayInfo(dtype='str', shape=[]),
            'x': self.get_image_array_info(),
        }

        super().__init__(splits, slots)
        self.root_dir = root_dir
        self.split_ids = split_ids

    def get_image_array_info(self) -> ArrayInfo:
        raise NotImplementedError()

    def load_image(self, image_id: str) -> np.ndarray:
        raise NotImplementedError()

    def load_items(self,
                   image_ids: np.ndarray,
                   slots: Sequence[str]) -> Tuple[np.ndarray, ...]:
        slot_arrays = [[] for _ in slots]
        for image_id in image_ids:
            for slot, target in zip(slots, slot_arrays):
                if slot == 'x':
                    target.append(self.load_image(image_id))
                elif slot == 'y':
                    # TODO: finish this
                    raise NotImplementedError()
                elif slot == 'id':
                    target.append(image_id)
                else:
                    raise RuntimeError(f'Unknown slot: {slot!r}')

        ret = []
        for i, arrays in enumerate(slot_arrays):
            ret.append(np.stack(arrays, axis=0))
        return tuple(ret)

    def _sample(self,
                split: str,
                slots: Sequence[str],
                n: int,
                with_replacement: bool = True) -> Tuple[mltk.Array, ...]:
        data_count = self.splits[split].data_count
        indexes = arg_sample(data_count, n, with_replacement)
        image_ids = self.split_ids[split][indexes]
        return self.load_items(image_ids, slots)

    def _get_arrays(self,
                    split: str,
                    slots: Sequence[str]) -> Tuple[mltk.Array, ...]:
        return self.load_items(self.split_ids[split], slots)

    def _get_stream(self,
                    split: str,
                    slots: Sequence[str],
                    batch_size: int,
                    shuffle: bool = False,
                    skip_incomplete: bool = False) -> mltk.DataStream:
        def id_to_arrays(image_ids):
            return self.load_items(image_ids, slots)

        stream = mltk.DataStream.arrays(
            [self.split_ids[split]], batch_size=batch_size, shuffle=shuffle,
            skip_incomplete=skip_incomplete
        )
        stream = MapperDataStream(
            source=stream,
            mapper=id_to_arrays,
            batch_size=batch_size,
            array_count=len(slots),
            data_shapes=tuple(self.slots[slot].shape for slot in slots),
            data_length=self.splits[split].data_count,
        )

        return stream


class CelebA(BaseCelebA):

    def get_image_array_info(self) -> ArrayInfo:
        return ArrayInfo(
            dtype='uint8',
            shape=[218, 178, 3],  # TODO: or 178, 218?
            is_discrete=True,
            min_val=0,
            max_val=255,
            n_discrete_vals=256,
            bit_depth=8,
        )

    def load_image(self, image_id: str) -> np.ndarray:
        image_path = os.path.join(
            self.root_dir, f'Img/img_align_celeba/{image_id}')
        im = PILImage.open(image_path)
        im_arr = im_bytes = None
        try:
            width, height = im.size
            im_bytes = im.tobytes()
            im_arr = np.frombuffer(im_bytes, dtype=np.uint8). \
                reshape((height, width, 3))
            return np.copy(im_arr)
        finally:
            im.close()
            del im_arr
            del im_bytes
            del im
