import unittest
from pprint import pprint
from typing import Tuple, Mapping

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from molpicker.utils.data import \
    load_coco_detection_dataset_for_generalized_rcnn, \
    CollateCocoDetectionOutput


class CocoDatasetTestCase(unittest.TestCase):

    def test_coco_dataset(self):
        ds = load_coco_detection_dataset_for_generalized_rcnn(
            "C:\\Users\\demianmedich\\data\\coco2017\\val2017",
            "C:\\Users\\demianmedich\\data\\coco2017\\annotations\\instances_val2017.json",
        )
        # print(data)
        rand_idx = np.random.randint(1, len(ds) - 1)
        self.assert_coco_dataset_type(ds[0])
        self.assert_coco_dataset_type(ds[-1])
        self.assert_coco_dataset_type(ds[rand_idx])

        dataloader = DataLoader(ds, batch_size=2,
                                collate_fn=CollateCocoDetectionOutput())
        iterator = iter(dataloader)
        data = next(iterator)
        pprint(data)

    def assert_coco_dataset_type(self, data):
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], Tensor))
        self.assertTrue(isinstance(data[1], dict))
        for v in data[1].values():
            self.assertTrue(isinstance(v, Tensor))


if __name__ == '__main__':
    unittest.main()
