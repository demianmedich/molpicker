import unittest
from pprint import pprint

from torch.utils.data import DataLoader

from molpicker.utils.data import load_coco_detection_dataset, \
    CollateCocoDetectionOutput


class CocoDatasetTestCase(unittest.TestCase):
    def test_coco_dataset(self):
        ds = load_coco_detection_dataset(
            "C:\\Users\\demianmedich\\data\\coco2017\\val2017",
            "C:\\Users\\demianmedich\\data\\coco2017\\annotations\\instances_val2017.json",
        )
        data = ds[0]
        print(data)

        dataloader = DataLoader(ds, batch_size=2,
                                collate_fn=CollateCocoDetectionOutput())
        iterator = iter(dataloader)
        data = next(iterator)
        pprint(data)


if __name__ == '__main__':
    unittest.main()
