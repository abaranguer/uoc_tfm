import unittest
from ham10000_dataset_splitter import Ham10000DatasetSplitter


class Ham10000DatasetSplitterTestCase(unittest.TestCase):
    def test_datasets_lengths(self):
        matadata_path = '/home/albert/UOC-TFM/dataset/HAM10000_metadata'
        images_path = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/300x225/'

        print('Start')
        splitter = Ham10000DatasetSplitter(matadata_path, images_path)

        num_of_images_in_train_dataset = len(splitter.train_dataset)
        num_of_images_in_validation_dataset = len(splitter.validation_dataset)
        num_of_images_in_test_dataset = len(splitter.test_dataset)
        total = num_of_images_in_train_dataset + num_of_images_in_validation_dataset + num_of_images_in_test_dataset

        print(f'splitter.train_dataset: {num_of_images_in_train_dataset}')
        print(f'splitter.validation_dataset: {num_of_images_in_validation_dataset}')
        print(f'splitter.splitter.test_dataset: {num_of_images_in_test_dataset}')

        self.assertEqual(7010, num_of_images_in_train_dataset)
        self.assertEqual(1502, num_of_images_in_validation_dataset)
        self.assertEqual(1503, num_of_images_in_test_dataset)
        self.assertEqual(10015, total)

        print('Done!')


if __name__ == '__main__':
    unittest.main()
