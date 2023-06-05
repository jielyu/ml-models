# encoding: utf-8


import unittest

import torch


class TestTest(unittest.TestCase):
    """测试杂项"""

    def test_cuda_shape(self):
        """测试cuda张量的shape特点"""
        t = torch.rand(2, 3, 4)
        t = t.to("cuda")
        size = t.shape
        self.assertEqual(size[0], 2)
        self.assertEqual(size[1], 3)
        self.assertEqual(size[2], 4)


def main():
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    main()
