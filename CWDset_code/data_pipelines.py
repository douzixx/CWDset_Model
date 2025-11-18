import numpy as np
import rasterio
from typing import Dict, Tuple, Optional, Union
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

# 导入 cv2 相关的库（最佳实践是放在文件顶部）
from cv2 import resize, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC


# ---------------------------------
# 1. TIF 图像/标签加载
# ---------------------------------

@TRANSFORMS.register_module()
class LoadTiffImageFromFile(BaseTransform):
    """Load 4-band TIFF image from file."""

    def __init__(self,
                 to_float32: bool = True,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2') -> None:
        self.to_float32 = to_float32

    def transform(self, results: Dict) -> Dict:
        filename = results['img_path']
        with rasterio.open(filename) as src:
            img = src.read()  # (bands, H, W)
            img = np.transpose(img, (1, 2, 0))  # (H, W, bands)

        if self.to_float32:
            img = img.astype(np.float32) / 65535.0

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['img_fields'] = ['img']
        return results


@TRANSFORMS.register_module()
class LoadTiffAnnotations(BaseTransform):
    """Load annotations for semantic segmentation from TIFF file."""

    def __init__(self,
                 reduce_zero_label: bool = False,
                 imdecode_backend: str = 'cv2') -> None:
        self.reduce_zero_label = reduce_zero_label

    def transform(self, results: Dict) -> Dict:
        filename = results['seg_map_path']
        with rasterio.open(filename) as src:
            gt_semantic_seg = src.read(1)  # 读取第一个波段

        if self.reduce_zero_label:
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'] = ['gt_seg_map']
        return results


# ---------------------------------
# 2. TIF 数据增强
# ---------------------------------

@TRANSFORMS.register_module()
class RandomCropTiff(BaseTransform):
    """Random crop for TIFF images and segmentation maps."""

    def __init__(self,
                 crop_size: Tuple[int, int],
                 cat_max_ratio: float = 1.0,
                 ignore_index: int = 255) -> None:
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def _get_crop_bbox(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        if h <= crop_h and w <= crop_w:
            return 0, 0, h, w

        if h <= crop_h:
            y1 = 0
        else:
            y1 = np.random.randint(0, h - crop_h + 1)

        if w <= crop_w:
            x1 = 0
        else:
            x1 = np.random.randint(0, w - crop_w + 1)

        y2 = min(y1 + crop_h, h)
        x2 = min(x1 + crop_w, w)

        return y1, x1, y2, x2

    def transform(self, results: Dict) -> Dict:
        img = results['img']
        y1, x1, y2, x2 = self._get_crop_bbox(img)

        results['img'] = img[y1:y2, x1:x2, :]
        results['img_shape'] = results['img'].shape[:2]

        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][y1:y2, x1:x2]

        return results


@TRANSFORMS.register_module()
class RandomFlipTiff(BaseTransform):
    """Random flip for TIFF images and segmentation maps."""

    def __init__(self,
                 prob: float = 0.5,
                 direction: str = 'horizontal') -> None:
        self.prob = prob
        self.direction = direction
        assert direction in ['horizontal', 'vertical', 'diagonal']

    def transform(self, results: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return results

        img = results['img']
        if self.direction == 'horizontal':
            results['img'] = np.flip(img, axis=1).copy()
        elif self.direction == 'vertical':
            results['img'] = np.flip(img, axis=0).copy()
        elif self.direction == 'diagonal':
            results['img'] = np.flip(np.flip(img, axis=0), axis=1).copy()

        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            if self.direction == 'horizontal':
                results['gt_seg_map'] = np.flip(gt_seg_map, axis=1).copy()
            elif self.direction == 'vertical':
                results['gt_seg_map'] = np.flip(gt_seg_map, axis=0).copy()
            elif self.direction == 'diagonal':
                results['gt_seg_map'] = np.flip(np.flip(gt_seg_map, axis=0), axis=1).copy()

        return results


@TRANSFORMS.register_module()
class RandomResizeTiff(BaseTransform):
    """Random resize for TIFF images and segmentation maps."""

    def __init__(self,
                 scale: Union[Tuple[float, float], list] = (0.5, 2.0),
                 ratio_range: Optional[Tuple[float, float]] = None,
                 interpolation: str = 'bilinear') -> None:
        self.scale = scale
        self.ratio_range = ratio_range
        self.interpolation = interpolation

    def _get_random_scale(self, img_shape: Tuple[int, int]) -> Tuple[int, int]:
        h, w = img_shape
        if isinstance(self.scale, tuple) and len(self.scale) == 2:
            scale_factor = np.random.uniform(self.scale[0], self.scale[1])
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
        else:
            new_h, new_w = self.scale[np.random.randint(len(self.scale))]

        if self.ratio_range is not None:
            ratio = np.random.uniform(self.ratio_range[0], self.ratio_range[1])
            new_w = int(new_w * ratio)

        return new_h, new_w

    def _resize(self, img: np.ndarray, size: Tuple[int, int],
                interpolation: str) -> np.ndarray:
        interp_dict = {
            'nearest': INTER_NEAREST,
            'bilinear': INTER_LINEAR,
            'bicubic': INTER_CUBIC
        }
        h, w = size
        return resize(img, (w, h), interpolation=interp_dict[interpolation])

    def transform(self, results: Dict) -> Dict:
        img = results['img']
        new_h, new_w = self._get_random_scale(img.shape[:2])

        results['img'] = self._resize(img, (new_h, new_w), self.interpolation)
        results['img_shape'] = (new_h, new_w)

        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            results['gt_seg_map'] = self._resize(
                gt_seg_map, (new_h, new_w), 'nearest'
            )

        return results