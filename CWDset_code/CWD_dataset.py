from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class CWDset(BaseSegDataset):
    """
    你自定义的 CWDset 数据集 "蓝图"。
    """

    METAINFO = dict(
        classes=('background', 'target'),
        palette=[[0, 0, 0], [255, 0, 0]]  # 调色板
    )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)