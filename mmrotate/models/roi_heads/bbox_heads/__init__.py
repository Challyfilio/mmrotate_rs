# Copyright (c) OpenMMLab. All rights reserved.
from .convfc_rbbox_head import (RotatedConvFCBBoxHead,
                                RotatedKFIoUShared2FCBBoxHead,
                                RotatedShared2FCBBoxHead)
from .gv_bbox_head import GVBBoxHead
from .rotated_bbox_head import RotatedBBoxHead
from .clip_convfc_rbbox_head import (ClipRotatedConvFCBBoxHead,
                                     ClipRotatedKFIoUShared2FCBBoxHead,
                                     ClipRotatedShared2FCBBoxHead)
from .clip_convfc_rbbox_head_2 import (ClipRotatedConvFCBBoxHead2,
                                       ClipRotatedKFIoUShared2FCBBoxHead2,
                                       ClipRotatedShared2FCBBoxHead2)
from .prototype_convfc_rbbox_head import (PrototypeRotatedConvFCBBoxHead,
                                          PrototypeRotatedKFIoUShared2FCBBoxHead,
                                          PrototypeRotatedShared2FCBBoxHead)
from .adapter_convfc_rbbox_head import (AdapterRotatedConvFCBBoxHead,
                                        AdapterRotatedKFIoUShared2FCBBoxHead,
                                        AdapterRotatedShared2FCBBoxHead)

__all__ = [
    'RotatedBBoxHead', 'RotatedConvFCBBoxHead', 'RotatedShared2FCBBoxHead',
    'GVBBoxHead', 'RotatedKFIoUShared2FCBBoxHead', 'ClipRotatedConvFCBBoxHead', 'ClipRotatedKFIoUShared2FCBBoxHead',
    'ClipRotatedShared2FCBBoxHead', 'ClipRotatedConvFCBBoxHead2', 'ClipRotatedKFIoUShared2FCBBoxHead2',
    'ClipRotatedShared2FCBBoxHead2', 'PrototypeRotatedConvFCBBoxHead', 'PrototypeRotatedKFIoUShared2FCBBoxHead',
    'PrototypeRotatedShared2FCBBoxHead'
]
