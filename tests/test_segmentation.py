import numpy as np
import torch
from decon.segmentation import refine_mask


def test_cleanup_preserves_detached_hand_and_thin_finger():
    mask=torch.zeros((40,40),dtype=torch.bool)
    mask[10:30,5:20]=True
    mask[10:15,28:33]=True  # disconnected visible hand
    mask[19,20:28]=True  # one-pixel-wide finger
    mask[15,10]=False  # pinhole
    mask[2,2]=True  # stray pixel
    out=refine_mask(mask)
    assert not out[2,2] and out[15,10]
    assert out[12,30] and out[19,24]
    assert out[0].sum()==0 and out.dtype==torch.bool


def test_prompts_protect_small_components_and_holes():
    mask=torch.zeros((30,30),dtype=torch.bool)
    mask[10:25,10:25]=True
    mask[3:5,3:5]=True
    mask[16:18,16:18]=False
    out=refine_mask(mask,sigma=0,points=[[3,3],[16,16]],labels=[1,0])
    assert out[3:5,3:5].all()
    assert not out[16:18,16:18].any()


def test_disabled_cleanup_and_empty_masks_unchanged():
    mask=torch.tensor([[1,0,1],[0,1,0]],dtype=torch.bool)
    assert torch.equal(refine_mask(mask,sigma=0,min_area=0),mask)
    assert not refine_mask(torch.zeros(10,10,dtype=torch.bool)).any()
