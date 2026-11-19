import numpy as np
import pytest
from PIL import Image

from decon.completion import prepare_completion, restore_completion


def test_edge_crop_padding_and_exact_visible_preservation():
    pixels = np.full((80, 120, 3), 255, np.uint8)
    human = np.zeros((80, 120), bool)
    human[0:60, 0:15] = True
    pixels[human] = [30, 80, 100]
    completion = np.zeros_like(human)
    completion[25:35, 15:20] = True
    original = Image.fromarray(pixels)
    crop, mask, box = prepare_completion(original, human, completion)
    assert crop.width == crop.height
    assert box[0] < 0 and box[1] < 0
    assert crop.getpixel((0, 0)) == (255, 255, 255)
    assert (np.asarray(mask)[:, :, 0] > 0).sum() > completion.sum()
    result = np.asarray(restore_completion(Image.new('RGB', (640, 640), 'red'), original, completion, box))
    assert np.array_equal(result[~completion], pixels[~completion])
    assert (result[completion] == [255, 0, 0]).all()


def test_empty_crop_is_rejected():
    mask = np.zeros((10, 10), bool)
    with pytest.raises(ValueError, match='empty'):
        prepare_completion(Image.new('RGB', (10, 10)), mask, mask)
