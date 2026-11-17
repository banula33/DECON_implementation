import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from decon import GGDOConfig, HumanDecouplingPipeline, HumanEstimate, PinholeCamera
from decon.artifacts import save_results
from decon.official_adapters import OfficialPowerPaintInpainter, OfficialSATHMRGeometryEstimator
from decon.sat_hmr_runner import export_predictions


def test_camera_projection_and_resize():
    camera = PinholeCamera.from_resized(torch.tensor([[100., 0, 50], [0, 100., 25], [0, 0, 1]]), (50, 100), (100, 200))
    vertices = torch.tensor([[-1., -0.25, 2.], [1., 0.25, 2.]])
    assert torch.allclose(camera.project(vertices), torch.tensor([[0., 25.], [200., 75.]]))
    assert torch.allclose(camera.bounding_box(vertices, padding=0), torch.tensor([0., 25., 199., 75.]))
    with pytest.raises(ValueError, match='front'):
        camera.project(-vertices)


def test_export_and_read_sat_predictions(tmp_path):
    (tmp_path / 'main.py').touch()
    # Two queries; only the confident person should be exported.
    outputs = {'pred_confs': torch.tensor([[[0.9], [0.1]]]),
               'pred_verts': torch.randn(1, 2, 5, 3), 'pred_poses': torch.randn(1, 2, 72),
               'pred_betas': torch.randn(1, 2, 10), 'pred_transl': torch.randn(1, 2, 3),
               'pred_intrinsics': torch.tensor([[[[100., 0, 50], [0, 100., 25], [0, 0, 1]]]])}
    export_predictions(outputs, [{'img_size': torch.tensor([50, 100])}], tmp_path, 0.3)
    adapter = OfficialSATHMRGeometryEstimator(tmp_path, device='cpu')
    people = adapter.read_predictions(tmp_path, (100, 200))
    assert len(people) == 1
    assert torch.equal(people[0].vertices, outputs['pred_verts'][0, 0])
    assert people[0].camera.intrinsics[0, 0] == 200
    assert people[0].mask.shape == (100, 200)
    assert people[0].pose.shape == (24, 3)


def test_sat_subprocess_isolates_inputs_and_preserves_existing_results(tmp_path, monkeypatch):
    (tmp_path / 'main.py').touch()
    existing = tmp_path / 'demo_results'
    existing.mkdir()
    (existing / 'keep.txt').write_text('keep')
    called = []
    def run(command, **kwargs):
        from pathlib import Path
        inputs = Path(command[command.index('--input-dir') + 1])
        assert Image.open(inputs / 'input.png').size == (10, 20)
        assert kwargs['cwd'] == tmp_path
        called.append(inputs)
    monkeypatch.setattr('decon.official_adapters.subprocess.run', run)
    adapter = OfficialSATHMRGeometryEstimator(tmp_path, device='cpu')
    assert adapter.estimate(Image.new('RGB', (10, 20))) == []
    assert (existing / 'keep.txt').read_text() == 'keep'
    assert not called[0].exists()


def test_powerpaint_preserves_visible_pixels_and_resolution(tmp_path):
    adapter = OfficialPowerPaintInpainter(tmp_path, tmp_path)
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[:] = [100, 20, 30]
    human = torch.zeros(6, 8, dtype=torch.bool)
    human[2:4, 2:4] = True
    completion = torch.zeros_like(human)
    completion[4:6, 2:4] = True
    def predict(inputs, *args):
        pixels = np.asarray(inputs['image'])
        assert (pixels[0, 0] == 255).all()
        assert inputs['mask'].mode == 'RGB'
        return [Image.new('RGB', (16, 12), 'blue')], []
    adapter._controller = SimpleNamespace(predict=predict)
    result = np.asarray(adapter.inpaint(rgb, human, completion))
    assert result.shape == rgb.shape
    assert np.array_equal(result[human], rgb[human])
    assert (result[completion] == [0, 0, 255]).all()
    assert (result[~(human | completion)] == 255).all()


def test_no_completion_skips_model_loading(tmp_path, monkeypatch):
    adapter = OfficialPowerPaintInpainter(tmp_path, tmp_path)
    monkeypatch.setattr(adapter, '_load', lambda: pytest.fail('must not load PowerPaint'))
    result = adapter.inpaint(Image.new('RGB', (8, 6)), torch.ones(6, 8, dtype=torch.bool), torch.zeros(6, 8, dtype=torch.bool))
    assert result.size == (8, 6)


def test_two_person_pipeline_and_artifacts(tmp_path):
    camera = PinholeCamera(torch.tensor([[20., 0, 16.], [0, 20., 16.], [0, 0, 1.]]), (32, 32))
    vertices = torch.tensor([[-0.4, -0.5, 2.], [0.4, 0.5, 2.]])
    estimates = [HumanEstimate(vertices + torch.tensor([dx, 0., 0.]), torch.zeros(32, 32), camera) for dx in (-0.7, 0.7)]
    masks = []
    for x in (9, 23):
        mask = torch.zeros(32, 32, dtype=torch.bool)
        mask[10:19, x-3:x+3] = True
        masks.append(mask)
    pixels = np.full((32, 32, 3), 40, dtype=np.uint8)
    pixels[masks[0]] = [255, 0, 0]
    pixels[masks[1]] = [0, 255, 0]
    image_path = tmp_path / 'input.png'
    Image.fromarray(pixels).save(image_path)
    index = iter(range(2))
    def segment(image, box):
        assert isinstance(image, Image.Image)
        assert (box >= 0).all() and (box <= 31).all()
        return masks[next(index)]
    def render(v, cam):
        center = cam.project(v).mean(0)
        y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
        return torch.exp(-((x-center[0])**2 + (y-center[1])**2) / 30)
    def inpaint(image, mask, completion):
        assert (np.asarray(image)[~mask] == 255).all()
        return image
    pipeline = HumanDecouplingPipeline(
        SimpleNamespace(estimate=lambda image: estimates), SimpleNamespace(segment=segment),
        SimpleNamespace(render_mask=render), SimpleNamespace(inpaint=inpaint),
        ggdo_config=GGDOConfig(translation_steps=10, pose_shape_steps=0))
    people = pipeline.run(image_path)
    assert len(people) == 2
    for person, mask in zip(people, masks):
        assert torch.equal(person.estimate.mask, mask)
        assert not (person.completion_mask & mask).any()
        assert person.image.size == (32, 32)
    output = save_results(image_path, people, tmp_path / 'out', torch.tensor([[0, 1, 0]]))
    assert json.loads((output / 'manifest.json').read_text())['people'] == 2
    assert (output / 'person_000/completed.png').is_file()
    data = json.loads((output / 'person_001/parameters.json').read_text())
    assert data['final_mse'] <= data['initial_mse']
    with pytest.raises(FileExistsError):
        save_results(image_path, people, output, torch.tensor([[0, 1, 0]]))


def test_empty_detections_fail():
    pipeline = HumanDecouplingPipeline(SimpleNamespace(estimate=lambda _: []), None, None)
    with pytest.raises(ValueError, match='no people'):
        pipeline.run(Image.new('RGB', (10, 10)))


def test_sam_receives_rgb_and_pixel_box(tmp_path):
    from decon.official_adapters import OfficialSAM2Segmenter
    adapter = OfficialSAM2Segmenter(tmp_path, device='cpu')
    calls = []
    def set_image(rgb):
        assert torch.is_inference_mode_enabled()
        calls.append(rgb.shape)
    def predict(**kwargs):
        assert np.array_equal(kwargs['box'], np.array([1, 2, 8, 9], dtype=np.float32))
        assert kwargs['multimask_output'] is False
        return np.ones((1, 12, 10)), None, None
    adapter._predictor = SimpleNamespace(set_image=set_image, predict=predict)
    mask = adapter.segment(Image.new('RGB', (10, 12)), torch.tensor([1, 2, 8, 9]))
    assert calls == [(12, 10, 3)]
    assert mask.dtype == torch.bool and mask.shape == (12, 10)


def test_smpl_wrapper_preserves_translation_and_parameter_gradients(tmp_path, monkeypatch):
    smplx = pytest.importorskip('smplx')
    from decon.body_model import SMPLBodyModel
    class Layer(torch.nn.Module):
        faces = np.array([[0, 1, 2]])
        def forward(self, global_orient, body_pose, betas):
            assert global_orient.shape == (1, 3)
            assert body_pose.shape == (1, 69)
            assert betas.shape == (1, 10)
            return SimpleNamespace(vertices=(global_orient + body_pose[:, :3] + betas[:, :3])[:, None].expand(1, 3, 3))
    monkeypatch.setattr(smplx, 'create', lambda *a, **kw: Layer())
    body = SMPLBodyModel(tmp_path, device='cpu')
    pose = torch.zeros(24, 3, requires_grad=True)
    shape = torch.zeros(10, requires_grad=True)
    translation = torch.tensor([1., 2., 3.], requires_grad=True)
    vertices = body(pose, shape, translation)
    assert torch.equal(vertices, translation.expand(3, 3))
    vertices.sum().backward()
    assert pose.grad.abs().sum() > 0
    assert shape.grad.abs().sum() > 0
    assert translation.grad.abs().sum() > 0


def test_pipeline_runs_both_parametric_stages():
    camera = PinholeCamera(torch.tensor([[20., 0, 16.], [0, 20., 16.], [0, 0, 1.]]), (32, 32))
    pose, shape, trans = torch.zeros(24, 3), torch.zeros(10), torch.tensor([0., 0., 2.])
    def body(theta, beta, translation):
        base = torch.tensor([[-.4, -.5, 0.], [.4, .5, 0.]])
        return base * (1 + beta[0]) + theta[0] + translation
    def render(vertices, cam):
        xy = cam.project(vertices)
        center = xy.mean(0)
        spread = (xy[1, 0] - xy[0, 0]).square() / 2
        y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
        return torch.exp(-((x-center[0])**2 + (y-center[1])**2) / spread)
    vertices = body(pose, shape, trans)
    mask = render(vertices, camera) > .5
    estimate = HumanEstimate(vertices, mask, camera, pose, shape, trans)
    pipeline = HumanDecouplingPipeline(
        SimpleNamespace(estimate=lambda _: [estimate]), SimpleNamespace(segment=lambda *_: mask),
        SimpleNamespace(render_mask=render), body_model=body,
        ggdo_config=GGDOConfig(translation_steps=3, pose_shape_steps=4))
    person = pipeline.run(Image.new('RGB', (32, 32)))[0]
    assert person.optimized_pose.shape == (24, 3)
    assert person.optimized_shape.shape == (10,)
    assert len(person.losses) == 8
    assert torch.allclose(person.optimized_vertices,
                          body(person.optimized_pose, person.optimized_shape, person.optimized_translation))


def test_sat_bridge_subprocess_roundtrip(tmp_path):
    """Run the real bridge/interpreter against a lightweight upstream API fixture."""
    (tmp_path / 'main.py').touch()
    for relative in ['configs/run', 'configs/models', 'engines']:
        (tmp_path / relative).mkdir(parents=True)
    (tmp_path / 'configs/run/demo.yaml').write_text('{"model": "default"}')
    (tmp_path / 'configs/models/default.yaml').write_text('{}')
    # The fixture configs are JSON, a YAML subset; avoid requiring PyYAML for this test.
    (tmp_path / 'yaml.py').write_text('import json\ndef safe_load(stream): return json.load(stream)\n')
    (tmp_path / 'engines/__init__.py').touch()
    (tmp_path / 'engines/engine.py').write_text('''
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
    def forward(self, samples, targets):
        return dict(pred_confs=torch.tensor([[[0.9]]]),
                    pred_verts=torch.tensor([[[[-1., -1., 3.], [1., 1., 3.]]]]),
                    pred_poses=torch.zeros(1, 1, 72), pred_betas=torch.zeros(1, 1, 10),
                    pred_transl=torch.tensor([[[0., 0., 3.]]]),
                    pred_intrinsics=torch.tensor([[[[10., 0, 5.], [0, 10., 10.], [0, 0, 1.]]]]))
class Engine:
    def __init__(self, args, mode):
        assert mode == 'infer' and not args.distributed_infer
        self.unwrapped_model = Model()
        self.infer_dataloader = [([torch.zeros(3, 20, 10)], [{'img_size': torch.tensor([20, 10])}])]
''')
    adapter = OfficialSATHMRGeometryEstimator(tmp_path, device='cpu')
    people = adapter.estimate(Image.new('RGB', (20, 40)))
    assert len(people) == 1
    assert torch.equal(people[0].translation, torch.tensor([0., 0., 3.]))
    assert people[0].camera.image_size == (40, 20)
    assert people[0].camera.intrinsics[0, 0] == 20


@pytest.mark.parametrize('enabled', [True, False])
def test_powerpaint_import_preserves_caller_gradient_mode(tmp_path, monkeypatch, enabled):
    # The real upstream app has this process-wide side effect on import.
    (tmp_path / 'app.py').write_text(
        'import torch\n'
        'torch.set_grad_enabled(False)\n'
        'class PowerPaintController:\n'
        '    def __init__(self, *args): pass\n'
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.syspath_prepend(str(tmp_path))
    adapter = OfficialPowerPaintInpainter(tmp_path, tmp_path)
    with torch.set_grad_enabled(enabled):
        adapter._load()
        assert torch.is_grad_enabled() is enabled
        if enabled:
            value = torch.tensor(2., requires_grad=True)
            value.square().backward()
            assert value.grad == 4.
