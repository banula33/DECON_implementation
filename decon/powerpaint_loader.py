"""Memory-bounded construction using the official PowerPaint v2 components."""

from pathlib import Path


def build_v2_controller(module, checkpoint_dir, dtype, local_files_only=True):
    """Load final weights directly, avoiding redundant initialized UNet copies.

    The upstream controller's prediction code is retained. BrushNet has a
    complete checkpoint, so its temporary SD initialization can be omitted.
    """
    import torch
    from accelerate import init_empty_weights, load_checkpoint_in_model
    from safetensors import safe_open

    root = Path(checkpoint_dir)
    base = root / "realisticVisionV60B1_v51VAE"
    config = module.UNet2DConditionModel.load_config(
        "runwayml/stable-diffusion-v1-5", subfolder="unet",
        local_files_only=local_files_only,
    )
    with init_empty_weights():
        template = module.UNet2DConditionModel.from_config(config)
        brushnet = module.BrushNetModel.from_unet(template, load_weights_from_unet=False)
    del template
    checkpoint = root / "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"
    expected = brushnet.state_dict()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as weights:
        if set(weights.keys()) != set(expected):
            raise ValueError("BrushNet checkpoint must contain every model tensor for direct loading")
        for key, value in expected.items():
            if tuple(weights.get_slice(key).get_shape()) != tuple(value.shape):
                raise ValueError(f"BrushNet checkpoint shape mismatch: {key}")
    del expected
    # Materialize components on the GPU as they load, rather than retaining
    # multiple full models in host RAM during construction. CPU offloading is
    # installed only after all temporary checkpoint state has been released.
    device_index = torch.cuda.current_device()
    device = f"cuda:{device_index}"
    load_checkpoint_in_model(brushnet, str(checkpoint), device_map={"": device}, dtype=dtype)
    if any(value.is_meta for value in brushnet.parameters()):
        raise ValueError("BrushNet checkpoint left uninitialized parameters")

    unet = module.UNet2DConditionModel.from_pretrained(
        str(base), subfolder="unet", torch_dtype=dtype,
        low_cpu_mem_usage=True, local_files_only=local_files_only, device_map={"": device},
    )
    encoder = module.CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder",
        torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=local_files_only,
        device_map={"": device},
    )
    pipe = module.StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
        str(base), unet=unet, brushnet=brushnet, text_encoder_brushnet=encoder,
        torch_dtype=dtype, low_cpu_mem_usage=True, safety_checker=None,
        local_files_only=local_files_only,
    )
    pipe.tokenizer = module.TokenizerWrapper(
        from_pretrained=str(base), subfolder="tokenizer", local_files_only=local_files_only,
    )
    module.add_tokens(
        tokenizer=pipe.tokenizer, text_encoder=encoder,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"], num_vectors_per_token=10,
    )
    state = torch.load(root / "PowerPaint_Brushnet/pytorch_model.bin",
                       map_location="cpu", weights_only=True, mmap=True)
    incompatible = encoder.load_state_dict(state, strict=False)
    # Transformers versions differ in whether this deterministic buffer is saved.
    allowed = {"text_model.embeddings.position_ids"}
    if set(incompatible.missing_keys) - allowed or set(incompatible.unexpected_keys) - allowed:
        raise ValueError(f"Incomplete PowerPaint text encoder checkpoint: {incompatible}")
    del state
    pipe.scheduler = module.UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload(gpu_id=device_index)

    controller = module.PowerPaintController.__new__(module.PowerPaintController)
    controller.version = "ppt-v2"
    controller.checkpoint_dir = str(root)
    controller.local_files_only = local_files_only
    controller.pipe = pipe
    return controller
