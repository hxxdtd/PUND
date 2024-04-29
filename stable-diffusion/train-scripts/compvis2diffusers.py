from convertModels import savemodelDiffusers

savemodelDiffusers(ckpt_path="stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt", # CompVis checkpoint path
                   dump_path="stable-diffusion/diffusers_ckpt/ORI/unet/diffusion_pytorch_model.bin", # Diffusers save path
                   compvis_config_file="stable-diffusion/configs/stable-diffusion/v1-inference.yaml", # CompVis config path
                   diffusers_config_file="stable-diffusion/diffusers_ckpt/ORI/unet/config.json") # Diffusers config path