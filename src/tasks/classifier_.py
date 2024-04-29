import os
from copy import deepcopy
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from uuid import uuid4
from .utils.metrics.id_score import calculate_id_score
from .utils.metrics.nudity_eval import if_nude, detectNudeClasses
from .utils.metrics.style_eval import style_eval,init_classifier
from .utils.metrics.object_score import calculate_object_score
from .utils.text_encoder import CustomTextEncoder
from .utils.datasets import get as get_dataset

class ClassifierTask:
    def __init__(
                self,
                concept_type,
                concept,
                model_name_or_path,
                target_ckpt,
                erase_ckpt,
                cache_path,
                dataset_path,
                criterion,
                sampling_step_num,
                n_samples = 50,
                ):
        # self.object_list = ['english_springer', 'jeep']
        # self.object_labels = [217, 609]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.concept_type = concept_type
        self.concept = concept
        self.cache_path = cache_path
        self.sampling_step_num = sampling_step_num
        self.dataset_path = dataset_path
        self.dataset = get_dataset(dataset_path)
        self.criterion = torch.nn.L1Loss() if criterion == 'l1' else torch.nn.MSELoss()
        self.vae = AutoencoderKL.from_pretrained(model_name_or_path, subfolder='vae').to(self.device)  

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(model_name_or_path, subfolder='text_encoder').to(self.device)
        self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.device)
        self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)

        self.target_unet_sd = UNet2DConditionModel.from_pretrained(target_ckpt.replace(target_ckpt.split('/')[-1], '')).to(self.device)
        self.erase_unet_sd = UNet2DConditionModel.from_pretrained(erase_ckpt.replace(erase_ckpt.split('/')[-1], '')).to(self.device)
        
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.T = 1000
        self.n_samples = n_samples
        start = self.T // self.n_samples // 2
        self.sampled_t = list(range(start, self.T, self.T // self.n_samples))[:self.n_samples]
        
        for m in [self.vae, self.text_encoder, self.custom_text_encoder, self.target_unet_sd, self.erase_unet_sd]:
            m.eval()
            m.requires_grad_(False)

    # get the loss $\mathcal{L}_{v}$ to update the embedding
    def get_loss(self,x0,t,input_ids,input_embeddings):

        x0 = x0.to(self.device)
        x0 = x0.repeat(input_embeddings.shape[0], 1, 1, 1)
        noise = torch.randn((1, 4, 64, 64), device=self.device)
        noise = noise.repeat(input_embeddings.shape[0], 1, 1, 1)
        noised_latent = x0 * (self.scheduler.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1).to(self.device) + \
                        noise * ((1 - self.scheduler.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1).to(self.device)
        encoder_hidden_states = self.custom_text_encoder(input_ids = input_ids,inputs_embeds=input_embeddings)[0]
        noise_pred = self.target_unet_sd(noised_latent,t,encoder_hidden_states=encoder_hidden_states).sample
        error = self.criterion(noise,noise_pred)
        return error
       
    # get the loss $\mathcal{L}_{\theta}$ to update the model parameters
    def get_loss_adv(self,x0,t,input_ids,input_embeddings_adv,encoder_hidden_states_neutral,**kwargs):

        x0 = x0.to(self.device)
        x0 = x0.repeat(input_embeddings_adv.shape[0], 1, 1, 1)
        noise = torch.randn((1, 4, 64, 64), device=self.device)
        noise = noise.repeat(input_embeddings_adv.shape[0], 1, 1, 1)
        noised_latent = x0 * (self.scheduler.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1).to(self.device) + \
                        noise * ((1 - self.scheduler.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1).to(self.device)
        
        encoder_hidden_states_adv = self.custom_text_encoder(input_ids = input_ids,inputs_embeds=input_embeddings_adv)[0]

        noise_pred_adv = self.target_unet_sd(noised_latent,t,encoder_hidden_states=encoder_hidden_states_adv).sample
        noise_pred_neutral = self.target_unet_sd(noised_latent,t,encoder_hidden_states=encoder_hidden_states_neutral).sample

        error = self.criterion(noise_pred_adv, noise_pred_neutral.detach())
        return error

    def str2id(self,prompt):
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt",truncation=True
        )
        return text_input.input_ids.to(self.device)
    
    def img2latent(self,image):
        with torch.no_grad():
            img_input  = image.unsqueeze(0).to(self.device)
            x0 = self.vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        return x0
    
    def id2embedding(self,input_ids):
        input_one_hot = F.one_hot(input_ids.view(-1), num_classes = len(self.tokenizer.get_vocab())).float()
        input_one_hot = torch.unsqueeze(input_one_hot,0).to(self.device)
        input_embeds = input_one_hot @ self.all_embeddings
        return input_embeds
    
    def sampling(self,unet,input_ids,text_embeddings,batch_size=1,seed=0,height=512,width=512,num_inference_steps=50,guidance_scale=7.5):
        
        text_embeddings = self.custom_text_encoder(input_ids=input_ids, inputs_embeds=text_embeddings)[0]
        
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        
        uncond_embeddings = self.id2embedding(uncond_input.input_ids.to(self.device))
        uncond_embeddings = self.custom_text_encoder(input_ids = uncond_input.input_ids.to(self.device), inputs_embeds=uncond_embeddings)[0]          
        
        generator = torch.manual_seed(seed)  
        latents = torch.randn(
            (batch_size, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * self.scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        self.scheduler.set_timesteps(num_inference_steps)

        # for t in tqdm(self.scheduler.timesteps):
        for t in self.scheduler.timesteps:
            
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=uncond_embeddings).sample
                noise_pred_text = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images[0]
    
    def eval(self,unet,input_ids,text_embeddings,seed=0,batch_size=1,height=512,width=512,guidance_scale=3.0):
        results = {}
        image = self.sampling(unet,input_ids,text_embeddings,batch_size=batch_size,seed=seed,height=height,width=width,num_inference_steps=self.sampling_step_num,guidance_scale=guidance_scale)
        results['image'] = Image.fromarray(image)
        img_tmp_path = os.path.join(self.cache_path, f'{uuid4()}.png')
        # store in the local cache for object, nudity classification
        results['image'].save(img_tmp_path)
        if self.concept_type == 'id':
            id_score = calculate_id_score(results['image'], self.concept)
            results['score'] = round(id_score, 5)
        elif self.concept_type == 'object':
            object_score = calculate_object_score(img_tmp_path, self.concept)
            results['score'] = round(object_score, 5)
        os.remove(img_tmp_path)
        return results

def get(**kwargs):
    return ClassifierTask(**kwargs)