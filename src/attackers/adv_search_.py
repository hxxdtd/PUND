from .base import Attacker
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import pandas as pd
from tqdm import tqdm

class AdvSearch(Attacker):
    def __init__(
                self,
                lr=1e-1,
                weight_decay=0.1,
                rand_init=False,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.rand_init = rand_init
    
    def init_adv(self, task, init_token='man'):
        token_embedding = task.text_encoder.text_model.embeddings.token_embedding

        adv_ids = task.tokenizer([init_token], return_tensors='pt').input_ids.to(task.device)[:,1:2]
        adv_ids = adv_ids.repeat((1, self.k)) # k is always set to 1

        print(f"Initial token is {init_token}, whose id is {adv_ids.detach().cpu().item()}")
        adv_embeds = token_embedding(adv_ids).detach()
        adv_embeds.requires_grad = True
        self.adv_embedding = adv_embeds

    def init_opt(self):
        self.optimizer = torch.optim.AdamW([self.adv_embedding],lr = self.lr,weight_decay=self.weight_decay)

    def split_embd(self,input_embed,orig_prompt_len):
        sot_embd, mid_embd, _, eot_embd = torch.split(input_embed, [1, orig_prompt_len, self.k, 76-orig_prompt_len-self.k], dim=1)
        self.sot_embd = sot_embd
        self.mid_embd = mid_embd
        self.eot_embd = eot_embd
        return sot_embd, mid_embd, eot_embd
    
    def split_id(self,input_ids,orig_prompt_len):
        sot_id, mid_id,_, eot_id = torch.split(input_ids, [1, orig_prompt_len, self.k, 76-orig_prompt_len-self.k], dim=1)
        return sot_id, mid_id, eot_id
    
    def construct_embd(self,adv_embedding):
        if self.insertion_location == 'suffix_k':   # Append k words after the original prompt (we use this setting)
            embedding = torch.cat([self.sot_embd,self.mid_embd,adv_embedding,self.eot_embd],dim=1)
        return embedding
    
    # def construct_id(self,adv_id,sot_id,eot_id,mid_id):
    #     if self.insertion_location == 'suffix_k':
    #         input_ids = torch.cat([sot_id,mid_id,adv_id,eot_id],dim=1)
    #     return input_ids

    def run(self, task, logger): # search for the adversarial embedding
        
        save_path = logger.img_root.replace(logger.img_root.split('/')[-1], '')
        os.makedirs(save_path + 'emb/') # the path of the obtained embeddings (the candidate embedding set $V$)
        concept = task.concept # the concept to restore
        self.init_adv(task, init_token=concept.split()[-1])
        self.init_opt()

        parameters = []
        for name, param in task.target_unet_sd.named_parameters(): # only update the cross attention parameters for efficiency
            if 'attn2' in name:
                # print(name)
                parameters.append(param)

        optimizer_sd = torch.optim.Adam(parameters, lr=1e-5)
        task.tokenizer.pad_token = task.tokenizer.eos_token
        # token_embedding = task.text_encoder.text_model.embeddings.token_embedding

        csv_path = os.path.join(task.dataset_path, 'choose.csv') # choose good training images 
        if os.path.exists(csv_path):
            print(f"> > > read csv file: {csv_path} < < <")
            df = pd.read_csv(csv_path)
            case_ls = list(df['case_number'])
            case_ls = case_ls[:self.total_data]
            # print(case_ls)
        else:
            print('> > > NO csv file < < <') # use the images without choosing
            case_ls = list(range(self.total_data))

        progress_bar = tqdm(total=self.epoch)
        for e in range(self.epoch): # 
            
            attack_idx = random.sample(case_ls, 1)[0]
            image, prompt, seed, guidance = task.dataset[attack_idx] # randomly pick the training image

            if concept != "nudity":
                prompt = prompt.replace(f' {concept}', '') # remove the target concept token(s) in the original prompt (e.g., "a photo of angelina jolie" ---> "a photo of")

            # print(f"Epoch: {e:4d} | Prompt: {prompt} s* | Seed: {seed:3d} | Guidance: {guidance:.1f}")

            if seed is None:
                seed = self.eval_seed

            x0 = task.img2latent(image)
            input_ids = task.str2id(prompt)
            orig_prompt_len = (input_ids == 49407).nonzero(as_tuple=True)[1][0]-1
            input_embeddings = task.id2embedding(input_ids)
            encoder_hidden_states_neutral = task.text_encoder(input_ids=task.str2id("a photo of"))[0] # y' for unet parameters update
            self.split_embd(input_embeddings,orig_prompt_len)

            task.target_unet_sd.eval()
            task.target_unet_sd.requires_grad_(False)

            for jj in range(4, -1, -1):
                t = random.randint(jj*200, (jj+1)*200-1) # timestep t
                for _ in range(2):

                    self.optimizer.zero_grad()

                    if self.attack_type == "embed":
                        adv_input_embeddings = self.construct_embd(self.adv_embedding)

                    input_arguments = {"x0":x0,"t":t,"input_ids":input_ids,"input_embeddings":adv_input_embeddings}
                    noise_pred_loss = task.get_loss(**input_arguments) # `input_ids` actually does not affect `last_hidden_state` (text condition)
                    
                    loss = noise_pred_loss / noise_pred_loss.detach()

                    if self.attack_type == "embed":
                        self.adv_embedding.grad = torch.autograd.grad(loss, [self.adv_embedding])[0]
                    
                    self.optimizer.step()
                    progress_bar.set_postfix(epoch=e, timestep=f'{t:4d}', loss=f'{noise_pred_loss.detach().cpu().item():.3f}')
                    # print(f" > > > t {t}, noise_pred_loss: {noise_pred_loss:.3f}")
            
            torch.save(self.adv_embedding.detach().cpu(), save_path + 'emb/' + f'{e:04d}.pth') # save the obtained embedding
            
            if e % 2 == 0: # input the obtained embeddings into the erased model to generate images every 2 epochs (for validation)
                with torch.no_grad():
                    text_embeddings = self.construct_embd(self.adv_embedding)
                    valid_seed = self.valid_seed if self.valid_seed is not None else seed
                    results = task.eval(task.erase_unet_sd, input_ids,text_embeddings=text_embeddings,seed=valid_seed,guidance_scale=guidance)
                    results['epoch'] = e
                    logger.save_img(f'epoch_{e:04d}-seed_{valid_seed:03d}', results.pop('image'))
                    logger.log(results)

            if self.no_adv == False: # with adversarial search
                if e % 200 == 0 and e > 0: # update the model parameters every 200 epochs

                    print(f"{' - ' * 5} Update Unet {' - ' * 5}")
                    for _ in range(1):
                        for kk in range(4, -1, -1):
                            t = random.randint(kk*200, (kk+1)*200-1)

                            task.target_unet_sd.train()
                            task.target_unet_sd.requires_grad_(True)
                            input_arguments_adv = {"x0":x0,"t":t,"input_ids":input_ids,"input_embeddings_adv":adv_input_embeddings.detach(),"encoder_hidden_states_neutral":encoder_hidden_states_neutral}

                            optimizer_sd.zero_grad()
                            noise_pred_loss = task.get_loss_adv(**input_arguments_adv)
                            loss = noise_pred_loss / noise_pred_loss.detach()
                            loss.backward()
                            optimizer_sd.step()
                    results = task.eval(task.target_unet_sd, input_ids,text_embeddings=adv_input_embeddings.detach(),seed=valid_seed,guidance_scale=guidance)
                    logger.save_img(f'epoch_{e:04d}-seed_{seed:03d}-ORI_After_Update', results.pop('image'))
            progress_bar.update()
                    
def get(**kwargs):
    return AdvSearch(**kwargs)