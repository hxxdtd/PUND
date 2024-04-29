from importlib import import_module
import os
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import OneOf, File, Folder, BoolAsInt
import argparse
import random
import torch
import numpy as np
from datetime import datetime

import sys
sys.path.append('src')

Section('overall', 'Overall configs').params(
    task = Param(OneOf(['classifier']), required=True, desc='Task type to attack'),
    attacker = Param(OneOf(['adv_search']), required=True, desc='Attack algorithm'),
    logger = Param(OneOf(['json']), default='none', desc='Logger to use'),
    resume = Param(Folder(), default=None, desc='Path to resume'),
    seed = Param(int, default=0, desc='Random seed'),
)

Section('task', 'General task configs').params(
    concept_type = Param(OneOf(['id', 'object', 'style', 'nsfw']), required=True, desc='Concept type'),
    concept = Param(OneOf(['van gogh', 'nudity', "barack obama", 'car', 'emma watson', 'hillary clinton', 'tom cruise', 'monet', 'angelina jolie', 'brad pitt', 'pablo picasso', 'marc chagall', 'english springer', 'dog', 'jeep']), required=True, desc='Concept to restore'), # add the concepts you want to restore
    model_name_or_path = Param(str, required=True, desc='Model directory (load vae, tokenizer, text_encoder)'),
    target_ckpt = Param(File(), required=True, desc='Target model checkpoint (the ORIginal SD in our setting)'),
    erase_ckpt = Param(File(), required=True, desc='Erased model checkpoint (to validate the obtained embedding)'),
    cache_path = Param(Folder(True), default='.cache', desc='Cache directory'),
    dataset_path = Param(Folder(), required=True, desc='Path to dataset'),
    criterion = Param(OneOf(['l1', 'l2']), default='l2', desc='Loss criterion'),
    sampling_step_num = Param(int, default=25, desc='Sampling step number'),
)

Section('attacker', 'General attacker configs').params(
    insertion_location = Param(OneOf(['suffix_k']), default='suffix_k', desc='Embedding insertion location'),
    k = Param(int, default=1, desc='k in insertion_location'),
    epoch = Param(int, default=3000, desc='Number of epochs'),
    eval_seed = Param(int, default=0, desc='Evaluation seed'),
    total_data = Param(int, required=False, desc='Total amount of data for the embedding search'),
    attack_type = Param(str, default='embed', desc='Find adversarial embedding or token (currently only embedding is supported)'),
    no_adv = Param(bool, is_flag=True, desc='Disable adversarial search'),
    valid_seed = Param(int, default=None, desc='Validation seed')
)

Section('attacker.adv_search', 'Adversarial Search').enable_if(
    lambda cfg: cfg['overall.attacker'] == 'adv_search'
).params(
    lr = Param(float, default=0.1, desc='Learning rate'),
    weight_decay = Param(float, default=0.1, desc='Weight decay'),
    rand_init = Param(BoolAsInt(), default=False, desc='Random initialization'),
)

Section('logger', 'General logger configs').params(
    name = Param(str, default=datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'), desc='Name of this run'),
)

Section('logger.json', 'JSON logger').enable_if(
    lambda cfg: cfg['overall.logger'] == 'json'
).params(
    root = Param(Folder(True), default='files/logs', desc='Path to log folder'),
)


class Main:

    def __init__(self) -> None:
        self.make_config()
        self.setup_seed()
        self.init_task()
        self.init_attacker()
        self.init_logger()
        self.run()

    def make_config(self, quiet=False):
        self.config = get_current_config()
        parser = argparse.ArgumentParser("Probing Unlearned Diffusion Models")
        self.config.augment_argparse(parser)
        self.config.collect_argparse_args(parser)

        if self.config['overall.resume'] is not None:
            self.config.collect_config_file(os.path.join(self.config['overall.resume'], 'config.json'))

        self.config.validate()
        if not quiet:
            self.config.summary()

    @param('overall.seed')
    def setup_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    @param('overall.task')
    def init_task(self, task):
        kwargs = self.config.get_section(f'task')
        kwargs.update(self.config.get_section(f'task.{task}'))
        self.task = import_module(f'tasks.{task}_').get(**kwargs)

    @param('overall.attacker')
    def init_attacker(self, attacker):
        kwargs = self.config.get_section(f'attacker')
        kwargs.update(self.config.get_section(f'attacker.{attacker}'))
        self.attacker = import_module(f'attackers.{attacker}_').get(**kwargs)

    @param('overall.logger')
    def init_logger(self, logger):
        kwargs = self.config.get_section(f'logger')
        kwargs.update(self.config.get_section(f'logger.{logger}'))
        kwargs['config'] = self.config.get_all_config()
        self.logger = import_module(f'loggers.{logger}_').get(**kwargs)
    
    def run(self):
        self.attacker.run(self.task, self.logger)


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="7"
    Main()