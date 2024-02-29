import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
from torchvision import utils
from torchvision.transforms import Resize

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from models.stylegan2.model import bg_extractor_repro, Generator

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


def run():
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_masks = os.path.join(test_opts.exp_dir, 'inference_masks', 
                                      'downsampling_{}'.format(test_opts.resize_factors))
        out_path_new_bg = os.path.join(test_opts.exp_dir, 'new_bg',
                                       'downsampling_{}'.format(test_opts.resize_factors))
        out_path_gen_bg = os.path.join(test_opts.exp_dir, 'generated_bg',
                                       'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
        out_path_masks = os.path.join(test_opts.exp_dir, 'inference_masks')
        out_path_new_bg = os.path.join(test_opts.exp_dir, 'new_bg')
        out_path_gen_bg = os.path.join(test_opts.exp_dir, 'generated_bg')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)
    os.makedirs(out_path_masks, exist_ok=True)
    os.makedirs(out_path_new_bg, exist_ok=True)
    os.makedirs(out_path_gen_bg, exist_ok=True)
    

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    # alpha model
    device = 'cuda'
    bg_extractor_ = bg_extractor_repro(image_size = 1024, min_res = 32).cuda()
    ckpt_ = torch.load(test_opts.ckpt_bg_extractor, map_location=lambda storage, loc: storage)
    bg_extractor_.load_state_dict(ckpt_["bg_extractor_ema"])
    bg_extractor_.eval()

    # bg generator
    bg_generator = Generator(1024, 512, 8, channel_multiplier=2).to(device)
    bg_generator.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
    bg_generator.eval()

    with torch.no_grad():
        mean_latent = bg_generator.mean_latent(4096)

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch, _  = net(input_cuda, randomize_noise=False, resize=opts.resize_outputs)
            
            sample_z2 = torch.randn(test_opts.test_batch_size, 512, device=device)
            sample_bg, __, ___ = bg_generator([sample_z2], truncation=0.5, truncation_latent=mean_latent, back = True)
            
            alpha_mask = bg_extractor_(_)
            hard_mask = (alpha_mask > test_opts.th).float()
            
            image_new_bg = result_batch * alpha_mask + (1 - alpha_mask) * sample_bg
            image_new_bg_hard = result_batch * hard_mask + (1 - hard_mask) * sample_bg

            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            new_bg = tensor2im(image_new_bg[i])
            hard_bg = tensor2im(image_new_bg_hard[i])
            gen_bg = tensor2im(sample_bg[i])
            im_path = dataset.paths[global_i]

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                mask_size = (test_opts.mask_size, test_opts.mask_size)
                if opts.resize_factors is not None:
                    # for super resolution, save the original, down-sampled, and output
                    source = Image.open(im_path)
                    res = np.concatenate([np.array(source.resize(resize_amount)),
                                          np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                          np.array(result.resize(resize_amount))], axis=1)
                else:
                    # otherwise, save the original and output
                    res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                          np.array(result.resize(resize_amount)),
                                          np.array(new_bg.resize(resize_amount)),
                                          np.array(hard_bg.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)
        
            im_save_path = os.path.join(out_path_new_bg, os.path.basename(im_path))
            Image.fromarray(np.array(new_bg)).save(im_save_path)

            im_save_path = os.path.join(out_path_masks, os.path.basename(im_path))
            utils.save_image(
                        Resize(mask_size)(hard_mask[i]),
                        im_save_path,
                        nrow=int(test_opts.test_batch_size ** 0.5),
                        normalize=False, 
                    )

            im_save_path = os.path.join(out_path_gen_bg, os.path.basename(im_path))
            Image.fromarray(np.array(gen_bg)).save(im_save_path)

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == '__main__':
    run()
