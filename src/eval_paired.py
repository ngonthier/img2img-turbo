import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
import json
import glob
import ntpath
import pathlib
import shutil
import torch_fidelity
from tqdm import tqdm
import pandas as pd

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def create_folder_per_modality(path_results='\results\flair_pix2pix\test_latest\images'):
    fakes = glob.glob(os.path.join(path_results, '*_fake_B.png'))
    reals = glob.glob(os.path.join(path_results, '*_real_B.png'))
    pathlib.Path(os.path.join(path_results,'reals')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(path_results,'fakes')).mkdir(parents=True, exist_ok=True)
    for fake in fakes:
        src = fake
        dst = os.path.join(path_results,'fakes',os.path.basename(fake))
        shutil.copyfile(src, dst)
    for real in reals:
        src = real
        head, tail = os.path.split(os.path.split(real)[0])
        dst = os.path.join(path_results,'reals',os.path.basename(real))
        shutil.copyfile(src, dst)


def eval_pix2pix(method, path):

    create_folder_per_modality(path_results=path)
    real_path = os.path.join(path,'reals')
    fake_path = os.path.join(path,'fakes')

    eval_dict = { 'method': method}

    # 'kid_subset_size': 50, 'kid_subsets': 10 pour le vrai cas 
    eval_args = {'isc': True, 'fid': True, 'kid': True, 'kid_subset_size': 50, 'kid_subsets': 10, 'verbose': True, 'cuda': True}
    metric_dict_AB = torch_fidelity.calculate_metrics(input1=real_path, input2=fake_path, **eval_args)
    print('metric_dict_AB',metric_dict_AB)
    eval_dict['ISC'] = metric_dict_AB['inception_score_mean']
    eval_dict['FID'] = metric_dict_AB['frechet_inception_distance']
    eval_dict['KID'] = metric_dict_AB['kernel_inception_distance_mean']*100.
    print('[*] evaluation finished!')
    
    #print('rmse: {0}, acc5: {1}, acc10: {2}, pFPR: {3}, iFPR: {4}'.format(RMSE, acc5, acc10, pFPR100, iFPR))
    return eval_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the image folder', default='/lustre/fsn1/projects/rech/abj/ujq24es/dataset/PixtoPixTurbo_FLAIR')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='/lustre/fsn1/projects/rech/abj/ujq24es/pix2pixTurbo_output', help='the directory to save the output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--method', type=str, default='pix2pix_turbo', help='Random seed to be used')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()

    path_test_A = os.path.join(args.path,'test_A')
    path_test_B = os.path.join(args.path,'test_B')
    output_path = args.path
    test_prompt_file = "test_prompts.json"
    path_csv_files='/lustre/fsn1/projects/rech/abj/ujq24es/dataset/FLAIR-INC'
    test_set_path = os.path.join(path_csv_files,'TEST_FLAIR-INC 1.csv')

    test_set = pd.read_csv(test_set_path,names=['img','msk'])
    test_set['img'] = test_set['img'].apply(lambda x: x.split('/')[-1])
    test_set['img'] = test_set['img'].apply(lambda x: x.split('.')[0])
    list_test_img = test_set['img'].to_list()

    with open(os.path.join(output_path,test_prompt_file), 'r') as f:
        test_prompt = json.load(f)
    mask_tif = glob.glob(os.path.join(path_test_A,"*.tif"))
    aerial_tif = glob.glob(os.path.join(path_test_B,"*.tif"))

    pathlib.Path(os.path.join(args.output_dir,'reals')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(args.output_dir,'fakes')).mkdir(parents=True, exist_ok=True)
    max_number_img = 10
    number_image = 0

    print('list_test_img',list_test_img)

    for img, msk in tqdm(zip(aerial_tif, mask_tif)):
        filename_img = os.path.basename(img).split('.')[0]
        img_with_ext = path_leaf(img)
        img_short = img_with_ext.split('.')[0]
        img_with_png = img_short + '.png'
        print('img_short',img_short)

        if number_image > max_number_img:
            break

        #number = filename_img.split('_')[1]
        # Need to pick up one random 
        if img_short in list_test_img: 
            print(img_short,'in test')
            number_image += 1
            img_with_png 
            prompt = test_prompt[img_with_png]
            # make sure that the input image is a multiple of 8
            input_image = Image.open(msk).convert('RGB')
            #new_width = input_image.width - input_image.width % 8
            #new_height = input_image.height - input_image.height % 8
            #input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
            bname = os.path.basename(args.input_image)

            # translate the image
            with torch.no_grad():
                c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
                output_image = model(c_t, args.prompt)

                output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

            # save the output image 
            fake_image = os.path.join(args.output_dir,'fakes', bname)
            output_pil.save(fake_image)
            print("save faek to :",fake_image)

            dst =  os.path.join(os.path.join(args.output_dir,'reals'),os.path.basename(img_with_png))
            print('saving',filename_img,'to',dst)
            shutil.copyfile(filename_img, dst) # src, dst

    print('==',number_image,'test images')
    eval_pix2pix(method=args.method, path=args.output_dir)

            
