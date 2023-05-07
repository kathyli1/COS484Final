'''
This file contains the code for our first method of altering CLIPScore: Comparing Image Crop CLIPScores (4.2).
The methods we made changes in are __getitem__ in the CLIPImageDataset class, extract_all_images,
and get_clip_score. Note: all changes commented with "NEW".
'''
import argparse
import clip
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import tqdm
import numpy as np
import random
import sklearn.preprocessing
import collections
import os
import pathlib
import json
import generation_eval_utils
import pprint
import warnings
from packaging import version
import torchvision.transforms as T


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'candidates_json',
        type=str,
        help='Candidates json mapping from image_id --> candidate.')

    parser.add_argument(
        'image_dir',
        type=str,
        help='Directory of images, with the filenames as image ids.')

    parser.add_argument(
        '--references_json',
        default=None,
        help='Optional references json mapping from image_id --> [list of references]')

    parser.add_argument(
        '--compute_other_ref_metrics',
        default=1,
        type=int,
        help='If references is specified, should we compute standard reference-based metrics?')

    parser.add_argument(
        '--save_per_instance',
        default=None,
        help='if set, we will save per instance clipscores to this file')

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith('.json'):
        print('if you\'re saving per-instance, please make sure the filepath ends in json.')
        quit()
    return args


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def func(self, image):
        return image.convert("RGB")

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            # lambda image: image.convert("RGB"),
            # lambda image: image.convert("RGB"),
            self.func,

            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        width, height = image.size
        images = []

        # original 
        # image = self.preprocess(imgage)
        # return {'image':image}

        # NEW: fixed size random crop of size crop_size*width and crop_size*height
        # crop_size = .9
        # for i in range(4):
        #     transform = T.RandomResizedCrop((int(width*crop_size),int(height*crop_size)))
        #     img = transform(image)
        #     images.append(img)

        # NEW: Random-sized random crop
        #'''
        for i in range(5):
            rand_width = int((.75 + random.random() / 4) * width)
            rand_height = int((.75 + random.random() / 4) * height)
            transform = T.RandomResizedCrop((rand_width, rand_height))
            img = transform(image)
            images.append(img)
        #'''

        # NEW: k center crops decreasing by crop_sz in size each time
        # k = 5
        # crop_sz = .1
        # for i in range(k):
        #     crop = crop_sz * i
        #     img = image.crop((int(width*crop), int(height*crop), int(width*(1-crop)), int(height*(1-crop))))
        #     images.append(img)
    
        for i in range(len(images)):
            images[i] = self.preprocess(images[i])

        return {'image': images}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=1):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=1):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)

    all_image_features = []
    with torch.no_grad():
        for images in tqdm.tqdm(data):
            im_feature_arr = []

            # NEW: additional loop to deal with multiple crops per image
            for b in images['image']:
                b = b.to(device)
                if device == 'cuda':
                    b = b.to(torch.float16)
                im_feature_arr.append(model.encode_image(b).cpu().numpy())

            im_feature_arr = np.stack(im_feature_arr, axis=1)
            all_image_features.append(im_feature_arr)
        all_image_features = np.vstack(all_image_features)

    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)
    candidates = np.expand_dims(candidates, 1)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=-1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=-1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis=-1, keepdims=True))
        candidates = candidates / \
            np.sqrt(np.sum(candidates**2, axis=-1, keepdims=True))

    prod = images * candidates

    # NEW: Compute CLIPScores for each crop of each image
    pers = []
    for i in range(prod.shape[1]):
        per = w*np.clip(np.sum(prod[:,i,:], axis=1), 0, None)
        pers.append(per)

    # NEW: Chooses MAX CLIPScore out of the crop CLIPScores
    # for crop in range(len(pers)):
    #     for image in range(len(per)):
    #         if pers[crop][image] > per[image]:
    #             per[image] = pers[crop][image]

    # NEW: Chooses MIN CLIPScore out of the crop CLIPScores
    # for crop in range(len(pers)):
    #     for image in range(len(per)):
    #         if pers[crop][image] < per[image]:
    #             per[image] = pers[crop][image]

    # NEW: Chooses MEDIAN CLIPScore out of the crop CLIPScores
    for i in range(len(per)):
        crops = [crop[i] for crop in pers]
        crops.sort()
        per[i] = crops[len(crops)//2]

    # alternate way to test:
    # prod = prod.max(1)
    # sum = np.sum(prod, axis=-1)
    # sum = sum.max(-1)
    # per = w*np.clip(sum, 0, None)

    return np.mean(per), per, candidates


def get_refonlyclipscore(model, references, candidates, device):
    '''
    The text only side for refclipscore
    '''
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = extract_all_captions(flattened_refs, model, device)

    if version.parse(np.__version__) < version.parse('1.21'):
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
        flattened_refs = sklearn.preprocessing.normalize(
            flattened_refs, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')

        candidates = candidates / \
            np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
        flattened_refs = flattened_refs / \
            np.sqrt(np.sum(flattened_refs**2, axis=1, keepdims=True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, cand in tqdm.tqdm(enumerate(candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        per.append(np.max(all_sims))

    return np.mean(per), per


def main():
    args = parse_args()

    image_paths = [os.path.join(args.image_dir, path) for path in os.listdir(args.image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    image_ids = [pathlib.Path(path).stem for path in image_paths]

    with open(args.candidates_json) as f:
        candidates = json.load(f)
    candidates = [candidates[cid] for cid in image_ids]

    if args.references_json:
        with open(args.references_json) as f:
            references = json.load(f)
            references = [references[cid] for cid in image_ids]
            if isinstance(references[0], str):
                references = [[r] for r in references]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8) #changed num_workers from 8 to 1 for weird data driver error

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device)

    if args.references_json:
        # get text-text clipscore
        _, per_instance_text_text = get_refonlyclipscore(
            model, references, candidate_feats, device)
        # F-score
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / \
            (per_instance_image_text + per_instance_text_text)
        scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
                  for image_id, clipscore, refclipscore in
                  zip(image_ids, per_instance_image_text, refclipscores)}

    else:
        scores = {image_id: {'CLIPScore': float(clipscore)}
                  for image_id, clipscore in
                  zip(image_ids, per_instance_image_text)}
        
        # Prints individual clip scores for each image
        #for s in scores.values():
        #    print(s['CLIPScore'])
        
        print('CLIPScore: {:.4f}'.format(
            np.mean([s['CLIPScore'] for s in scores.values()])))

    if args.references_json:
        if args.compute_other_ref_metrics:
            other_metrics = generation_eval_utils.get_all_metrics(
                references, candidates)
            for k, v in other_metrics.items():
                if k == 'bleu':
                    for bidx, sc in enumerate(v):
                        print('BLEU-{}: {:.4f}'.format(bidx+1, sc))
                else:
                    print('{}: {:.4f}'.format(k.upper(), v))
        print('CLIPScore: {:.4f}'.format(
            np.mean([s['CLIPScore'] for s in scores.values()])))
        print('RefCLIPScore: {:.4f}'.format(
            np.mean([s['RefCLIPScore'] for s in scores.values()])))

    if args.save_per_instance:
        with open(args.save_per_instance, 'w') as f:
            f.write(json.dumps(scores))


if __name__ == '__main__':
    main()
