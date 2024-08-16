import argparse
import sys
import gc
import time
import datasets
import img_text_composition_models
import numpy as np
from torch.autograd import Variable
import test_retrieval
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm
from copy import deepcopy
import socket
import os
from datetime import datetime
import PIL


def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='')
    parser.add_argument('--comment', type=str, default='fashion200k_composeAE')
    parser.add_argument('--dataset', default='fashion200k')
    parser.add_argument('--dataset_path', default='data/fashion200k')
    parser.add_argument('--model', type=str, default='composeAE')
    parser.add_argument('--image_embed_dim', type=int, default=512)
    parser.add_argument('--use_bert', type=bool, default=True)
    parser.add_argument('--use_complete_text_query', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--learning_rate_decay_frequency', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--category_to_train', type=str, default='all')
    parser.add_argument('--num_iters', type=int, default=7)
    parser.add_argument('--loss', type=str, default='batch_based_classification')
    parser.add_argument('--loader_num_workers', type=int, default=8)
    parser.add_argument('--log_dir', type=str, default='../logs/')
    parser.add_argument('--test_only', type=bool, default=False)
    parser.add_argument('--model_checkpoint', type=str, default='')

    args = parser.parse_args()
    return args

def load_dataset(opt):
    """Loads the input datasets."""
    print('Reading dataset ', opt.dataset)
    if opt.dataset == 'fashion200k':
        trainset = datasets.Fashion200k(
            path=opt.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.Fashion200k(
            path=opt.dataset_path,
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
    
    else:
        print('Invalid dataset', opt.dataset)
        sys.exit()

    print('trainset size:', len(trainset))
    print('testset size:', len(testset))
    return trainset, testset

def create_model_and_optimizer(opt, texts):
    """Builds the model and related optimizer."""
    print("Creating model and optimizer for", opt.model)
    text_embed_dim = 512 if not opt.use_bert else 768
    
    if opt.model == 'composeAE':
        model = img_text_composition_models.ComposeAE(texts,
                                                     image_embed_dim=opt.image_embed_dim,
                                                     text_embed_dim=text_embed_dim,
                                                     use_bert=opt.use_bert,
                                                     name = opt.model)

    model = model.cuda()

    # create optimizer
    params = [{
        'params': [p for p in model.img_model.fc.parameters()],
        'lr': opt.learning_rate
    }, {
        'params': [p for p in model.img_model.parameters()],
        'lr': 0.1 * opt.learning_rate
    }, {'params': [p for p in model.parameters()]}]

    for _, p1 in enumerate(params):  # remove duplicated params
        for _, p2 in enumerate(params):
            if p1 is not p2:
                for p11 in p1['params']:
                    for j, p22 in enumerate(p2['params']):
                        if p11 is p22:
                            p2['params'][j] = torch.tensor(0.0, requires_grad=True)

    optimizer = torch.optim.SGD(params,
                                lr=opt.learning_rate,
                                momentum=0.9,
                                weight_decay=opt.weight_decay)

    return model, optimizer

opt = parse_opt()
print('Arguments:')
for k in opt.__dict__.keys():
    print('    ', k, ':', str(opt.__dict__[k]))
trainset, testset = load_dataset(opt)
loaded_model, optimizer = create_model_and_optimizer(opt, [t for t in trainset.get_all_texts()])

#path
home_path = '/app'
model_path = home_path + '/models'
model_name = 'model24.pth'
model_save_path = model_path + '/' + model_name
print('Loading model at : ' + model_save_path) 
loaded_model.load_state_dict(torch.load(f = model_save_path))
loaded_model.eval()
print('Model loaded successfully')

def read_img(img_path):
    with open(img_path, 'rb') as f:
        img = PIL.Image.open(f)
        img = img.convert('RGB')

    transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ])
    return transform(img)

def read_query(query_path):
    with open(query_path, 'r') as file:
        content = file.read()
    
    return content


#path
path = home_path + '/data/encoded.npy'
all_imgs = np.load(path)
all_captions = [img['captions'][0] for img in testset.imgs]
all_links = [img['file_path'] for img in testset.imgs]

#path
img_path = home_path + '/data/fashion200k/women/dresses/casual_and_day_dresses/56037632/56037632_0.jpeg'
query = 'replace black with blue'

def return_topk(image_path, query, topk):
    img = read_img(image_path)
    query = read_query(query)
    all_queries = []
    mods = []
    imgs = []
    for i in range(0, 1):
        mods += [query]
        imgs += [img]

    imgs = torch.stack(imgs).float()
    imgs = torch.autograd.Variable(imgs).cuda()

    dct_with_representations = loaded_model.compose_img_text(imgs.cuda(), mods)
    f = dct_with_representations["repres"].data.cpu().numpy()
    all_queries += [f]
    all_queries = np.concatenate(all_queries)

    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])

    sims = all_queries.dot(all_imgs.T)
    nn_result = [np.argsort(-sims[i, :])[:topk] for i in range(sims.shape[0])]
    nn_result_link = [[all_links[nn] for nn in nns] for nns in nn_result]    
    return nn_result_link

def get_folder_file(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return files

container_image_folder = home_path + '/query_image'
current_image_size = 0#len(get_folder_file(container_image_folder))

container_text_folder = home_path + '/query_text'
current_text_size = 0#len(get_folder_file(container_text_folder))

while(True):

    link = None
    while(link is None):
        print('Waiting New Image')

        files = get_folder_file(container_image_folder)
        if (len(files) == current_image_size):
            continue
        else:
            link = container_image_folder + '/' + files[current_image_size]
            current_image_size += 1
            

    query = None
    while(query is None):
        print('Waiting New text')

        files = get_folder_file(container_text_folder)
        if (len(files) == current_text_size):
            continue
        else:
            query = container_text_folder + '/' + files[current_text_size]
            current_text_size += 1

    print(link)
    print(query)

    topk = return_topk(link, query, 10)

    query_retrieval_path = home_path + '/query_retrieval/retrieve_text_' + str(current_image_size - 1) + '.txt'

    #print(len(topk[0]))
    #print(topk[0])
    #print(query_retrieval_path)

    with open(query_retrieval_path, 'w') as file:
        for line in topk:
            for i, ans in enumerate(line):
                print(ans)
                file.write(ans + '\n')
    
    break
    #os.remove(link)
    #os.remove(query)
'''
docker build -t compose .
docker run --gpus all -it --name retrieval_model -v query_image:/app/query_image -v query_text:/app/query_text -v query_retrieval:/app/query_retrieval compose
docker run --gpus all -it --name retrieval_model compose

'''