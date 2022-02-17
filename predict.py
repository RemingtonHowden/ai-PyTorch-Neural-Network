import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from torchvision import models, datasets, transforms
import PIL
from PIL import Image
from collections import OrderedDict
import seaborn as sns
import argparse 
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    class_to_idx = cat_to_name
# function that loads a checkpoint and rebuilds the model

#image_path = 'flowers/test/102/image_08042.jpg'
def load(checkpoint, gpu):   
    check_point = torch.load(checkpoint)
    model = check_point['model']
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(gpu)
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    model.eval()
    
    return model



def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    crop_size = 224 
    
    # PIL image
    test_image = PIL.Image.open(image_path)
    
    # Boundaries
    orig_width, orig_height = test_image.size # Acquiring image size data
    print('{} {}'.format(orig_width, orig_height)) # displaying size info 
    
    if orig_height >= orig_width:
        test_image.thumbnail(size = [256, 256**2])
        print('height > width')
        
    else:
        test_image.thumbnail(size = [256**2, 256])
        print('width > height')
        
    left = orig_width/4 - crop_size/2
    top = orig_height/4 - crop_size/2
    right = (left + crop_size)
    bottom = (top + crop_size)
    print((left, top, right, bottom))
    
    # cropping image
    test_image = test_image.crop((left, top, right, bottom))
    
    np_image = np.array(test_image)/255
    
    normalize_means = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    np_image = (np_image - normalize_means)/normalize_std
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

#

def imshow(image_path, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image_path = image
    image = np.array(image).transpose((1, 2, 0))
    
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#image_path = 'flowers/test/102/image_08042.jpg'

# predicts class of image 
def predict(image_path, model, top_k):
    # load and process image
    img = process_image(image_path)
    # update to one sized dimension 
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    # converting to Tensor for correct pass to forward pass model 
    img = img.float()
    img = img.cuda()
    #model = load(checkpoint)
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    
    with torch.no_grad():
        ps = torch.exp(model.forward(img))
        top_prob, top_class = ps.topk(top_k, dim=1)
    classes = [idx_to_class[x] for x in top_class.cpu().numpy()[0]]
    return top_prob[0], top_class, [cat_to_name[c] for c in classes]

def main():
    
    # Command line interface arguments for user interaction and model modifications
    parser = argparse.ArgumentParser(description = 'Testing Neural Network')
    parser.add_argument('input', type=str, help = 'Specify path to input image')
    parser.add_argument('checkpoint', type=str, help = 'Please specify where to load model from...')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', type=str, default='cuda', help = 'Utilize graphics processing unit for inference if available')
    parser.add_argument('--top_k', type=int, default=5, help = 'Specifiy amount of top predicted classes to display')
    arg = parser.parse_args()
    image_path = arg.input
    checkpoint = arg.checkpoint
    category_names = arg.category_names
    gpu = arg.gpu
    top_k = arg.top_k
    
    model = load(checkpoint, gpu)
    img = process_image(image_path)
    #imshow(image_path, ax=None, title = None)
    probs, labs, flowers =  predict(image_path, model, top_k)
    print(probs)
    print(flowers)
# calling main function to run program
if __name__ == '__main__':
    main()