import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms, models
import argparse
from torch import optim
import torch
from torch import nn
import json


def get_input_args():
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--arch', type = str, default = 'densenet', help = 'CNN model architecture to use can only use vgg and densenet')
    parser.add_argument('--top_k', type = int, default = 5, help = 'Return top K most likely classes')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint1.pth', help = 'saved checkpoint')
    parser.add_argument('--device', type = str, default = 'gpu', help = 'gpu or cpu')
    parser.add_argument('--category_name', type = str, default = 'cat_to_name.json', help = 'mapping of categories to real names')
    parser.add_argument(dest='image_directory', help=' This is a image directory')
    
    return parser.parse_args()

in_arg = get_input_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    pil_image = Image.open(image)
    width, height = pil_image.size
    
    if (width < height):
        pil_image.thumbnail((256, 50000))
    else:
        pil_image.thumbnail((50000, 256))
        
    transform = transforms.CenterCrop(224)
    transform_img = transform(pil_image)
    

    np_image = np.array(transform_img)/255 # Got this idea from someones else code snapshot posted on knowledge
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    np_image = (np_image - mean) / std # Got this idea from someones else code snapshot posted on knowledge
    
    np_image = np_image.transpose((2, 0, 1)) # confirmed it was (2,0,1) from multiple codesnapshot posted on knowledge
    
    
    return np_image




def predict(image_path, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    in_arg    = get_input_args()
    
    model.eval()
    processed_image = process_image(image_path) # now in nd_array
    img = torch.from_numpy(processed_image)
    img = torch.unsqueeze(img, 0) # Got this idea from someones else code snapshot posted on knowledge
    img = img.type(torch.FloatTensor) # Got this idea from someones else code snapshot posted on knowledge
    
        
    if in_arg.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    with torch.no_grad(): # Got this idea from someones else code snapshot posted on knowledge
        ps = model(img.to(device)) # Got this idea from someones else code snapshot posted on knowledge
    
    
    ps = torch.exp(ps)
    probs, labels = ps.topk(top_k, dim=1)
    
    class_idx_dict = {model.class_to_idx[key]: key for key in model.class_to_idx} # Got this idea from mentors answers to someones question on knowledge
    
    classes = list()
    
    cpu_prob = probs.cpu().numpy().flatten() # Got this idea online
    cpu_label = labels.cpu().numpy().flatten() # Got this idea online

    
    for label in cpu_label:
        classes.append(class_idx_dict[label])

    return cpu_prob, classes

def load_checkpoint(filepath):
    
    #initialize model and optimizer
    if in_arg.arch == "densenet":
        model = models.densenet161(pretrained=True)
        checkpoint = torch.load(filepath)    
    
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        
    
        optimizer = optim.Adam(model.classifier.parameters(), lr = 0.002) # Got this idea online (pytorch website)
        optimizer.load_state_dict(checkpoint['optim_state_dict']) # Got this idea online (pytorch website)
        
        return model
    elif in_arg.arch == "vgg":
        model = models.vgg19(pretrained=True)
        checkpoint = torch.load(filepath)    
    
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        
    
        optimizer = optim.Adam(model.classifier.parameters(), lr = 0.002) # Got this idea online (pytorch website)
        optimizer.load_state_dict(checkpoint['optim_state_dict']) # Got this idea online (pytorch website)
        
        return model
    else:
        print("Error: We can only handle vgg and densenet")


def main():
    
    in_arg    = get_input_args()
    
    with open(in_arg.category_name, 'r') as f:
        cat_to_name = json.load(f)


    img = in_arg.image_directory
    
   
    saved_model = load_checkpoint(in_arg.checkpoint)
    
    if in_arg.device == 'gpu':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    saved_model.to(device)
    
    prob, classes = predict(img, saved_model, in_arg.top_k)

#     # top 5
#     names = []
#     for i in range(in_arg.top_k):
#         names.append(cat_to_name[str(classes[i])])
        
    
    print(f"\nThe most likely class for this image is {cat_to_name[str(classes[0])]} with probability {prob[0]}. \n\n")
    
    print(f"The top {in_arg.top_k} classes with their probabilities are: \n\n")
    
    for k in range(in_arg.top_k):
        print(f"{cat_to_name[str(classes[k])]} with probability {prob[k]}.\n")
          
#     plt.figure()
#     image = Image.open(img)
#     plt.imshow(transform2(transform1(image)))
#     plt.axis('off')
#     plt.title(cat_to_name[str(cpu_class[0])]);


#     plt.figure()
#     plt.barh(names, cpu_prob)
          
if __name__ == '__main__':
    main()
         
