from DeConv import DeconvNet
import math
from PIL import Image
import torch 
from torch import nn
from matplotlib import pylot as plt 
from torchvision import transforms




class CNN_Vis2D():
    def __init__(self,cnn:nn.Module):
        self.model = cnn
        conv_list = []
        weights_list = []
        layers = list(cnn.children())
        for index,layer in enumerate(layers):
            if type(layer) == torch.nn.modules.conv.Conv2d:
                conv_list.append((layer,index))
                weights_list.append(layer.weight)
        self.conv_list = conv_list
        self.weights_list = weights_list
        self.im_transforms = transforms.Compose(
            [
                transforms.ToTensor()

            ]
        )
        
    def show_layers(self):
       for item,index in enumerate(self.conv_list):
           print(f"{index}\t{item}")
     
    def channel_alert(self,mean_channels:bool,channel_index):
        if mean_channels == True:
            print("Taking across multiple channels in single layer")
        else:
            print(f"plotting filters for channel {channel_index}") 
    
    

    def plot_filters(self,layer_index:int,cmap='grey',mean_channels=False,channel_index=0,save_kernels=False,save_dir=''):
        #handle multiple channels by either taking the mean accross the channels or taking the first item
        self.channel_alert(mean_channels,channel_index)
        figure_size_block = len(filters) * math.floor(len(filters[0,0,:,:]) / 2)
        plt.figure(figsize=(figure_size_block,figure_size_block))
        filters = self.conv_list[layer_index][0].weight
        print(f"plotting {len(filters)} filters")
        for index,fliter in enumerate(filters):
            if mean_channels == True:
                plot = fliter.detach().mean(dim=0)
            else:
                plot = filter[channel_index].detach()
            plt.subplot(len(plot),len(plot),index + 1,cmap=cmap)
            plt.axis('off')
            if save_kernels == True:
                plt.savefig(save_dir)
        plt.show()

           

             
                
    def __repr__(self):
        for item,index in enumerate(self.conv_list):
           return self.weights_list

    
    def Deconv(self,input: torch.tensor ,layer_index,filter_index):
        deconv_net = DeconvNet(self.model,input)
        deconv_map = deconv_net.deconv(filter_index,layer_index)
        #plot deconv map 
        