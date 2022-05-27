import torch 
from torch import nn 
from RELUclasses import DeconvReLU

torch.use_deterministic_algorithms = True


class MaxUnpool2d_indexed(nn.Module):
    def __init__(self,index,max_pool:nn.Maxpool2d):
        super(MaxUnpool2d_indexed,self).__init__()
        self.unpool = nn.MaxUnpool2d(max_pool.kernel_size,max_pool.stride,max_pool.padding)
        self.index = index
    def forward(self,input):
        return self.unpool(self.index,input)


class DeconvNet():
    def __init__(self,model,input):
        self.input = input
        #get the size of the input through the forward pass through the CNN 
        self.forward_pass_sizes = [] 
        self.pooling_indices = []
        self.conv_model = model 
        self.max_positions = []
        self.deconv_model = []
        layers =  list(self.conv_model.children())
        for index,layer in enumerate(layers):
            if type(layer) == nn.MaxPool2d:
                self.pooling_indices.append(index)
        for index in self.pooling_indices:
            mini_model = nn.Sequential(*list(self.conv_model.module()[:index+1]))
            mini_model[-1].return_indices = True
            max_indices,_ = mini_model(input)
            self.max_positions.append((index,max_indices))
        
        for index,layer in enumerate(layers):
            if type(layer) == nn.Conv2d:
                mini_model = nn.Sequential(*list(self.conv_model.module()[:index]))
                output = mini_model(input)
                self.forward_pass_sized.append(output.size())

        
    
        #create deconv layer 
        deconv_model = layers.copy()
        deconv_model.reverse()
        for index,layer in enumerate(deconv_model):
            self.deconv_one(deconv_model,layer,index)
        self.deconv_model = nn.Sequential(*deconv_model)
        self.output = self.deconv_model

    def deconv_one(self,model,layer,index):
        if type(layer) == nn.ReLU:
            #replace relu with Deconv Relu
            model[index] = DeconvReLU()
        elif type(layer) == nn.Conv2d:
            model[index] = nn.ConvTranspose2d(model[index],output_size = self.forward_pass_sizes[-1])
            self.forward_pass_sizes.pop()
        elif type(layer) == nn.MaxPool2d:
            model[index] == MaxUnpool2d_indexed(self.pooling_indices.pop()) 
        else:
            pass
    
    def deconv(self,filter_index,layer_index):
        #index -> index of layer in original filter 
        ret_model = list(self.deconv_model.children())
        for index,filter in enumerate(ret_model[::-1][layer_index].weight):
            if index == filter_index:
                pass 
            else:
                #zero out all the activations except selected activation
                ret_model[::-1][layer_index].weight[index] = torch.zeros(ret_model[::-1][layer_index][index].shape)
        ret_model = nn.Sequential(*ret_model)
        return ret_model(input)
         
            
