from gc import is_finalized
import torch
from torch import nn 
import torch.nn.functional as F

torch.use_deterministic_algorithms = true

class myDeconvReLU(torch.autograd.function):
    def __init__(self):
        

        @staticmethod 
        def forward(ctx:torch.autograd.function.FunctionCtx,input):
            #forward pass for DeconvReLU and Normal ReLU are the same
            ctx.save_for_backward(input)
            return input.clamp(min=0)


        @staticmethod
        def backward(ctx,grad_output):
            #run ReLU on the error signal
            input = ctx.saved_tensors()
            grad_input = grad_output.clone()
            return F.relu(grad_input)



class DeconvReLU(nn.Module):
    def __init__(self):

        super(myDeconvReLU,self).__init__()
        self.relu = myDeconvReLU.apply
    def forward(self,input):
        return self.relu(input)





class myGuidedBackpropReLU(torch.autograd.function):
    def __init__(self):
        

        @staticmethod 
        def forward(ctx:torch.autograd.function.FunctionCtx,input):
            #forward pass for DeconvReLU and Normal ReLU are the same
            ctx.save_for_backward(input)
            return input.clamp(min=0)


        @staticmethod
        def backward(ctx,grad_output):
            #run ReLU on the error signal
            input = ctx.saved_tensors()
            grad_input = grad_output.clone() #i think we had to copy because the value is still being used by the rest of the model
            grad_input[input < 0] = 0
            grad_input_conv = grad_output.clone()
            return F.relu(grad_input_conv) * grad_input 




class GuidedBackpropReLU(nn.Module):
    def __init__(self):

        super(myDeconvReLU,self).__init__()
        self.relu = myGuidedBackpropReLU.apply
    def forward(self,input):
        return self.relu(input)




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
        else:
            pass
    
    def deconv(self,filter_index,layer_index):
        for index in range(len(self.deconv_model.weights[::-1]))
