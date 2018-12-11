# Hector Rabago and Eduardo Despradel
# W4731 Computer Vision Final Project - Artistic Style Transfer
# Keras implementation of Artistic Style Transfer as described by Gatys et al 2015/6

# NOTE: keras.image_data_format assumed to be channels last

import sys
import time
import numpy as np
from keras.applications import vgg19
from keras import backend as K

from keras.preprocessing.image import load_img, save_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b



OUT_SHAPE = (224,224)

N,M = OUT_SHAPE

LAYERS = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']



class NeuralStyleTransfer():

    def __init__(self,contentPath,stylePath,outPath):

        self.contentPath = contentPath
        self.stylePath = stylePath
        self.outPath = outPath

        self.loss_value = None
        self.grads_values = None

        self.outShape = OUT_SHAPE

        # process the input image to be keras tensor variable for vgg net
        self.contentTensor = K.variable(self.imgToTensor(contentPath))
        self.styleTensor = K.variable(self.imgToTensor(stylePath))
        self.finalTensor = K.placeholder((1, *OUT_SHAPE, 3))

        self.mainTensor = K.concatenate([self.contentTensor,\
                                        self.styleTensor,\
                                        self.finalTensor], axis=0)

        #alhpa and beta for the total loss equation
        # totalLoss = alpha * contentLoss + beta * styleLoss
        self.alpha = 0.05
        self.beta = 5.0

        #building VGG 19 pretrained from imagenet and setting up a dictionary the layers
        self.VGG19 = vgg19.VGG19(input_tensor=self.mainTensor,weights='imagenet', include_top=False)
        self.layersDict = dict([(layer.name, layer.output) for layer in self.VGG19.layers])


        # combine these loss functions into a single scalar
        self.L_total = K.variable(0.0)
        self.block5_conv2_features = self.layersDict['block5_conv2']
        self.contentRepresentation = self.block5_conv2_features[0, :, :, :]
        self.outputRepresentation = self.block5_conv2_features[2, :, :, :]

        # initializing the content loss to be the SSR of the difference between content representation
        # and whitenoise image representation /out image just like the formula in the paper
        self.L_content = K.sum(K.square(self.contentRepresentation - self.outputRepresentation))
        self.L_total += self.alpha * self.L_content


        # iterate over all the layers of the vgg and add up the style loss across
        # the layers to the total loss
        for layer in LAYERS:

            # 
            self.blockFeatures = self.layersDict[layer]
            self.styleFeatures = self.blockFeatures[1, :, :, :]
            self.bothFeatures = self.blockFeatures[2, :, :, :]

            # calculating the gram matrixes
            self.gramStyle = self.G(self.styleFeatures)
            self.gramContent = self.G(self.bothFeatures)
            self.size = N**2
            #getting the SSR of between the G_style and G_content
            #just like formula 4 from the 2015 Gatys paper
            self.L_style = K.sum(K.square(self.gramStyle - self.gramContent)) / (4.0 * (3 ** 2) * (self.size ** 2))
            self.L_total += (self.beta / len(LAYERS)) * self.L_style

        # getting derivatives of the tensor with respective to the total Loss, L_total
        self.grads = K.gradients(self.L_total, self.finalTensor)


        # setting the output values for the total loss and adding the gradients
        self.outputs = [self.L_total]
        self.outputs += self.grads

        # self.features = K.function([finalTensor], self.outputs)
        self.features = K.function([self.finalTensor], self.outputs)


    #this function computes the loss and gradient values for the input Keras tensor x
    def L_dK(self,x):
        x = x.reshape((1, *OUT_SHAPE, 3))
        outs = self.features([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values



    # takes and image path as input, and the output shape
    # outputs the Keras VGG tensor representation of the image
    def imgToTensor(self,imgPath,shape=OUT_SHAPE):
        kerasImg = load_img(imgPath, target_size=shape)
        numpyImg = np.expand_dims(img_to_array(kerasImg), axis=0)
        tensor = vgg19.preprocess_input(numpyImg)
        return tensor

    # from keras tensor to img 
    def tensorToImg(self,tensorX):

        tensorX = tensorX.reshape((*OUT_SHAPE, 3))
        #converting brg to rgb tensor for output

        # removing mean value to make the final image brighter as
        # was suggested by stack over flow
        tensorX[:, :, 0] += 100.0
        tensorX[:, :, 1] += 110.0
        tensorX[:, :, 2] += 120.0

        tensorX = tensorX[:, :, ::-1]
        tensorX = np.clip(tensorX, 0, 255).astype('uint8')
        return tensorX

    def lossDescent(self, x):
        loss_value, grad_values = self.L_dK(x)
        # loss_value = loss_value
        # grad_values = grad_values
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def gradsDescent(self, x):
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


    # input: a three dimension tensor, out: gram matrix for tensor
    def G(self,x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

def gradientDescent(NeuralTransfer, epochsCount):

    # convert content image to keras tensor
    x = NeuralTransfer.imgToTensor(NeuralTransfer.contentPath)
    for i in range(epochsCount):

        # we found this function very useful to perform gradient descent
        x, min_val, info = fmin_l_bfgs_b(NeuralTransfer.lossDescent, x.flatten(),fprime=NeuralTransfer.gradsDescent, maxfun=20)
        # savging image with style transferred
        img = NeuralTransfer.tensorToImg(x.copy())
        # img = NeuralTransfer.tensorToImg(x

        #saving a generated image at each epoch
        fname = NeuralTransfer.outPath +str(i)+".png"
        save_img(fname, img)

if __name__ == "__main__":

    # print(len(sys.argv))
    try:

        assert len(sys.argv) == 4

    except AssertionError as error:
        print("Incorrect Usage")
        print("Usage:python neuralStylerTransfer.py contentPath stylePath outPath/out")

    else:

        contentPath = sys.argv[1]
        stylePath = sys.argv[2]
        outPath = sys.argv[3]
        # print(contentPath,stylePath,outPath)
        transfer = NeuralStyleTransfer(contentPath,stylePath,outPath)
        gradientDescent(transfer,10)




