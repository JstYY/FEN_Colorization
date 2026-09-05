from torchvision import models
from collections import namedtuple
import torch
import torch.nn as nn

def vgg_preprocess(tensor):#函数对所有的图片完成rbg三个通道的根据设定值[0.485, 0.456, 0.406]与[0.229, 0.224, 0.225]的标准化
    #RGB张量通常指一个四维张量，这个张量的各个维度分别代表了批大小（batch size）、图像通道数（channel）、图像的高（height）和宽（width）
    #PyTorch中，单张图像和一批图像的张量在进行运算时，会进行广播（broadcasting）操作，使得mean_val和std_val可以和每张图片的每个像素进行操作。
    # input is RGB tensor which ranges in [0,1]
    # output is RGB tensor which ranges
    #计算平均值和标准差
    mean_val = torch.Tensor([0.485, 0.456, 0.406]).type_as(tensor).view(-1, 1, 1)
    std_val = torch.Tensor([0.229, 0.224, 0.225]).type_as(tensor).view(-1, 1, 1)
    tensor_norm = (tensor - mean_val) / std_val
    return tensor_norm

class vgg19(nn.Module):#继承于nn.Module
    
    def __init__(self, pretrained_path = "E:\Python_Project\Color2Embed-main\experiments\\vgg19-dcbb9e9d.pth", require_grad = False):
        #定义子类vgg19的构造函数，它接受两个参数：预训练模型的路径和一个表示是否需要计算梯度的布尔值。
        super(vgg19, self).__init__()
        #super()是一个函数，它会临时返回我们当前类的父类（或者多个父类）对象。
        # 在这里，vgg19是我们当前的类，self是当前类的实例。
        # .__init__() 则是调用这个父类的初始化函数。
        self.vgg_model = models.vgg19()
        #创建一个来自PyTorch的预定义VGG19模型实例，并将其赋值给了self.vgg_model。这里的 models.vgg19() 是PyTorch的模块，可以实例化预训练的VGG19模型。
        if pretrained_path != None:
            print('----load pretrained vgg19----')
            self.vgg_model.load_state_dict(torch.load(pretrained_path))#将预训练模型的参数（权重）加载到我们的 vgg19 对象中。
            print('----load done!----')
        self.vgg_feature = self.vgg_model.features
        #获取 vgg_model 中的特征提取部分（包括各卷积层和池化层）并将其赋给 self.vgg_feature。
        self.seq_list = [nn.Sequential(ele) for ele in self.vgg_feature]
        #将特征提取部分的每一层都包装成一个 nn.Sequential 对象，并将这些对象放入一个列表中。
        #nn.Sequential(ele)将每一个网络层单独封装成一个nn.Sequential对象,实现了对网络层的快捷访问。

        # self.vgg_layer = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 
        #                  'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        #                  'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        #                  'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        #                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']

        # self.vgg_layer = ['relu1_2', 'relu2_2', 'relu3_2', 'relu4_2', 'relu5_2']
        
        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
        #这行代码的意思就是遍历模型的所有参数。而在每次循环中，parameter就代表了模型的一个参数。
        #对于每个参数，我们将其 requires_grad 属性设置为 False，这样在进行反向传播的时候，这些参数的梯度就不会被计算。
        
    def forward(self, x, layer_name='relu5_2'):
        #定义了一个forward函数，这个函数接受一个输入 x 和一个表示需要提取特征的层（输出层）的名字 layer_name。
        ### x: RGB [0, 1], input should be [0, 1]
        x = vgg_preprocess(x)
        #标准化输入张量x，使其符合vgg19模型的输入要求。
        #conv1_1、conv1_2、...、conv5_4：表示VGG19模型中的卷积层的输出。
        # relu1_1、relu1_2、...、relu5_4：表示VGG19模型中的ReLU层的输出。
        # pool1、pool2、...、pool5：表示VGG19模型中的池化层的输出。

        conv1_1 = self.seq_list[0](x)
        relu1_1 = self.seq_list[1](conv1_1)
        conv1_2 = self.seq_list[2](relu1_1)
        relu1_2 = self.seq_list[3](conv1_2)
        pool1 = self.seq_list[4](relu1_2)
        
        conv2_1 = self.seq_list[5](pool1)
        relu2_1 = self.seq_list[6](conv2_1)
        conv2_2 = self.seq_list[7](relu2_1)
        relu2_2 = self.seq_list[8](conv2_2)
        pool2 = self.seq_list[9](relu2_2)
        
        conv3_1 = self.seq_list[10](pool2)
        relu3_1 = self.seq_list[11](conv3_1)
        conv3_2 = self.seq_list[12](relu3_1)
        relu3_2 = self.seq_list[13](conv3_2)
        conv3_3 = self.seq_list[14](relu3_2)
        relu3_3 = self.seq_list[15](conv3_3)
        conv3_4 = self.seq_list[16](relu3_3)
        relu3_4 = self.seq_list[17](conv3_4)
        pool3 = self.seq_list[18](relu3_4)
        
        conv4_1 = self.seq_list[19](pool3)
        relu4_1 = self.seq_list[20](conv4_1)
        conv4_2 = self.seq_list[21](relu4_1)
        relu4_2 = self.seq_list[22](conv4_2)
        conv4_3 = self.seq_list[23](relu4_2)
        relu4_3 = self.seq_list[24](conv4_3)
        conv4_4 = self.seq_list[25](relu4_3)
        relu4_4 = self.seq_list[26](conv4_4)
        pool4 = self.seq_list[27](relu4_4)
        
        conv5_1 = self.seq_list[28](pool4)
        relu5_1 = self.seq_list[29](conv5_1)
        conv5_2 = self.seq_list[30](relu5_1)
        relu5_2 = self.seq_list[31](conv5_2) # [B, 512, 16, 16]
        conv5_3 = self.seq_list[32](relu5_2)
        relu5_3 = self.seq_list[33](conv5_3)
        conv5_4 = self.seq_list[34](relu5_3)
        relu5_4 = self.seq_list[35](conv5_4)
        pool5 = self.seq_list[36](relu5_4) # [B, 512, 8, 8]
        
        # vgg_output = namedtuple("vgg_output", self.vgg_layer)
        
        # vgg_list = [conv1_1, relu1_1, conv1_2, relu1_2, pool1, 
        #                  conv2_1, relu2_1, conv2_2, relu2_2, pool2,
        #                  conv3_1, relu3_1, conv3_2, relu3_2, conv3_3, relu3_3, conv3_4, relu3_4, pool3,
        #                  conv4_1, relu4_1, conv4_2, relu4_2, conv4_3, relu4_3, conv4_4, relu4_4, pool4,
        #                  conv5_1, relu5_1, conv5_2, relu5_2, conv5_3, relu5_3, conv5_4, relu5_4, pool5]

        if layer_name == 'relu5_2':
            vgg_list = [relu5_2]
        elif layer_name == 'conv5_2':
            vgg_list = [conv5_2]
        elif layer_name == 'relu5_4':
            vgg_list = [relu5_4]
        elif layer_name == 'pool5':
            # print('pool5')
            vgg_list = [pool5]
        elif layer_name == 'all':
            vgg_list = [relu1_2, relu2_2, relu3_2, relu4_2, relu5_2]
        
        # out = vgg_output(*vgg_list)
        
        return vgg_list

class vgg19_class_fea(nn.Module):
    
    def __init__(self, pretrained_path = 'E:\Python_Project\Color2Embed-main\experiments\\vgg19-dcbb9e9d.pth', require_grad = False):
        super(vgg19_class_fea, self).__init__()
        self.vgg_model = models.vgg19()
        print('----load pretrained vgg19----')
        self.vgg_model.load_state_dict(torch.load(pretrained_path))
        print('----load done!----')
        self.vgg_feature = self.vgg_model.features
        self.avgpool = self.vgg_model.avgpool
        self.classifier = self.vgg_model.classifier

        self.seq_list = [nn.Sequential(ele) for ele in self.vgg_feature] # 37层
        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
        
    def forward(self, x):
        ### x: RGB [0, 1], input should be [0, 1]
        x = vgg_preprocess(x)

        for i in range(len(self.seq_list)):
            x = self.seq_list[i](x)
            if i == 31:
                relu5_2 = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_class = self.classifier(x)
        return x_class, relu5_2