import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import sys

class alexnet_features(nn.Module):
    def __init__(self, output_n, imgae_size):
        super(alexnet_features, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True).features  # 21.69/5.94

    def forward(self, x):
        x = self.alexnet(x)
        return x

class resnet152_1d(nn.Module):
    def __init__(self):
        super(resnet152_1d, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = torchvision.models.resnet152(pretrained=False)  # 21.69/5.94

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        return x

class densenet161_1d(nn.Module):
    def __init__(self):
        super(densenet161_1d, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.densenet = torchvision.models.densenet161(pretrained=False)  # 22.35/6.20

    def forward(self, x):
        x = self.conv(x)
        x = self.densenet(x)
        return x

class densenet201_1d(nn.Module):
    def __init__(self, output_n, image_size, dim):
        super(densenet201_1d, self).__init__()
        self.dim = dim
        if dim != 3:
            self.conv = nn.Conv2d(dim, 3, kernel_size=1)
        self.densenet = torchvision.models.densenet201(pretrained=True)
        # # del the last layer
        removed = list(self.densenet.classifier.children())[:-1]
        self.densenet.classifier = nn.Sequential(*removed)

        output_size = self.linear_input((3, image_size, image_size))

        self.fc = nn.Sequential(
            nn.Linear(output_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 125),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(125, output_n)
        )

    def forward(self, x):
        if self.dim != 3:
            x = self.conv(x)

        # # visualize---------------------
        # for name, midlayer in self.densenet._modules.items():
        #     test = x
        #     for i in range(len(midlayer)):
        #         test = midlayer[i](test)
        #         # if i == 4:
        #         #     break
        #         #     # print(i, test.shape)
        #     return test
        # -------------------------------------

        x = self.densenet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

    def linear_input(self, shape):
        bs = 2
        input = Variable(t.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        out = self.densenet(x)
        return out


class vgg19_1d(nn.Module):
    def __init__(self):
        # sys.setrecursionlimit(int(2e4))
        super(vgg19_1d, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.vgg = torchvision.models.vgg19_bn(pretrained=False)  # 25.766/8.15

    def forward(self, x):
        x = self.conv(x)
        x = self.vgg(x)
        return x


class mini_AE(nn.Module):
    def __init__(self):
        super(mini_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        # define: decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.ConvTranspose2d(64, 128, 2, 2),
            nn.ConvTranspose2d(128, 3, 2, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class VAE(nn.Module):
    def __init__(self, dim, imgae_size, pretrain):
        super(VAE, self).__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),
            )
        output_size = self.linear_input((dim, imgae_size, imgae_size))
        self.dnn_encoder = nn.Sequential(
            nn.Linear(output_size, 100),
            )

        # VAE: These two layers are for getting logvar and mean
        self.mean = nn.Linear(100, 64)
        self.var = nn.Linear(100, 64)

        self.dnn_decoder = nn.Sequential(
            nn.Linear(64, output_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            )
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.RReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.RReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=dim, kernel_size=2, stride=2, padding=1),
            nn.Tanh()
        )
        self.cnn_encoder.apply(gaussian_weights_init)
        self.dnn_encoder.apply(gaussian_weights_init)
        self.dnn_decoder.apply(gaussian_weights_init)
        self.cnn_decoder.apply(gaussian_weights_init)
        self.pretrain = pretrain


    def forward(self, out):
        # encoder
        out = self.cnn_encoder(out)
        cnn_shape = out.shape
        out = out.view(out.size()[0], -1)
        out= self.dnn_encoder(out)
        encoder, var = self.de_noise(out)
        # decoder
        out = self.dnn_decoder(encoder)
        out = out.view([-1, cnn_shape[1], cnn_shape[2], cnn_shape[3]])
        decoder = self.cnn_decoder(out)
        return encoder, var, decoder

    def linear_input(self, shape):
        bs = 1
        input = Variable(t.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        out = self.cnn_encoder(x)
        return out

    def de_noise(self, out):
        mean = self.mean(out)
        var = self.var(out)
        if self.pretrain:
            return mean, var
        noise = t.randn_like(mean)
        std = t.exp(0.5 * var)
        return noise.mul(std).add_(mean), var




class Autoencoder(nn.Module):
    def __init__(self, dim, imgae_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.RReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.RReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=dim, kernel_size=2, stride=2, padding=1),
            nn.Tanh()
        )


        self.encoder.apply(gaussian_weights_init)
        self.decoder.apply(gaussian_weights_init)

    def forward(self, out):
        encoder = self.encoder(out)
        decoder = self.decoder(encoder)


        # out = out.view(out.size()[0], -1)
        # out = self.fc(out)
        return encoder, decoder

    def linear_input(self, shape):
        bs = 1
        input = Variable(t.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        out = self.cnn(x)
        return out



# LeNet-5
class ConvNet(nn.Module):
    def __init__(self, output_n, dim, imgae_size):
        super(ConvNet, self).__init__()
        self.dim = dim
        if dim != 3:
            self.conv = nn.Conv2d(dim, 3, kernel_size=1)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        output_size = self.linear_input((3, imgae_size, imgae_size))

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=output_size, out_features=120),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=120, out_features=84),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=84, out_features=output_n),
            )

        self.cnn.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        if self.dim != 3:
            x = self.conv(x)
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def linear_input(self, shape):
        bs = 1
        input = Variable(t.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        out = self.cnn(x)
        return out

    def name(self):
        return "ConvNet"

class Fully(nn.Module):
    def __init__(self, output_n, imgae_size):
        super(Fully, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=imgae_size*imgae_size*3, out_features=1024),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=64),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=16, out_features=output_n),
        )
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = x
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def name(self):
        return "Fully"


class ResidualBlock( nn.Module ):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super( ResidualBlock, self ).__init__()
        # normal
        self.left = nn.Sequential(
            nn.Conv2d( inchannel, outchannel, 3, stride, 1, bias=False ),
            nn.BatchNorm2d( outchannel ),
            nn.ReLU( inplace=True ),
            nn.Conv2d( outchannel, outchannel, 3, 1, 1, bias=False ),
            nn.BatchNorm2d( outchannel ) )
        # skip the layer
        self.right = shortcut

    def forward(self, x):
        out = self.left( x )
        residual = x if self.right is None else self.right( x )
        out += residual
        return F.relu( out )


class ResNet( nn.Module ):
    def __init__(self, num_classes=1000):
        super( ResNet, self ).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d( 3, 64, 7, 2, 3, bias=False ),
            nn.BatchNorm2d( 64 ),
            nn.ReLU( inplace=True ),
            nn.MaxPool2d( 3, 2, 1 ) )

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer( 64, 64, 3 )
        self.layer2 = self._make_layer( 64, 128, 4, stride=2 )
        self.layer3 = self._make_layer( 128, 256, 6, stride=2 )
        self.layer4 = self._make_layer( 256, 512, 3, stride=2 )

        # 分类用的全连接
        self.fc = nn.Linear( 512, num_classes )

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        构建layer,包含多个residual block
        '''
        shortcut = nn.Sequential(
            nn.Conv2d( inchannel, outchannel, 1, stride, bias=False ),
            nn.BatchNorm2d( outchannel ) )

        layers = []
        layers.append( ResidualBlock( inchannel, outchannel, stride, shortcut ) )

        for i in range( 1, block_num ):
            layers.append( ResidualBlock( outchannel, outchannel ) )
        return nn.Sequential( *layers )

    def forward(self, x):
        x = self.pre( x )

        x = self.layer1( x )
        x = self.layer2( x )
        x = self.layer3( x )
        x = self.layer4( x )

        x = F.avg_pool2d( x, 7 )
        x = x.view( x.size( 0 ), -1 )
        return self.fc( x )

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


class TA_2018_Classifier(nn.Module):
    def __init__(self, output_n, imgae_size):
        super(TA_2018_Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # [64, 24, 24]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0)       # [256, 3, 3]
        )

        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, output_n)
        )

        self.cnn.apply(gaussian_weights_init)
        # self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        # out = self.fc(out)
        return out