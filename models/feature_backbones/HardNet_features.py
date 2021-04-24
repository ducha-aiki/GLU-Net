import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F
from copy import deepcopy
class DenseHardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self, _stride = 2):
        super(DenseHardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=_stride, padding=1,  bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=_stride, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Conv2d(128, 128, kernel_size=8, bias = False, padding=3),
            nn.BatchNorm2d(128, affine=False),
        )
        return
    def forward(self, input, upscale = False):
        b,ch,h,w = input.size()
        x = input
        if input.size(1) > 1:
            x = x.mean(dim = 1, keepdim = True)
        #std, mean = torch.std_mean(x, dim=(2,3))
        #x = (x - mean) / (std + 1e-8)
        feats = self.features(x)
        return F.normalize(feats, p=2, dim=1)
    
    

class HardNetPyramid(nn.Module):
    def __init__(self, train=False):
        super().__init__()
        self.n_levels = 5
        source_model = DenseHardNet()
        weights_dict = torch.load('checkpoint_liberty_with_aug.pth', map_location=torch.device('cpu'))
        source_model.load_state_dict(weights_dict['state_dict'])
        modules = OrderedDict()
        modules['level_0'] = nn.Sequential(*source_model.features[:6])
        modules['level_1'] = nn.Sequential(*source_model.features[6:12])
        modules['level_2'] = nn.Sequential(*source_model.features[12:18])
        modules['level_3'] = nn.Sequential(*source_model.features[19:])
        modules['level_3'][0].stride = (2,2)
        modules['level_4'] = nn.Sequential(nn.AvgPool2d(2,2))
        modules['level_5'] = nn.InstanceNorm2d(1, affine = False, track_running_stats=False)
        for i in range(self.n_levels+1):
            for param in modules['level_' + str(i)].parameters():
                param.requires_grad = train
        self.__dict__['_modules'] = modules

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        outputs = []
        b,ch,h,w = x.size()
        x = x
        if x.size(1) > 1:
            x = x.mean(dim = 1, keepdim = True)
        x = self.__dict__['_modules']['level_5'](x)

        if quarter_resolution_only:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            outputs.append(x_full)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
            x_eight = self.__dict__['_modules']['level_' + str(3)](x_quarter)
            outputs.append(x_eight)
        else:
            for layer_n in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                outputs.append(x)

            if float(torch.__version__[:3]) >= 1.6:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area', recompute_scale_factor=True)
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            outputs.append(x)

            if float(torch.__version__[:3]) >= 1.6:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area', recompute_scale_factor=True)
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            outputs.append(x)
        return outputs
class HardNetPyramid2(nn.Module):
    def __init__(self, train=False):
        super().__init__()
        self.n_levels = 5
        source_model = DenseHardNet()
        weights_dict = torch.load('checkpoint_liberty_with_aug.pth', map_location=torch.device('cpu'))
        source_model.load_state_dict(weights_dict['state_dict'])

        modules = OrderedDict()
        modules['level_0'] = deepcopy(nn.Sequential(*source_model.features[:6]))
        modules['level_0'][0].padding = (1,1)
        modules['level_1'] = nn.Sequential(*source_model.features[6:12])
        modules['level_all'] = source_model
        modules['level_all'].features[0].padding = (2,2)
        modules['level_5'] = nn.InstanceNorm2d(1, affine = False, track_running_stats=False)
        for i in range(2):
            for param in modules['level_' + str(i)].parameters():
                param.requires_grad = train
        for param in modules['level_all'].parameters():
            param.requires_grad = train
        for param in modules['level_5'].parameters():
            param.requires_grad = train
        print (modules)
        self.__dict__['_modules'] = modules

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        outputs = []
        b,ch,h,w = x.size()
        x = x
        if x.size(1) > 1:
            x = x.mean(dim = 1, keepdim = True)
        x = self.__dict__['_modules']['level_5'](x)
        if quarter_resolution_only:
            x_quarter = self.__dict__['_modules']['level_all'](x)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_full = self.__dict__['_modules']['level_0'](x)
            outputs.append(x_full)
            x_quarter = self.__dict__['_modules']['level_all'](x)
            outputs.append(x_quarter)
            if float(torch.__version__[:3]) >= 1.6:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area', recompute_scale_factor=True)
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            x_eight = self.__dict__['_modules']['level_all'](x)
            outputs.append(x_eight)
        else:
            x0 = self.__dict__['_modules']['level_0'](x)
            outputs.append(x0)
            x1 = self.__dict__['_modules']['level_1'](x0)
            outputs.append(x1)
            x_down = x
            for i in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_all'](x_down)
                outputs.append(x)
                if float(torch.__version__[:3]) >= 1.6:
                    x_down = torch.nn.functional.interpolate(x_down, scale_factor=0.5, mode='area', recompute_scale_factor=True)
                else:
                    x_down = torch.nn.functional.interpolate(x_down, scale_factor=0.5, mode='area')
        return outputs

class HardNetPyramid3(nn.Module):
    def __init__(self, train=False):
        super().__init__()
        self.n_levels = 5
        source_model = DenseHardNet()
        weights_dict = torch.load('checkpoint_liberty_with_aug.pth', map_location=torch.device('cpu'))
        source_model.load_state_dict(weights_dict['state_dict'])

        modules = OrderedDict()
        modules['level_0'] = deepcopy(nn.Sequential(*source_model.features[:6]))
        modules['level_0'][0].padding = (1,1)
        modules['level_1'] = nn.Sequential(*source_model.features[6:12])
        modules['level_all'] = source_model
        modules['level_all'].features[0].padding = (2,2)
        modules['level_5'] = nn.InstanceNorm2d(1, affine = False, track_running_stats=False)
        for i in range(2):
            for param in modules['level_' + str(i)].parameters():
                param.requires_grad = train
        for param in modules['level_all'].parameters():
            param.requires_grad = train
        for param in modules['level_5'].parameters():
            param.requires_grad = train
        print (modules)
        self.__dict__['_modules'] = modules

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        outputs = []
        b,ch,h,w = x.size()
        x = x
        if x.size(1) > 1:
            x = x.mean(dim = 1, keepdim = True)
        x = self.__dict__['_modules']['level_5'](x)
        if quarter_resolution_only:
            x_quarter = self.__dict__['_modules']['level_all'](x)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_full = self.__dict__['_modules']['level_0'](x)
            outputs.append(x_full)
            x_quarter = self.__dict__['_modules']['level_all'](x)
            outputs.append(x_quarter)
            if float(torch.__version__[:3]) >= 1.6:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area', recompute_scale_factor=True)
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            x_eight = self.__dict__['_modules']['level_all'](x)
            outputs.append(x_eight)
        else:
            x0 = self.__dict__['_modules']['level_0'](x)
            outputs.append(x0)
            x1 = self.__dict__['_modules']['level_1'](x0)
            outputs.append(x1)
            x_down = x
            for i in range(0, self.n_levels-2):
                x = self.__dict__['_modules']['level_all'](x_down)
                outputs.append(x)
                if float(torch.__version__[:3]) >= 1.6:
                    x_down = torch.nn.functional.interpolate(x_down, scale_factor=0.5, mode='area', recompute_scale_factor=True)
                else:
                    x_down = torch.nn.functional.interpolate(x_down, scale_factor=0.5, mode='area')
            for i in range(2):
                xcur = outputs[-1]
                if float(torch.__version__[:3]) >= 1.6:
                    xcur = torch.nn.functional.interpolate(xcur, scale_factor=0.5, mode='area', recompute_scale_factor=True)
                else:
                    xcur = torch.nn.functional.interpolate(xcur, scale_factor=0.5, mode='area')
                outputs.append(xcur)
        return outputs
