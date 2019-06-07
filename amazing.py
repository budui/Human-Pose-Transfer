from util.sample.pose import interpolation
from test.DPIG.sample_pose import _load_model
from util.util import show_with_visibility as show_pose
import torch
import torch.backends.cudnn as cudnn
from models.DPIG import PoseDecoder, PoseEncoder
from dataset.key_point_dataset import KeyPointDataset



device = "cpu"
cudnn.benchmark = True
torch.cuda.set_device(0)
print(torch.cuda.current_device())
print(torch.cuda.device_count())
