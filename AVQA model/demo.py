import os
import math
import scipy
import skvideo
import skvideo.io
import torchaudio
import numpy as np
from torch import nn
from PIL import Image
from scipy.io import wavfile
from torchvision import transforms
from torch.utils.data import Dataset
_PI = np.pi

dis_video_path = './test_av_seq/'
dis_audio_path = './test_av_seq/'

dis_video_name = ['Boxing_QP50_S.yuv']
dis_audio_name = ['Boxing_8kbps.wav']

SDdatainfo = './VisualSaliency/V_position_single.mat'
ckpt_path = './checkpoints/epoch9'


position_width = []
position_height = []
height = 1080
width = 1920
dis_patch = 200
patchSize = 224
patchNum = 25

SDInfo = scipy.io.loadmat(SDdatainfo)
for h in range(0, height, dis_patch):
    if h < height - patchSize + 1:
        for w in range(0, width, dis_patch):
            if w < width - patchSize:
                position_width.append(w)
                position_height.append(h)
            else:
                position_height.append(h)
                position_width.append(width - patchSize)
                break
    else:
        for w in range(0, width, dis_patch):
            if w < width - patchSize:
                position_height.append(height - patchSize)
                position_width.append(w)
            else:
                position_height.append(height - patchSize)
                position_width.append(width - patchSize)
                break
        break

position = [position_height, position_width]


class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self,  dis_videos_dir, dis_video_names, video_format='RGB', width=None, height=None, position=None, SDInfo=None):
        super(VideoDataset, self).__init__()
        self.dis_videos_dir = dis_videos_dir
        self.dis_video_names = dis_video_names
        self.format = video_format
        self.width = width
        self.height = height
        self.position = position
        self.SDInfo = SDInfo

    def __len__(self):
        return len(self.dis_video_names)

    def __getitem__(self, idx):
        dis_video_name = self.dis_video_names[idx]

        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            dis_video_data = skvideo.io.vread(os.path.join(self.dis_videos_dir, dis_video_name), self.width, self.height,inputdict={'-pix_fmt': 'yuv420p'})
        else:
            dis_video_data = skvideo.io.FFmpegReader(os.path.join(self.dis_videos_dir, dis_video_name), self.height, self.width,inputdict={'-pix_fmt': 'rgb24'})

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        video_channel = dis_video_data.shape[3]
        video_height = dis_video_data.shape[1]
        video_width = dis_video_data.shape[2]

        video_length = 192
        print('video length: ', video_length)

        transformed_dis_video = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            dis_frame = dis_video_data[frame_idx]
            dis_frame = Image.fromarray(dis_frame)
            dis_frame = transform(dis_frame)
            transformed_dis_video[frame_idx] = dis_frame
        sample = {'dis_video_name': dis_video_name, 'dis_video': transformed_dis_video, 'sal_index': SDInfo['sal_index']}
        return sample


def melSpectrogram(audiofile):
    waveform, sr = torchaudio.load(audiofile)
    waveform = waveform[0, :].unsqueeze(0)

    mel_bins = 224
    target_length = 1597
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=5)

    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    fbank = fbank.numpy()

    windowsize = round(sr * 0.02)
    overlap = 0.75
    window_overlap = int(windowsize * overlap)
    num_blocks = int((waveform.size(1) - window_overlap) / (windowsize - window_overlap))
    t_sp = np.empty(num_blocks, dtype=float)
    for i in range(num_blocks):
        t_sp[i] = (i * (windowsize - window_overlap) + windowsize / 2) / sr
    return fbank, t_sp


class AudioDataset(Dataset):
    def __init__(self, audios_dir, audios_names, audio_frameRate):
        super(AudioDataset, self).__init__()
        self.audios_dir = audios_dir
        self.audios_names = audios_names
        self.audios_frameRate = audio_frameRate

    def __len__(self):
        return len(self.audios_names)

    def __getitem__(self, idx):
        audios_names = self.audios_dir + self.audios_names[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        [mel_spectrogram, T] = melSpectrogram(audios_names)
        transforms_audio = transform(mel_spectrogram)
        transforms_audio = transforms_audio.permute(0, 2, 1)

        sample = {'audio_name': self.audios_names[idx], 'audio': transforms_audio, 'tStamp': T, 'frameRate': 24}
        return sample


from detectron2.config import get_cfg
from config import add_vovnet_config
from vovnet import VoVNet
from detectron2.model_zoo import get_config_file
from detectron2.engine import default_argument_parser, default_setup


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    add_vovnet_config(cfg)
    cfg.merge_from_file(get_config_file("mask_rcnn_V_57_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = "vovnet57_ese_detectron2.pth"
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)


def get_features(dis_video_data, position, sal_index, frame_length, frame_interval, patchSize, patchNum, device='cpu'):
    """feature extraction"""
    args1 = default_argument_parser().parse_args()
    extractor = VoVNet(cfg=setup(args1), input_ch=3, out_features=["stage2", "stage3", "stage4", "stage5"])
    ckpt = torch.load('./checkpoints/vovnet57_ese_detectron2.pth', map_location=torch.device('cpu'))
    state_dict = {}

    prefix_to_remove = "backbone.bottom_up."

    for key, value in ckpt.items():
        if key.startswith(prefix_to_remove):
            new_key = key[len(prefix_to_remove):]
            state_dict[new_key] = value
        else:
            state_dict[key] = value

    extractor.load_state_dict(state_dict, False)
    extractor = extractor.to(device)
    extractor.eval()
    dis_output = torch.Tensor().to(device)

    ipatch = 0
    with torch.no_grad():
        for iframe in range(0, frame_length, frame_interval):
            sal_row = int(iframe / 2)
            dis_output1 = torch.Tensor().to(device)
            dis_output2 = torch.Tensor().to(device)
            for idx in range(patchNum):
                patch_idx = sal_index[sal_row, idx]
                dis_batch = dis_video_data[iframe:iframe + 1, 0:3,
                            position[0][patch_idx]:position[0][patch_idx] + patchSize,
                            position[1][patch_idx]:position[1][patch_idx] + patchSize].to(device)

                dis_features = extractor(dis_batch)
                dis_features = dis_features['stage5']
                dis_features_mean = nn.functional.adaptive_avg_pool2d(dis_features, 1)
                dis_features_std = global_std_pool2d(dis_features)
                dis_output1 = torch.cat((dis_output1, dis_features_mean), 0)
                dis_output2 = torch.cat((dis_output2, dis_features_std), 0)

                ipatch = ipatch + 1
                print('\r iframe: {} ipatch: {} ' .format(iframe, ipatch), end=' ')

            dis_output = torch.cat((dis_output, torch.cat((dis_output1.mean(axis=0, keepdim=True), dis_output2.mean(axis=0, keepdim=True)), 1)), 0)
            # print(dis_output.shape)
            ipatch = 0
        dis_output = dis_output.squeeze()
    return dis_output
    

def get_Afeatures(audios_data, audio_tStamp, frameRate, frame_interval, device='cpu'):
    """feature extraction"""
    args1 = default_argument_parser().parse_args()
    extractor = VoVNet(cfg=setup(args1), input_ch=3, out_features=["stage2", "stage3", "stage4", "stage5"])
    ckpt = torch.load('./checkpoints/vovnet57_ese_detectron2.pth', map_location=torch.device('cpu'))
    state_dict = {}

    prefix_to_remove = "backbone.bottom_up."

    for key, value in ckpt.items():
        if key.startswith(prefix_to_remove):
            new_key = key[len(prefix_to_remove):]
            state_dict[new_key] = value
        else:
            state_dict[key] = value

    extractor.load_state_dict(state_dict, False)
    extractor = extractor.to(device)

    dis_output1 = torch.Tensor().to(device)
    dis_output2 = torch.Tensor().to(device)
    extractor.eval()
    patchSize = 224
    with torch.no_grad():
        for iFrame in range(1, int(frameRate * 8), frame_interval):
            tCenter = np.argmin(abs(audio_tStamp - iFrame / frameRate))
            tStart = tCenter - patchSize / 2 + 1
            tEnd = tCenter + patchSize / 2
            if tStart < 1:
                tStart = 1
                tEnd = patchSize
            else:
                if tEnd > audios_data.shape[2]:
                    tStart = audios_data.shape[2] - patchSize + 1
                    tEnd = audios_data.shape[2]
            specRef_patch = audios_data[:, :, int(tStart - 1): int(tEnd)]
            refRGB = torch.cat((specRef_patch, specRef_patch, specRef_patch), 0)

            last_batch = refRGB.view(1, 3, specRef_patch.shape[1], specRef_patch.shape[2]).float().to(device)

            dis_features_VoVNet57 = extractor(last_batch)
            dis_features = dis_features_VoVNet57['stage5']  # .mean(dim=1, keepdim=True)
            dis_features_mean = nn.functional.adaptive_avg_pool2d(dis_features, 1)
            dis_features_std = global_std_pool2d(dis_features)
            dis_output1 = torch.cat((dis_output1, dis_features_mean), 0)
            dis_output2 = torch.cat((dis_output2, dis_features_std), 0)

        dis_output = torch.cat((dis_output1, dis_output2), 1).squeeze()
    return dis_output


'----------------------------------------A/V Feature Fusion------------------------------------------------'


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()

        self.encoder1 = nn.Linear(2048, 512) # 512 128 -> 2048 512
        self.encoder2 = nn.Linear(2048, 512) # 512 128 -> 2048 512

        self.affine_a = nn.Linear(24, 24, bias=False) # 8 8 -> 24 24
        self.affine_v = nn.Linear(24, 24, bias=False) # 8 8 -> 24 24

        self.W_a = nn.Linear(24, 96, bias=False) # 8 32 -> 24 96
        self.W_v = nn.Linear(24, 96, bias=False) # 8 32 -> 24 96
        self.W_ca = nn.Linear(1024, 96, bias=False) # 256 32 -> 1024 96
        self.W_cv = nn.Linear(1024, 96, bias=False) # 256 32 -> 1024 96

        self.W_ha = nn.Linear(96, 24, bias=False) # 32 8 -> 96 24
        self.W_hv = nn.Linear(96, 24, bias=False) # 32 8 -> 96 24

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.regressor = nn.Sequential(nn.Linear(1024, 512), nn.Dropout(0.6), nn.Linear(512, 1))
                                                                    # 256 128 128 1 -> 1024 512 512 1
        self.fc3 = nn.Linear(24, 16)  # 24 16
        self.relu1 = nn.ReLU()
        self.dro1 = nn.Dropout()
        self.fc4 = nn.Linear(16, 1) # 16 1

    def forward(self, f1_norm, f2_norm):

        sequence_outs = []

        for i in range(f1_norm.shape[0]):
            audfts = f1_norm[i, :, :]#.transpose(0,1)
            visfts = f2_norm[i, :, :]#.transpose(0,1)

            aud_fts = self.encoder1(audfts)
            vis_fts = self.encoder2(visfts)
            aud_vis_fts = torch.cat((aud_fts, vis_fts), 1)
            a_t = self.affine_a(aud_vis_fts.transpose(0, 1))
            att_aud = torch.mm(aud_fts.transpose(0, 1), a_t.transpose(0, 1))
            audio_att = self.tanh(torch.div(att_aud, math.sqrt(aud_vis_fts.shape[1])))

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 1)
            v_t = self.affine_v(aud_vis_fts.transpose(0, 1))
            att_vis = torch.mm(vis_fts.transpose(0, 1), v_t.transpose(0, 1))
            vis_att = self.tanh(torch.div(att_vis, math.sqrt(aud_vis_fts.shape[1])))

            H_a = self.relu(self.W_ca(audio_att) + self.W_a(aud_fts.transpose(0, 1)))
            H_v = self.relu(self.W_cv(vis_att) + self.W_v(vis_fts.transpose(0, 1)))

            att_audio_features = self.W_ha(H_a).transpose(0, 1) + aud_fts
            att_visual_features = self.W_hv(H_v).transpose(0, 1) + vis_fts

            audiovisualfeatures = torch.cat((att_audio_features, att_visual_features), 1)
            outs = self.regressor(audiovisualfeatures) #.transpose(0, 1))

            sequence_outs.append(outs)
        final_outs = torch.stack(sequence_outs)

        final_outs = final_outs.squeeze(2)
        final_outs = self.fc3(final_outs)
        final_outs = self.relu1(final_outs)
        final_outs = self.dro1(final_outs)
        final_outs = self.fc4(final_outs)
        return final_outs


'-----------------------------------------Main Progress------------------------------------------------'
import time
import torch

dis_frameRate = 24
dis_audio_dataset = AudioDataset(dis_audio_path, dis_audio_name, dis_frameRate)
dis_dataset = VideoDataset(dis_video_path, dis_video_name, 'YUV420', height, width, position, SDInfo)

current_dis_video = dis_dataset[0]
current_dis_video = current_dis_video['dis_video']

sal_index = dis_dataset[0]
sal_index = sal_index['sal_index']

frame_length = 192
frame_interval = 2

current_data = dis_audio_dataset[0]
current_audio = current_data['audio']
current_tStamp = current_data['tStamp']

start_time = time.time()

Vfeatures = get_features(current_dis_video, position, sal_index, frame_length, frame_interval, patchSize, patchNum, device='cpu')
Afeatures = get_Afeatures(current_audio, current_tStamp, current_data['frameRate'], frame_interval, device='cpu')

Vfeatures = Vfeatures.unsqueeze(0)
Afeatures = Afeatures.unsqueeze(0)

V_chunks = torch.chunk(Vfeatures, 4, dim=1)
A_chunks = torch.chunk(Afeatures, 4, dim=1)

model = CAM()
state_dict = torch.load(ckpt_path)
model.load_state_dict(state_dict)
print('loaded pretrained model')
model.eval()
scores = []
with torch.no_grad():
    for V_chunk, A_chunk in zip(V_chunks, A_chunks):
        score = model(V_chunk, A_chunk)
        scores.append(score)

average_score = sum(scores) / len(scores)

end_time = time.time()

print('Final A/V quality score:', average_score*100)
time_consumption = end_time - start_time
print(f"Time cost: {time_consumption} sec.")
