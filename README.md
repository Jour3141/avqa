# avqa
***IEEE TCSVT2026 🎉🎉***: [Multi-modal Cross-Attention Guided Network for Audio-Visual Quality Evaluation via Visual Saliency and Mel-spectrum Features](https://ieeexplore.ieee.org/document/11345165)

### 💬 Introduction
We proposed a **no-reference audio-visual quality assessment (NR-AVQA)** method that explicitly models cross-modal interactions.

* Leverage visual saliency to focus on human-perceptually important regions.
* Use Mel-spectrum to align with human auditory system characteristics.
* Design a multi-modal cross-attention fusion module to capture long-term audio-visual dependencies.

### 📖 Datasets

|  Dataset  | Distortion Type | Orig Seq Num | Dis Seq Num |              Resolution              | Duration |
|:---------:|:---------------:|:------------:|:-----------:|:------------------------------------:|:--------:|
| LIVE-SJTU |    Synthetic    |     14       |     336     |             1920 * 1080              |    8s    |
|  UnB-AVC  |    Synthetic    |      6       |     72      |              1280 * 720              |    8s    |
| SJTU-UAV  |     Natural     |     520      |     520     | 1280 * 720 / 960 * 720 / 1920 * 1080 |    8s    |

### ️️📃 Download

* **LIVE-SJTU Dataset:** https://live.ece.utexas.edu/research/avqa/index.html.
* **UnB-AVC Dataset:** https://www.ene.unb.br/mylene/databases.html.
* **SJTU-UAV Dataset:** https://drive.google.com/drive/folders/1158ZHQsF2vVSqHquMdbBE6gLqhY2CfCe?usp=drive_link.
* **Pretrained VoVNet-v2:** https://github.com/youngwanLEE/vovnet-detectron2.

### ️️💡 Guide

* Run **VisualSaliency/sal_model.m** in Matlab to obtain the position map of video frames.
* The complete model is contained in **demo.py**. Create a folder named **test_av_seq** and put the audio and video files into it. 
* There exist works that outperform our method on the **SJTU-UAV database**. Researchers can further improve our work based on the limitations discussed in paper.

### 🌹🌹 Acknowledgments 🌹🌹

This work is primarily developed based on the following research:

#### AVQA Baseline
Our research follows **ANNAVQA** train/test framework, the related code is available at https://github.com/charlotte9524/ANNAVQA-pytorch.
```latex
@ARTICLE{10075375,
  author={Cao, Yuqin and Min, Xiongkuo and Sun, Wei and Zhai, Guangtao},
  journal={IEEE Transactions on Image Processing}, 
  title={Attention-Guided Neural Networks for Full-Reference and No-Reference Audio-Visual Quality Assessment}, 
  year={2023},
  volume={32},
  number={},
  pages={1882-1896},
  keywords={Feature extraction;Visualization;Quality assessment;Measurement;Streaming media;Video recording;Computer architecture;Audio-visual quality assessment;attentional neural networks;multimodal fusion},
  doi={10.1109/TIP.2023.3251695}}
```

### 😊 Cite Us

If you have any questions, please contact us via email.
If you find this work useful, please cite our paper, we would be grateful.
```latex
@ARTICLE{11345165,
  author={Lin, Junhao and Cui, Yueli and Fang, Chenli and Pan, Binghong and Pan, Chencheng and Jiang, Gangyi and Zhang, Shiqing and Ma, Siwei and Tian, Qi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Multi-modal Cross-Attention Guided Network for Audio-Visual Quality Evaluation via Visual Saliency and Mel-spectrum Features}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Measurement;Feature extraction;Visualization;Videos;Quality assessment;Convolutional neural networks;Vectors;Technological innovation;Nonlinear distortion;Multimedia communication;Multi-modal cross-attention;audio-visual quality evaluation;visual saliency;Mel-spectrum feature},
  doi={10.1109/TCSVT.2026.3652641}}
```
