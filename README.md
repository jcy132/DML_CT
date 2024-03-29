## Official Pytorch implementation of "Patch-wise Deep Metric Learning for Unsupervised Low-Dose CT Denoising" (MICCAI 2022)
[Chanyong Jung](https://sites.google.com/view/jcy132), Joonhyung Lee, Sunkyoung You, [Jong Chul Ye](https://bispl.weebly.com/professor.html)

Link: [https://arxiv.org/abs/2207.02377](https://arxiv.org/abs/2207.02377)


<p align="center">
<img src="https://user-images.githubusercontent.com/52989204/178413091-4cf84127-bb45-4db0-96bf-5521f46d15d0.jpg" width="800"/>
</p> 

### Implementation
* We provide the source code for AAPM dataset. \
(2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge dataset)\
We randomly select 3112 images for train set, 421 images for validation set. \
421 images are used for test set. 

* Refer the following code to obtain the model:
```
python main.py --prj_name [folder-name] --log_name [log-file-name] \
--dataset_name AAPM --data_root [path-to-data] --gpu_ids 0
```


### Acknowledgement
Our source code is based on the official implementation of [CUT](https://github.com/taesungp/contrastive-unpaired-translation). 
