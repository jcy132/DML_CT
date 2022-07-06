## Official Pytorch implementation of "Patch-wise Deep Metric Learning for Unsupervised Low-Dose CT Denoising" (MICCAI 2022)
[Chanyong Jung](https://sites.google.com/view/jcy132), Joonhyung Lee, Sunkyoung You, [Jong Chul Ye](https://bispl.weebly.com/professor.html)

<p align="center">
<img src="https://user-images.githubusercontent.com/52989204/177431169-816f061c-49b5-4632-a532-99054e7cab29.jpg" width="800"/>
</p> 



### Implementation
* We provide the source code for AAPM dataset. \
(2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge dataset)

* Refer the following code to obtain the model:
```
python main.py --prj_name [folder-name] --log_name [log-file-name] \
--dataset_name AAPM --data_root [path-to-data] --gpu_ids 0,1
```


### Acknowledgement
Our source code is based on the official implementation of [CUT](https://github.com/taesungp/contrastive-unpaired-translation). 
