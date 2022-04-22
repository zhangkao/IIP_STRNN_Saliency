# IIP_STRNN_Video_Saliency

It is a re-implementation code for the following paper: 

* Kao Zhang, Zhenzhong Chen, Shan Liu. A Spatial-Temporal Recurrent Neural Network for Video Saliency Prediction. IEEE Transactions on Image Processing (TIP), vol. 30, pp. 572-587, 2021. <br />
Github: https://github.com/zhangkao/IIP_STRNN_Saliency

Related Project

* Kao Zhang, Zhenzhong Chen. Video Saliency Prediction Based on Spatial-Temporal Two-Stream Network. IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), vol. 29, no. 12, pp. 3544-3557, 2019. <br />
Github: https://github.com/zhangkao/IIP_TwoS_Saliency


## Installation 
### Environment:
The code was developed using Python 3.6+ & pytorch 1.10+ & CUDA 10.0+. There may be a problem related to software versions.
* Windows10/11 or Ubuntu20.04
* Anaconda latest, Python 
* CUDA, CUDNN, and CUPY

### Python requirements
You can try to create a new environment in anaconda, as follows

    *The implementation environment of our experiment (GTX1080 and TITAN Xp) 

        conda create -n strnn python=3.6
        conda activate strnn
        conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 cudnn
        conda install cupy==7.8.0 
        pip install opencv-python torchsummary hdf5storage h5py scipy scikit-image matplotlib
		

    *For GEFORCE RTX 30 series, such as RTX3060, 3080, etc.
        
        conda create -n strnn python=3.8
        conda activate strnn
        conda install pytorch torchvision torchaudio cudatoolkit=11.3 cudnn
        conda install cupy
        pip install opencv-python torchsummary hdf5storage h5py scipy scikit-image matplotlib

        *Please change the parameters in "correlation.py" for this environment(cudatoolkit >= 11.0)
        "@cupy.util.memoize(for_each_device=True)" --> "@cupy.memoize(for_each_device=True)"

### Pre-trained models
Download the pre-trained models and put the pre-trained model into the "weights" file.

* **PWC-model**
 [OneDrive](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EcXC4vlGlvBFmEb5TWDiJj4BKg9iLdKe6bEOPnTWikkp_w?e=a1bFnN);
 [Google Drive](https://drive.google.com/file/d/1-oi-rqrkLdvzVUaEtLn0c86XmpmJ9Gw2)
 (36M)
* **STRNN-model** 
[OneDrive](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/ERc5CZfApppJk9lwcskAvs8BsTCD1YPDqH9JLiju5D9R5Q?e=TlK8qe);
[Google Drive](https://drive.google.com/file/d/1scBAo3UBWPe0q_NH2H1Eot_a12LE6ix5)
(361M)
         

### Train and Test

**The parameters**

* Please change the working directory: "dataDir" to your path in the "Demo_Test.py" and "Demo_Train_DIEM.py" files, like:

        dataDir = 'E:/DataSet'
        
* More parameters are in the "train" and "test" functions.
* Run the demo "Demo_Test.py" and "Demo_Train_DIEM.py" to test or train the model.

**The full training process:**

* Our model is trained on SALICON and part of the DIEM dataset and then tested on DIEM20, CITIUS-R, LEDOV41, SFU and DHF1K benchmark. In the SF-Net module, we retrain the St-Net on SALICON dataset and fine-tune the PWC-Net of the OF-Net on the training set of DIEM dataset. Then, we train the whole network on the training set of DIEM dataset and fix the parameters of the trained PWC-Net.


**The training and testing datasets:**

* **Training dataset**: 
[SALICON(2015)](http://salicon.net/), and 
[DIEM](https://thediemproject.wordpress.com/)
* **Testing dataset**: 
[CITIUS](https://wiki.citius.usc.es/aws-d:citius_video_database), 
[LEDOV](https://github.com/remega/LEDOV-eye-tracking-database), 
[SFU](http://www.sfu.ca/~ibajic/), 
[DHF1K](https://github.com/wenguanwang/DHF1K/)

**The training and test data examples:**
* **Training data example**: 
[DIEM](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EfPmbUU20fxPhQ8OMFbjbhABrtkgmnUhkRnCV-9b0TlXWQ?e=8uqlSM) (364M)
* **Testing data example**:
[DIEM20](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EQ7zlEj2sDVAryCN5olLKoIBEpG4wcpAuxNW-KrQEe6uIA?e=ewHiGz) (132M)




### Output
And it is easy to change the output format in our code.
* The results of video task is saved by ".mat"(uint8) formats.
* You can get the color visualization results based on the "Visualization Tools".
* You can evaluate the performance based on the "EvalScores Tools".


**Results**: [ALL](https://whueducn-my.sharepoint.com/:f:/g/personal/zhangkao_whu_edu_cn/Es2k8IqwSOBMs0jpKxDG8V4Bx49k3IP_r_B6ceRqgU16FQ?e=lzmwT3) (6.2G):
[DIEM20](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EeLxb5aqHg9DobSBDi2jizYBNw02Y-W-8eaUIPIkIvCcwg?e=7aW359) (261M),
[CITIUS-R](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/Eaak96t4PH1KvGWRDHmwft8By9yQTL-TgUt1DyZ7GuJP4w?e=O11DeY) (54M), 
[LEDOV41](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EUcrQQuut2FIuLDwnHv4K9MB0Ut-_NFWlHVaayCoeiKnDA?e=cRtyZv) (839M), 
[SFU](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EchdaLI1cOlMu83tDZ5tHDIBtCUmwAIP0uSmwntffcncPA?e=USeWcC) (39M)




**Results for DHF1K**: 

We use the first 300 frames of each video from the DHF1K training set to retrain the model and generate the new results.

* **strnn_res_dhf1k_test** : 
[OneDrive](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EWb98WMO7bZGt-bq9AOnkM4BEWsC7_cm4gEuYDCrJnk22Q?e=J7havj); 
[Google Drive](https://drive.google.com/file/d/1mfV5WXkPECLDfMvBoRoaCV6GYluyBzmW)
(3.94G)<br />

* **strnn_res_dhf1k_val** : 
[OneDrive](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/ESD48TTUxHhDkzF-XxvgDgcBfQ5-IbZk9RRdHWOFUVE6MA?e=gGghd6) (1.09G)<br />



## Paper & Citation

If you use the STRNN video saliency model, please cite the following paper: 
```
@article{zhang2020spatial,
  title={A spatial-temporal recurrent neural network for video saliency prediction},
  author={Zhang, Kao and Chen, Zhenzhong and Liu, Shan},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={572--587},
  year={2021}
}
```

## Contact
Kao ZHANG  <br />
Laboratory of Intelligent Information Processing (LabIIP)  <br />
Wuhan University, Wuhan, China.  <br />
Email: zhangkao@whu.edu.cn  <br />

Zhenzhong CHEN (Professor and Director) <br />
Laboratory of Intelligent Information Processing (LabIIP)  <br />
Wuhan University, Wuhan, China.  <br />
Email: zzchen@whu.edu.cn  <br />
Web: http://iip.whu.edu.cn/~zzchen/  <br />