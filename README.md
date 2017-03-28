## Video-Summarization-with-LSTM
Providing the data and codes for evaluation for our ECCV 2016 Paper (Video Summarization with Long Short-term Memory)
# Data
Please refer to the following link as the data used in our paper: 
https://www.dropbox.com/s/717k8523ui0zaio/Data_releasing.zip?dl=0

Note that we down-sampled the original video by 2fps, so I think it would be better to provide the features and corresponding labels in this setting. 
1) file name: in the format 'Data_$Dataset$_google_p5.h5', e.g. Data_SumMe_google_p5.h5, means the frame level feature of SumMe dataset. 
2) the index of videos are stored as ‘idx’ in the file, in most cases it’s from 1 to n, where n is the number of videos in the dataset (except for Youtube dataset).
3) feature & ground-truth: the feature is indexed as ‘fea_i’ , the importance is indexed as ‘gt_1_i’ (real number, from the original dataset), and the keyframe we used is indexed as ‘gt_2_i’  (binary value transferred from the original dataset) for the i-th video in the dataset.

Original videos and annotations for each dataset are also available from the the authors' project page

* TVSum dataset [1]: https://github.com/yalesong/tvsum
* SumMe dataset [2]: https://people.ee.ethz.ch/~gyglim/vsum/#benchmark
  
* OVP and YouTube datasets [3]: https://sites.google.com/site/vsummsite/
  

# Code for evaluation
For both SumMe and TVsum datasets, you can find the code for evaluation provided by the author:
* SumMe Dataset: https://people.ee.ethz.ch/~gyglim/vsum/#benchmark
* TVSum Dataset: https://github.com/yalesong/tvsum

I also provided the evaluation code with wrappers that help adapt to the datasets above

## Reference
[1] Yale Song, Jordi Vallmitjana, Amanda Stent, and Alejandro Jaimes. "Tvsum: Summarizing web videos using titles." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5179-5187, 2015.

[2] Michael Gygli, Helmut Grabner, Hayko Riemenschneider, and Luc Van Gool. "Creating summaries from user videos." In European conference on computer vision, pp. 505-520, 2014.

[3] S. E. F. de Avila, A. P. B. Lopes, A. da Luz, and A. de Albuquerque Ara´ujo. "Vsumm: A mechanism designed to produce static video summaries and a novel evaluation method," Pattern Recognition Letters, 32(1):56–68, 2011.
