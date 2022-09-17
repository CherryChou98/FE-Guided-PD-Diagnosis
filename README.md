# FE-Guided-PD-Diagnosis


This work entitled "Facial Expression Guided Diagnosis of Parkinson’s
Disease via High-quality Data Augmentation" has been submitted to IEEE Transactions on Multimedia (TMM). 

In this package, we implement our PD diagnosis model using Pytorch.

-------------------------------------------------------------------------


To train the diagnosis model and get the evaluation results, you can run the following command:
```bash
$ python train.py --batch_size 32 --optimizer adam --lr 1e-3 --weight_decay 1e-5 --epochs 70
```
To get the quality scores which are results of facial image quality assessment (FIQA), you can use ```calc_quality_scores.py```. 



The software is free for academic use, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.