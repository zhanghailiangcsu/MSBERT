# MSBERT
Improve the accuracy of database search by using BERT to embed MS/MS reasonably 
# 1. Introduction

# 2. Depends
[Anaconda](https://www.anaconda.com) for Python 3.9  
[Pytorch](https://pytorch.org/) 1.12  
# 3. Install
Install anaconda  
2.Install [Git](https://git-scm.com/downloads)  
3.Open commond line, create environment and enter with the following commands.   
```
conda create -n MSBERT python=3.9  
conda activate MSBERT
```
4.Clone the repository and enter.  
```
git clone https://github.com/zhanghailiangcsu/MSBERT.git
```
5.Install dependency with the following commands.
```
pip install -r requirements.txt
```
6.Install  Pytorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
# 4. Usage
The WeightFormer is public at [homepage](https://github.com/zhanghailiangcsu), every user can download and use it.
You can download the trained model on Github.
Then refer to the example to use the model for prediction, and directly obtain the embedding vectors from MS/MS.
Alternatively, you can retrain MSBEERT on your own data refer to example.
# 5.Contact
Hailiang Zhang  
E-mail 2352434994@qq.com  
