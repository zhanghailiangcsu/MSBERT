# MSBERT
Mass Spectrometer Embedder for Improving Mass Spectrometry Similarity Score Using BERT
# 1. Introduction
In this study, MSBERT based on mask learning and contrastive learning was proposed to get a reasonable embedding representation of MS/MS. 
MSBERT used the transformer encoder as the backbone and take advantage of the randomness of the mask to construct positive samples for contrastive learning. 
MSBERT was trained and tested on GNPS dataset. MSBERT had a stronger ability in library matching, with top 1, top5, and top 10 were 0.7871, 0.8950, and 0.9080 on Orbitrap test dataset. 
The results are significantly better than Spec2Vec and cosine similarity. 
The rationality of embedding was demonstrated by reducing the dimensionality of the embedding vectors, calculating structural similarity, and spectra clustering. 

# 2. Depends
[Anaconda](https://www.anaconda.com) for Python 3.9  
[Pytorch](https://pytorch.org/) 1.12  
# 3. Install
1.Install anaconda  
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
The MSBERT is public at [homepage](https://github.com/zhanghailiangcsu), every user can download and use it.
You can download the trained model on Github.
Then refer to the example to use the model for prediction, and directly obtain the embedding vectors from MS/MS.
Alternatively, you can retrain MSBEERT on your own data refer to example.
# 5.Contact
Hailiang Zhang  
E-mail 2352434994@qq.com  
