# cnn_sent_classification
1-D CNN for sentence classification TEST

---

## Quick start

Pipenv 

```
pipenv install
pipenv shell
```

Download dataset : **Movie Review(MR)**
and word vector :**fastText**

```
sh script/MR_download.sh
sh script/fasttext_download.sh
```
Model architecture

![Screenshot from 2022-04-08 14-01-52](https://user-images.githubusercontent.com/49869328/162374222-6bb5a7be-4509-4767-b17c-294b20a8417f.png)


Running

```
python main.py --model CNN-rand
```

  * `CNN-rand` initializes the word embeddings randomly and learns them.
  * `CNN-static` initializes the word embeddings to word2vec and keeps the weight static.
  * `CNN-nonstatic` also initializes to word2vec, but allows them to be learned.

Result



| model          | ACC   |
| -------------- | ----- |
| CNN-rand       | 72.75 |
| CNN-static     | 83.47 |
| CNN-non-static | 83.74 |

refernce:

[harvardnlp/sent-conv-torch](https://github.com/harvardnlp/sent-conv-torch)

[A Complete Guide to CNN for Sentence Classification with PyTorch](https://chriskhanhtran.github.io/posts/cnn-sentence-classification/)

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
