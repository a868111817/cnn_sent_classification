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

```
sh script/MR_download.sh
sh script/fasttext_download.sh
```

Running

```
python main.py --model CNN-rand
```

  * `CNN-rand` initializes the word embeddings randomly and learns them.
  * `CNN-static` initializes the word embeddings to word2vec and keeps the weight static.
  * `CNN-nonstatic` also initializes to word2vec, but allows them to be learned.
  * 

Result



| model          | ACC   |
| -------------- | ----- |
| CNN-rand       | 72.75 |
| CNN-static     | 83.47 |
| CNN-non-static | 83.74 |
