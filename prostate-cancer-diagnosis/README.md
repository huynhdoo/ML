# Udacity Machine Learning Engineer Nanodegree

## Capstone Project: automating prostate cancer diagnosis on whole-slide image biopsy using patches extraction and TensorFlow

---

### Project overview

This project propose a solution that use deep learning techniques to automatically evaluate the ISUP grade from a whole-slide image biopsy and in fine help pathologist make accurate and faster diagnosis of prostate cancer tissue.

*keywords : Healthcare, digital pathology, prostate cancer, Google Colab, TensorFlow, EfficientNet, TPU, Udacity*

---

### Technology stack

- Python
- Keras
- TensorFlow
- EfficientNet
- Google Cloud Storage
- TPU
- Openslide

---

### Main documents

- **[Capstone proposal](./proposal)**
- **[Capstone report](./report)**
- **Notebooks:**
    - [PANDA 1 - Extract patches from WSI](./PANDA%201%20-%20Extract%20patches%20from%20WSI.ipynb)
    - [PANDA 2 - Export patches to TFRecords](./PANDA%202%20-%20Export%20patches%20to%20TFRecords.ipynb)
    - [PANDA 3 - CNN training on TPU](./PANDA%203%20-%20CNN%20training%20on%20TPU.ipynb)
    - [PANDA 4 - Kaggle model inference](./PANDA%204%20-%20Kaggle%20model%20inference.ipynb)
 
--- 
 
### Usage

The notebook are all autonomous and will install any necessary dependancies. Notebooks 1 to 3 should be run on [Google colab](https://colab.research.google.com/). Notebook 4 is runnable as [Kaggle kernel](https://www.kaggle.com/). 

You will also need a [Google cloud storage bucket](https://cloud.google.com/storage) to store generated patches. For convenience, already generated patches are publicly available with read-only access.

They are also directly accessibles following these read-only links:
- [PANDA 1 - Extract patches from WSI (colab notebook)](https://colab.research.google.com/drive/1LbvovE3QRAqwhEnfTnUKqGr11Xa8GzYH?usp=sharing)
- [PANDA 2 - Export patches to TFRecords (colab notebook)](https://colab.research.google.com/drive/11o3LGaieiTjPq1L2Tg9y-0qFGUji42ob?usp=sharing)
- [PANDA 3 - CNN training on TPU (colab notebook)](https://colab.research.google.com/drive/1I0tCDXVKoR6-ifBl0kJAcS4D8WYFg96J?usp=sharing)
- [PANDA 4 - Kaggle model inference (kaggle kernel)](https://www.kaggle.com/huynhdoo/panda-keras-model-inference)

---

### License

Feel free to execute, copy, reuse and change everything.
