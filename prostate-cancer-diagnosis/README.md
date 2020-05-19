# Udacity Machine Learning Engineer Nanodegree
## Capstone Proposal 
Do HUYNH 
[Github](http://www.github.com/huynhdoo/) | [Linkedin](http://www.linkedin.com/in/huynhdoo/)
May 19th, 2020

---
## Biopsy prostate cancer diagnosis: a digital pathology web app
---

### Domain Background 

---

According to the world health organization, cancer is the second leading cause of death in the world with 9,6 millions death in 2018[^cancer]. Moreover according to the world cancer research fund, prostate cancer is the second most common cancer in men with 1.3 millions new case in 2018[^prostate].

[^cancer]: World health organization - https://www.who.int/news-room/fact-sheets/detail/cancer
[^prostate]: World cancer research fund - https://www.wcrf.org/dietandcancer/cancer-trends/prostate-cancer-statistics

In the healthcare system, the role of a histopathologist (from greek "histos" = tissue) is to search and detect cancerous cells from a tissue sample (biopsy). It is a sensible responsability depending on the experience of the specialist and the technicals conditions. With the high definition digitalization of microscopy images (call Whole-Slide Image WSI) and the progress of AI image analysis during the last decade, digital pathology has made great improvement in pathology detection assisting the medicals professionnals in their diagnosis and prognosis assumptions [^digital_pathology].
[^digital_pathology]: Artificial intelligence in digital pathology — new tools for diagnosis and precision oncology - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6880861/

![AI and healthcare support](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6880861/bin/nihms-1059940-f0004.jpg)
_Source: [Nat Rev Clin Oncol. 2019 Nov; 16(11): 703–715.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6880861/)_

One of the big challenge in healthcare is to help the physicians diagnose any disease more fastly and more acurately for a best health treatment. Since the advent of deep learning, many AI algorithm using neural network have been test and deploy on medical image analysis[^medical_analysis]. As a machine learning engineer, i am personnaly concerned with any human and social positive impact of technology. The application of AI in healthcare is clearly one of them.

[^medical_analysis]: The medical image analysis journal publish monthly a selection of the last research paper on deep learning techniques and algorithms apply to medical images (MRI, CT, echography, etc.) - https://www.sciencedirect.com/journal/medical-image-analysis/vol/62/suppl/C

---

### Problem Statement

---

The Gleason's score is the world common indicator of cancerous state infere from a prostate biopsy image (from 1 to 5). This value is an indicator of the cancer development (from 1=benign to 5=abnormal tissue).

[^histopathology]: Clinical histopathology of the biopsy cores : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1769959/

![Gleason's pattern](https://www.prostateconditions.org/images/about/murtagh7e_c114_f04-2.png)

The sum of gleason score of the most present pattern and the second present pattern composed the ISUP grade (from the International Society of Urological Pathology) which globally indicate the state of the cancerous cells (from 1=localized to 5=spreaded)

![Gleason grading process](https://storage.googleapis.com/kaggle-media/competitions/PANDA/Screen%20Shot%202020-04-08%20at%202.03.53%20PM.png)
_Source: [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview/description)_

Today, the 'gold standard' in prostate cancer detection is the human visual diagnosis[^histopathology] which can lead to important variation in the cancer diagnosis depending on the pathologist experience. The aim of the project is to help pathologist predict the gleason score and then the ISUP grade from any prostate biopsy image.

---

### Datasets and Inputs

---

We will work on the dataset provide by the Prostate cANcer graDe Assessment Challenge host by Kaggle from April 21, 2020 to July 22, 2020: [PANDA challenge dataset](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data). This dataset is composed of 11,000 prostate biopsies whole-slide images scored by uro-pathologists from 2 differents medical institutions (Karolinska Institute and Radboud University Medical).

Because of the Kaggle format competition, we do not have access to the final test set ground truth. However, for the purpose of this project, we will reserve a random and stratified part of the training dataset, which is quite large, for the final evaluation step.

---

### Solution Statement 

---

We will create an AI web application using deep learning model that can read a biopsy image and return a possible diagnosis of Gleason score/ISUP grade to the end-user. For confidentiality reason, the image will be read on the fly and stay on the user side.

---

### Benchmark Model 

---

Many AI models are already been proposed by the scientific community. Here is a selection:
|Model|Sample|
|---|---|
|Handcraft ML|[Automated image analysis system for detecting boundaries of live prostate cancer cells](https://www.ncbi.nlm.nih.gov/pubmed/9551604)|
|CNN|[Development and validation of a deep learning algorithm for improving Gleason scoring of prostate cancer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6555810/)|
|GAN|[ProstateGAN: Mitigating Data Bias via Prostate Diffusion Imaging Synthesis with Generative Adversarial Networks](https://arxiv.org/abs/1811.05817)|
|Last research from the PANDA challenge organizers|[Pathologist-Level Grading of Prostate Biopsies with Artificial Intelligence](https://arxiv.org/abs/1907.01368) & [Automated Gleason Grading of Prostate Biopsies using Deep Learning](https://arxiv.org/abs/1907.07980)|

Obviously, each time it is possible, we will take inspiration from this previous models.

---

### Evaluation Metrics 

---

Following the Prostate cANcer graDe Assessment Challenge organizers, the evaluation of the submission will be calculate with **the quadratic weighted kappa**[^evaluation]:

![quadratic weighted kappa](https://render.githubusercontent.com/render/math?math=\kappa=1-\frac{\sum{i,j}w{ij}O{ij}}{\sum{i,j}w{ij}E{ij}})

with:
- O: an N x N histogram matrix such that Oij corresponds to the number of isup_grades i (actual) that received a predicted value j
- W: an N x N matrix of weights Wij calculated on the difference between actual i and predicted values j

![matrix of weights](https://render.githubusercontent.com/render/math?math=w_{i,j}=\frac{\left(i-j\right)^2}{\left(N-1\right)^2})

- E: an NxN histogram matrix of expected outcomes, E, calculated assuming that there is no correlation between values. This is calculated as the outer product between the actual histogram vector of outcomes and the predicted histogram vector, normalized such that E and O have the same sum.

A usefull detail implementation with python can be found here: [Quadratic Kappa Metric explained in 5 simple steps](https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps)

[^evaluation]: PANDA challenge evaluation metrics - https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview/evaluation

---

### Project Design

---

We will apply a deep learning method to predict the ISUP grade of a given prostate biopsy image. The project design will follow differents steps:
1. **Exploratory Data Analysis** 
    _Objective : check the distributions of the whole dataset according to the Gleason scores, pixel-level assessments and datas normalization_
    - Load images and labels from dataset
    - Explore imaging properties (size, pixel intensities)

2. **Build and train model**
    _Objective: define and train a deep learning model that can predict the Gleason score from a biopsy image_
    - Split dataset into training / validation / test set with score balance
    - Image pre-processing (noise, normalization, augmentation, resize)
    - Build a dataframe images with ground truth
    - Implement the evaluation metrics
    - Select and fine-tune a CNN according to bests benchmark model
    - Train and evaluate the model

3. **Predict Gleason score**: 
    _Objective: evaluate the model on a test dataset_
    - Load test dataset images
    - Evaluate accuracy score of the model
    - Refine-tune the model

4. **Deployment**: 
    _Objective: create and deploy a prostate cancer diagnose web application_
    - Train and deploy the model with AWS SageMaker endpoint
    - Create a lambda function to access endpoint
    - Create and deploy a web app on AWS
    - Clean any uneccessary ressources (endpoint, model, lambda, files, notebook, etc.)
