# Trash or Treasure: How to Utilize Emojis in Social Media Sentiment Classification

Reseach Project by Bale Chen from New York University Shanghai. If interested, please contact via email (bale.chen@nyu.edu).

## Data

The emoji2vec twitter sentiment analysis data is originally from the paper [**emoji2vec: Learning Emoji Representations from their Description**](https://arxiv.org/pdf/1609.08359.pdf) by Ben Eisner, Tim Rocktäschel, Isabelle Augenstein, Matko Bošnjak, and Sebastian Riedel. Their repository can be accessed [here](https://github.com/uclnlp/emoji2vec).

The SemEval-2015 data is the [**SemEval-2015 Task 11: Sentiment Analysis of Figurative Language in Twitter**](https://aclanthology.org/S15-2080/) by Aniruddha Ghosh, Guofu Li, Tony Veale, Paolo Rosso, Ekaterina Shutova, John Barnden, and Antonio Reyes. Please refer to their original paper for more information. Note that many tweets have perished or been deleted. Thanks to the SemEval-15 organizers who created a Twitter bot and posted those tweets again to avoid perishing.

## Report

As a form of dissemination of knowledge, the research work and findings are presented in from of an academic report. Please refer to the `Report.pdf` file to see the complete paper. If you want to understand the main idea of this project, I suggest to read it first!

## Scripts

In the `scripts` folder are the multiple scripts that I wrote for different functionalities. The `train.py` is essentially the main idea of the whole piple, though it's not the final version I ran to generate results. Yet, it's adequate for understanding the whole process. `train_robust.py` contains the code I ran for the final experienment on each model-method pair. `train_meta_robust.py` is a complementary script for `train_robust.py` that trains the models specifically with the meta-feature method (refer to Section 3.3.2 in the report). `contraction_map.py` and `utils.py` have some handy functions for processing data. Note that all scripts are executed on NYU HPC Greene Cluster, so personal computers might not be able afford such heavy computation.

## Visuals

In the `visuals` folder I provided some visualization of results and data. 

## Dependencies

- python 3.10.4
- emoji 1.7.0
- numpy 1.22.3
- pandas 1.4.2
- pytorch 1.12.0
- openpyxl 3.0.9
- regex 2022.3.15
- gensim 4.1.2
- nltk 3.7
- scikit-learn 1.1.1
- scipy 1.7.3
- tqdm 4.64.0

## Acknowledgement

I  would like to extend my warmest gratitude to my research supervisor and mentor Professor Mathieu Laurière. He provides me with scholarly and resourceful advice, guides me through this summer research. It is my great honor and pleasure to finish this study with him and receive his email greeting on my birthday. This work was also supported in part through the NYU IT High Performance Computing resources, services, and staff expertise. Besides, I genuinely appreciate NYU and NYUSH for offering me the DURF research opportunity. Thanks to all my friends and family who helped me throughout this summer. The research would not have been possible without any of you ❤️.
