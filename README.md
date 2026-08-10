This folder contains the code for the paper "Under the Canopy: A Methodological Comparison for Computational Motif Analysis in Nineteenth-Century Painting".

## Structure

Image classification and ablation study:

```
LandscapePaintings_Detection_Classification_and_Evaluation/classification

├── data/                            # put the train-classifier data and or directly the ablation results from huggingface here 
├── heads/                           # put the projection heads from huggingface here 
├── classifiers/                     # gets created when running train-classifier, contains metrics, joblib and predictions per model
│   ablation.py                      # runs ablation on trained classifiers. See config at top for parameters
│   ablation-viewer.py               # viewer to explore the ablation maps
│   train-classifier.py
│   requirements.txt                     # requirements for the whole pipeline (be mindful of the correct torch version for your system)

```

## Data

Data to train the classifier: https://huggingface.co/datasets/chcaa/northern-landscape-painting/tree/main/train-classifier
Projection heads: https://huggingface.co/datasets/chcaa/northern-landscape-painting/tree/main/projection-heads
Tree-based classifier training data: 
Ablation results: https://huggingface.co/datasets/chcaa/northern-landscape-painting/tree/main/ablations