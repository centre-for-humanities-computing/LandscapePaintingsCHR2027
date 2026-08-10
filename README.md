This folder contains the code for the paper "Under the Canopy: A Methodological Comparison for Computational Motif Analysis in Nineteenth-Century Painting".

# Image classification and ablation study

## Data and use

This folder contains scripts to train a classifier, conduct the ablation and view the ablation maps. You can either go through the stepts yourself with the provided data for each ([train-classifier file](https://huggingface.co/datasets/chcaa/northern-landscape-painting/tree/main/train-classifier), [projection heads](https://huggingface.co/datasets/chcaa/northern-landscape-painting/tree/main/projection-heads)) or use the full parquet file with the predictions and results [provided](https://huggingface.co/datasets/chcaa/northern-landscape-painting/tree/main/ablations). For the tree classification, the full parquet with both cut-out segments, full images with bounding boxes and prediticions can be found [here](https://huggingface.co/datasets/chcaa/CHR_detection_trees/tree/main/classifier).

## Structure

```
LandscapePaintings_Detection_Classification_and_Evaluation/classification

├── data/                            # put the train-classifier data and or directly the ablation results from huggingface here 
├── heads/                           # put the projection heads from huggingface here 
├── classifiers/                     # gets created when running train-classifier, contains metrics, joblib and predictions per model
│   ablation.py                      # runs ablation on trained classifiers. See config at top for parameters
│   ablation-viewer.py               # viewer to explore the ablation maps
│   requirements.txt                 # requirements for the whole pipeline (be mindful of the correct torch version for your system)
│   train-classifier.py              # script to train classifiers for ablation
│   tree-viewer.py                   # viewer to explore the tree-based classifier

```