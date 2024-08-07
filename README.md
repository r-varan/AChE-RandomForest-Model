# AChE-RandomForest-Model
A Random Forest Regression Model that predicts the inhibiting capacity (IC50) of a Biomolecule to Acetylcholinesterase based off its chemical features.

This repository contains a Python script for analyzing and predicting the bioactivity of compounds targeting acetylcholinesterase using data from the ChEMBL database. The analysis includes data cleaning, feature extraction, and machine learning model training to predict bioactivity based on molecular descriptors.

## Overview

The script performs the following tasks:

Query ChEMBL Database: Retrieve data related to acetylcholinesterase.
Data Cleaning and Processing: Clean and preprocess the data.
Feature Extraction: Compute molecular descriptors using Lipinski's rules.
Normalization and pIC50 Calculation: Normalize bioactivity values and compute pIC50.
Dataset Preparation: Prepare data for machine learning.
Machine Learning Model: Train and evaluate a Random Forest Regressor model.
Prerequisites

## Dependencies

pandas
numpy
rdkit
chembl_webresource_client
scikit-learn

You also need to run a script that provides custom functions for the Predictor Tool PADEL. PADEL is a software tool used for creating a 'molecular fingerprint' along with descriptors for your biomolecule from SMILES strings (biomolecule chemical structure). This is the ultimate input that will be used for predicting the inhibition force of a biomolecule. To access the script go to: 
- https://github.com/dataprofessor/bioinformatics/raw/master/padel.zip
- https://github.com/dataprofessor/bioinformatics/raw/master/padel.sh

## Script Breakdown

1. Query ChEMBL Database
The script queries the ChEMBL database to retrieve bioactivity data related to acetylcholinesterase.

2. Data Cleaning and Processing
Cleans the data by removing rows with missing values and duplicates. It also standardizes SMILES representations.

3. Feature Extraction
Computes Lipinski molecular descriptors for each compound to evaluate drug-likeness.

4. Normalization and pIC50 Calculation
Normalizes the bioactivity values and computes the pIC50 values for model training.

5. Dataset Preparation
Combines the cleaned data with computed descriptors and saves it for model training.

6. Machine Learning Model
Trains a Random Forest Regressor to predict bioactivity based on the molecular descriptors and evaluates its performance.
