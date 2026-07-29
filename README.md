# Sentiment-Analysis-with-Recurrent-Neural-Networks
Comparing RNN, LSTM, and BRNN models for IMDb movie review sentiment analysis.

## Overview
This project classifies IMDb movie reviews as positive or negative using deep learning. The code implements and compares three different sequence models: a Vanilla Recurrent Neural Network (RNN), a Long Short-Term Memory network (LSTM), and a Bidirectional LSTM (BRNN). The goal is to evaluate how different memory mechanisms handle sequence context and solve the vanishing gradient problem.

## Prerequisites
You need a Python environment to run this Jupyter Notebook. The project relies on the following packages:
* Python 3.9+
* PyTorch 2.x
* HuggingFace datasets
* numpy
* matplotlib

## Usage
1. Clone this repository to your local machine.
2. Open the `rnn_sentiment.ipynb` notebook in Jupyter or your preferred IDE.
3. Run all cells in order from top to bottom.
4. The notebook will automatically download the IMDb dataset, train all three models, and generate comparison graphs.
5. Scroll to the "Live Prediction" section to test the trained Bidirectional LSTM with your own custom text inputs.

## Key Findings
* **Performance:** The Bidirectional LSTM achieved the highest test accuracy at 86.04%. The standard LSTM followed closely at 85.51%. The Vanilla RNN struggled significantly, peaking at only 62.71% accuracy.
* **Training Efficiency:** The LSTM and BRNN converged much faster per epoch than the Vanilla RNN. The internal gating mechanisms successfully retained long-term memory and prevented the vanishing gradient problem.
* **Contextual Nuance:** Error analysis showed that sequence models still struggle with "bait-and-switch" reviews and misleading comparative statements. The sheer volume of positive words in a negative review can overpower the mathematical weight of the true sentiment.

## Tech Stack
* Python
* PyTorch
* HuggingFace Datasets
* Matplotlib
