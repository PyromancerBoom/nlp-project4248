# Semi-Supervised Labeling for Sarcasm Classification

This project demonstrates an effective semi-supervised learning approach for classifying sarcastic news headlines into distinct subtypes. Starting with a small set of manually labeled examples, this approach leverages the power of transformer-based language models to expand the labeled dataset significantly.

## Dataset and Labels

The dataset consists of sarcastic news headlines categorized into three distinct types:

1. **Logical Subversion (0)**
   * Headlines that deliberately undermine common sense or expected logic
   * Creates humor through absurdity, contradiction, or impossible scenarios
   * Example: "Man Dies After Switching to Diet Soda"

2. **Disproportionate Framing (1)**
   * Headlines that create humor through inappropriate scale or emphasis
   * Includes both tonal mismatches (overly formal/enthusiastic language) and exaggerated social patterns
   * Example: "BREAKING: Local Man Heroically Remembers to Buy Milk"

3. **Targeted Mockery (2)**
   * Headlines that deliberately ridicule specific institutions, authorities, or groups
   * Clear satirical intent aimed at particular targets
   * Example: "Congress Passes Bill to Fund Search for Their Own Spines"

## Methodology

### Initial Labeling
- Manually labeled 600 samples (200 per class) to create a balanced seed dataset
- Ensured high-quality annotations with clear distinctions between classes

### Semi-Supervised Learning Process
The process follows these steps, as implemented in `semi_supervised.ipynb`:

1. **Model Training**: Train a BERT model on the current labeled dataset
2. **Pseudo-Labeling**: Use the trained model to predict labels for the unlabeled data
3. **Confidence Filtering**: Select only high-confidence predictions (threshold = 0.8)
4. **Dataset Expansion**: Add these new pseudo-labeled examples to the training set
5. **Iteration**: Repeat the process for multiple iterations

### Results
Starting with just 600 manually labeled examples, the semi-supervised approach produced:
- Final labeled dataset of 13,633 examples
- Iterative improvement in model performance:
  - Iteration 1: F1 score = 0.653
  - Iteration 2: F1 score = 0.946
  - Iteration 3: F1 score = 0.888
  - Iteration 4: F1 score = 0.902
  - Iteration 5: F1 score = 0.898

## Model Evaluation

Several classification models were tested on the expanded dataset:

1. **Baseline: TF-IDF + Random Forest**
   - Initial accuracy: 64.9%
   - Tuned accuracy: 68.9%
   - F1 score: 0.688

2. **BERT Fine-Tuning**
   - Accuracy: 90.3%
   - F1 score: 0.902
   - Best performing model overall

3. **RoBERTa Fine-Tuning**
   - Accuracy: 84.0%
   - F1 score: 0.839

## Files and Structure

- `semi_supervised.ipynb`: Implementation of the semi-supervised learning loop
- `classifier_experiments.ipynb`: Various classification models tested on the expanded dataset
- `results_final.csv`: The final expanded dataset with labels
- `semi_supervised_annotation/`: Directory containing the final train/validation/test splits