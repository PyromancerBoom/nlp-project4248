# Sarcasm Subtypes Classification Project

## Overview
This project aims to annotate a dataset of approximately 13,000 sarcastic headlines with specific sarcasm subtypes. We use a three-stage annotation process with Llama 3 (8B) to create high-quality labels that can be used for training a supervised classifier.

## Sarcasm Categories
We classify headlines into five distinct categories of sarcasm:

1. **CONTEXTUAL CONTRADICTION**: Sarcasm that presents a situation that contradicts common knowledge, expectations, or reality.
   - Example: "Study finds regular exercise better than sitting on couch all day"
   - Example: "Local man shocked to discover fast food isn't healthy"

2. **MOCK ENTHUSIASM**: Sarcasm that expresses false excitement or positive sentiment about clearly negative situations.
   - Example: "Can't wait to spend my weekend fixing the plumbing!"
   - Example: "Awesome! 3-hour flight delay with screaming toddler nearby"

3. **STYLISTIC IRONY**: Sarcasm conveyed through specific writing techniques like quotation marks, exaggeration, or structural patterns.
   - Example: "New diet book promises to help readers lose 'up to' zero pounds"
   - Example: "Area man writes absolutely perfect, flawless email with zeroooo mistakes"

4. **INSTITUTIONAL CRITIQUE**: Sarcasm that mocks institutions, systems, or authority figures by highlighting absurd or hypocritical behaviors.
   - Example: "Congress introduces bill to help struggling Americans by losing their tax returns"
   - Example: "Health insurer celebrates record profits by denying more claims"

5. **BEHAVIORAL OBSERVATION**: Sarcasm that highlights common human behaviors or social patterns in an exaggerated or formalized way.
   - Example: "Dad enters 15th consecutive hour of watching 'just five minutes' of golf"
   - Example: "Man completely transforms personality while talking to waiter"

## Annotation Process

Our annotation pipeline follows a three-stage process to ensure high-quality labels:

### Stage 1: Initial Classification
**Script**: `annotate_dataset.py`

This script performs the initial classification of all headlines. For each headline, the LLM is asked to:
1. Think step by step about what makes the headline sarcastic
2. Determine which of the five categories best describes the PRIMARY type of sarcasm
3. Return only the category name

The results are saved to `sarcasm_classifications.csv`.

### Stage 2: Verification
**Script**: `verify_annotations.py`

This script verifies the initial classifications from Stage 1. For each headline:
1. The LLM is presented with both the headline and its initial classification
2. It's asked to consider all five categories with their definitions
3. It either confirms the current classification or suggests a more appropriate category

The results are saved to `new_sarcasm_classifications.csv`.

### Stage 3: Conflict Resolution
**Script**: `finalize_annotations.py`

This script resolves any disagreements between Stage 1 and Stage 2:
1. It identifies all headlines where the first and second stage classifications differ
2. For each conflicting case, it presents both classifications and their definitions
3. The LLM makes a final decision on which classification better captures the primary form of sarcasm

The final results are saved to `final_sarcasm_classifications.csv`.

## Technical Implementation

- **Model**: Llama 3 (8B) via Ollama
- **Independence**: Each stage uses a fresh chat context to avoid being influenced by previous decisions
- **Consistency**: The same system prompt with category definitions is used across all stages
- **Error Handling**: The process includes automatic retries and periodic saving of results
- **Preprocessing**: The scripts handle normalization of category names for consistency

## Usage Instructions

1. Ensure you have Ollama installed and the Llama 3 (8B) model available
2. Run the scripts in the following order:
   ```
   python annotate_dataset.py     # Initial classification
   python verify_annotations.py   # Verification
   python finalize_annotations.py # Conflict resolution
   ```
3. Monitor the generated CSV files at each stage for quality control, as there may be some unexpected responses due to profanity or hate speech in the headline
4. You can use the visualisation in `label_exploration.ipynb` to check for unexpected labels
5. The final dataset in `final_sarcasm_classifications.csv` contains the highest quality annotations

## Notes

- The subjective nature of sarcasm classification means some headlines may legitimately fit multiple categories
- Distribution shifts between stages are expected and represent the inherent ambiguity in sarcasm classification
- When using this data for training models, consider weighing examples by confidence or using a multi-label approach for ambiguous cases
- These annotations are based on linguistic patterns rather than author intent, which isn't available

## Technical Details

### Hardware Configuration
- CPU: AMD Ryzen 7 7700X 8-Core Processor
- RAM: 32.0 GB
- GPU: NVIDIA GeForce RTX 4070 SUPER

### Runtime Information
- Initial Classification (annotate_dataset.py): ~50 minutes for 13,634 headlines
- Verification (verify_annotations.py): ~50 minutes
- Conflict Resolution (finalize_annotations.py): ~50 minutes

### Software Environment
- Python version: 3.11.9
- Ollama version: 0.5.13
- OS: Windows

## References

Our categorization scheme is inspired by academic literature on sarcasm detection, particularly the types of sarcasm identified in:

- Chaudhari, P., & Chandankhede, C. (2017). [Literature survey of sarcasm detection](https://ieeexplore.ieee.org/abstract/document/8300120). In 2017 International Conference on Wireless Communications, Signal Processing and Networking (WiSPNET), IEEE.