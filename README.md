# Fine-tuning LLaMA for Spam Detection 🚀

## Large Language Model adaptation for NLP text classification

This project explores the effectiveness of fine-tuning a Large Language Model (LLaMA)
for spam message classification.

The main objective was to compare a domain-adapted LLM approach against traditional
machine learning classifiers and evaluate whether fine-tuning improves classification
quality, especially on unseen data.

The project includes:

- exploratory data analysis
- dataset quality evaluation
- synthetic dataset generation using LLM APIs
- LLaMA fine-tuning using QLoRA
- benchmarking against classical ML algorithms
- generalization experiments


# 🎯 Research Objective

Spam detection is a common Natural Language Processing (NLP) problem.

Traditional approaches usually rely on:

- TF-IDF representations
- manually selected features
- statistical machine learning models

This project investigates whether Large Language Models can provide better
generalization capabilities after domain adaptation.

Main research question:

> Can a fine-tuned LLaMA model detect spam messages more effectively than
> classical machine learning classifiers, especially on previously unseen data?


# 📈 Project Workflow

The complete pipeline:
```
Dataset Analysis
    |
    v
Dataset Quality Evaluation
    |
    v
Synthetic Data Generation
    |
    v
Dataset Preparation
    |
    v
LLaMA Fine-tuning (QLoRA)
    |
    v
Model Evaluation
    |
    v
Comparison with Classical ML Models
    |
    v
Generalization Testing
```

# 📊 Dataset Analysis

Before training, publicly available spam datasets from Kaggle were analyzed.

The exploratory analysis focused on:

- dataset quality
- class distribution
- duplicated samples
- data consistency
- suitability for LLM training


The analysis was performed in: `spam_dataset_quality_analysis.ipynb`

# 🧬 Synthetic Data Generation

Due to limitations of available datasets, a custom data generation pipeline was created.

The generator uses different LLM APIs to create additional high-quality spam examples.

Generated dataset:

- approximately 7000 samples
- diverse spam patterns
- controlled generation process
- improved training consistency

Implementation: `data_generator/`

# 🤖 LLaMA Fine-tuning

The project uses an Ollama-based LLaMA model adapted for spam classification.

The fine-tuning process was performed using:

- LLaMA 8B model
- QLoRA approach
- custom training configuration
- prepared classification datasets


Configuration: `classifier/config.py`

The training and evaluation pipeline is located in: `classifier/run_llama_classifier.py`

Run:
```powershell
python -X utf8 .\classifier\run_llama_classifier.py
```

The pipeline performs:

model loading
dataset preparation
fine-tuning
model saving
evaluation
metric calculation

# 🧪 Benchmark Against Classical ML Models

To evaluate the effectiveness of LLaMA fine-tuning, the model was compared with
traditional machine learning classifiers.

Implemented baseline models:

- Logistic Regression
- Random Forest
- Support Vector Machine
- Multinomial Naive Bayes
- Gradient Boosting
- Decision Tree
- K-Nearest Neighbors

# 📈 Results and Findings
## Performance on known data

On datasets similar to the training distribution, the fine-tuned LLaMA model achieved
results comparable to classical classifiers.

This shows that traditional approaches can still perform very well on controlled
classification tasks.

## Generalization Performance

The most important experiment was performed on an independent dataset that was not
included in training.

The fine-tuned LLaMA model achieved significantly better generalization compared
to classical classifiers.

## Key observations:

Better handling of unseen spam patterns
stronger semantic understanding
improved robustness outside the training distribution

This experiment showed the main advantage of LLM-based approaches.

# 🧠 Technical Insights

The project demonstrated that:

- data quality strongly affects model performance
- synthetic data generation can improve training datasets
- fine-tuning allows LLM adaptation to specialized tasks
- accuracy alone is insufficient for evaluating modern NLP models
- generalization is a critical factor in real-world applications