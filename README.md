<h1><strong>News Article Summarizer: Mahakumbh Festival</strong></h1>
  <p><em>Automated Abstractive Summarization of Indian News Coverage Using T5 Transformers</em></p>
</div>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aditya-ADII/News-Article-Summarizer/blob/main/Main/main.ipynb)

---

## Project Overview

This project develops an automated system for summarizing news articles about the **Mahakumbh Festival**, one of the world's largest religious gatherings. By leveraging state-of-the-art natural language processing (NLP) techniques, the system condenses lengthy articles into concise, informative summaries.

The end-to-end pipeline, implemented in a Google Colab notebook, includes:
- **Data Scraping**: Collecting articles from Indian news sources.
- **Text Preprocessing**: Cleaning and preparing the data.
- **Model Training**: Fine-tuning a T5 transformer model for abstractive summarization.
- **Evaluation**: Assessing performance with metrics such as ROUGE scores.

---

## Motivation

In today's information-saturated world, quickly extracting key insights from news articles is essential. Manual summarization is labor-intensive, particularly for large volumes of content. This project showcases how deep learning-based abstractive summarization can automate the process efficiently while maintaining high quality. The Mahakumbh Festival was selected as the focus due to its cultural significance in India and the abundance of related media coverage.

---

## Task Breakdown

### Task 1: Dataset Collection
- **Sources**: Articles scraped from *The Indian Express* website.
- **Tools Used**:
  - [Selenium](https://www.selenium.dev/) for dynamic browser automation.
  - [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing and content extraction.
- **Output**: Structured dataset containing article titles, dates, and summaries, saved as `mahakumbh_articles.csv`.

### Task 2: Dataset Preparation
- **Text Cleaning**: Removed HTML artifacts, resolved encoding issues, and normalized text.
- **Data Formatting**: Combined article titles and content as input; used original summaries as ground-truth targets.
- **Annotation Format**:
  - **Input**: Cleaned title + article body.
  - **Target**: Official summary.

### Task 3: Model Development and Summarization
#### Model Architecture
- **Base Model**: [T5-base](https://huggingface.co/t5-base), an encoder-decoder transformer optimized for sequence-to-sequence tasks.
- **Framework**: [Hugging Face Transformers](https://huggingface.co/transformers/) with `Seq2SeqTrainer` for training and evaluation.

#### Pipeline
- Tokenized the dataset and split it into training and validation sets.
- Fine-tuned the model on GPU for abstractive summarization.
- Monitored key metrics:
  - Training and validation loss.
  - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

#### Output
- Generated summaries for unseen articles.
- Results exported to `outputT5_fixed.csv`.

---

## Evaluation Metrics

The model's performance was evaluated using **ROUGE** scores, which measure the overlap between generated and reference summaries:
- **ROUGE-1**: Unigram overlap (focuses on individual words).
- **ROUGE-2**: Bigram overlap (captures phrase-level similarity).
- **ROUGE-L**: Longest common subsequence (assesses sentence structure preservation).

These metrics provide a quantitative assessment of summary quality and fidelity to the original content.

---

## Contributors and Work Distribution

| Name                      | Roll No.   | Contribution                                                                 |
|---------------------------|------------|------------------------------------------------------------------------------|
| **Aditya Vilasrao Bhagat** | 2411AI27   | Led data scraping using Selenium and BeautifulSoup; handled preprocessing and dataset storage. |
| **Divyanshu Singh**        | 2411AI41   | Managed model development, including tokenization, data pipeline setup, and training/validation configuration. |
| **Vaibhav Shikhar Singh**  | 2411AI48   | Oversaw model training, evaluation, ROUGE scoring, and performance analysis. |

---

## Tech Stack

- **Programming Language**: Python.
- **Key Libraries and Tools**:
  - **Web Scraping**: Selenium, BeautifulSoup.
  - **Data Handling**: Pandas, NumPy.
  - **NLP and Modeling**: Hugging Face Transformers, Datasets, Seq2SeqTrainer.
  - **Evaluation**: scikit-learn, ROUGE.

---
## Project Structure
```
News-Article-Summarizer/
├── LICENSE                    # Project license
├── README.md                  # Project overview, tasks, contributors, and setup instructions
└── Main/
  ├── mahakumbh_articles.csv # Raw scraped dataset (titles, dates, summaries)
  ├── main.ipynb             # Main Colab notebook containing the entire pipeline (scraping, preprocessing, training, evaluation, and inference)
  └── outputT5_fixed.csv     # Generated summaries from model inference
```
---

## Description of Key Components

- **`main.ipynb`**: This Google Colab notebook encapsulates the complete project workflow. It includes sections for:
  - Web scraping using Selenium and BeautifulSoup to collect Mahakumbh-related articles from The Indian Express.
  - Text cleaning, normalization, and preparation of input-target pairs for training.
  - Fine-tuning the T5-base model with Hugging Face's `Seq2SeqTrainer`, including tokenization, training, evaluation with ROUGE scores, and summary generation.
  - Helper functions for text cleaning and summary generation are defined inline.

- **`mahakumbh_articles.csv`**: Contains the scraped data, including article titles, publication dates, and original summaries.

- **`outputT5_fixed.csv`**: Stores the model-generated abstractive summaries alongside the source articles for easy comparison.

To run the project:
1. Open `main.ipynb` in Google Colab (use the badge above).
2. Install dependencies as listed in the notebook (e.g., via `!pip install` commands).
3. Execute the cells sequentially to scrape data, train the model, and generate summaries.

---

## License

This project is licensed under the terms specified in the `LICENSE` file.

---
