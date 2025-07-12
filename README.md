<div align="center">
  <h1><strong>📰 News Article Summarizer<br>(Mahakumbh Festival)</strong></h1>
  <p><em>Automated abstractive summarization of Indian news coverage using T5 Transformers</em></p>
</div>

---

## 📌 Project Overview

This project focuses on the **automated summarization** of news articles related to the **Mahakumbh Festival**, one of the largest religious gatherings in the world. The goal is to condense long news articles into concise, meaningful summaries using **state-of-the-art NLP techniques**.

The pipeline includes:
- **Data scraping** from Indian news sources
- **Cleaning and preprocessing** text
- **Training a transformer-based model (T5)** for summarization
- **Evaluating** performance using industry-standard metrics like ROUGE

---

## 🧭 Motivation

In the era of information overload, it's crucial to extract relevant content quickly. Manually summarizing articles is time-consuming, especially for large datasets. This project demonstrates how **abstractive summarization** using deep learning can help automate this process with high-quality results. We chose the **Mahakumbh Festival** due to its cultural relevance and rich media coverage in India.

---

## ✅ Task Breakdown

### 🔹 Task 1: Dataset Collection
- **Source:** Articles were scraped from *The Indian Express* website.
- **Tools Used:**  
  - [`Selenium`](https://www.selenium.dev/) for dynamic browser automation  
  - [`BeautifulSoup`](https://www.crummy.com/software/BeautifulSoup/) for parsing HTML and extracting content
- **Output:** Extracted article titles, dates, and summaries; saved as structured data in `mahakumbh_articles.csv`.

---

### 🔹 Task 2: Dataset Annotation
- **Text Cleaning:** Removed noisy HTML elements, fixed encoding issues, and normalized text.
- **Data Preparation:** Combined article titles and content for model input. The corresponding original summaries served as ground truth labels.
- **Annotation Format:**  
  - `Input`: Cleaned title + article content  
  - `Target`: Official article summary

---

### 🔹 Task 3: Model Development & Summarization

#### 🧠 Model
- **Architecture:** [T5-base](https://huggingface.co/t5-base) – a powerful encoder-decoder model for sequence-to-sequence tasks.
- **Framework:** [Hugging Face Transformers](https://huggingface.co/transformers/) and `Seq2SeqTrainer` for training and evaluation.

#### ⚙️ Pipeline
- Tokenized and split dataset into training and validation sets.
- Trained the model on news data using GPU-accelerated fine-tuning.
- Tracked:
  - **Training Loss**
  - **Validation Loss**
  - **ROUGE Scores** (ROUGE-1, ROUGE-2, ROUGE-L)

#### 📤 Output
- Generated abstractive summaries for unseen inputs.
- Final results saved to `outputT5_fixed.csv`.

---

## 📈 Evaluation Metrics

Used **ROUGE** scores to evaluate the quality of generated summaries:
- **ROUGE-1**: Overlap of unigrams
- **ROUGE-2**: Overlap of bigrams
- **ROUGE-L**: Longest common subsequence

These metrics help quantify how close the generated summaries are to the human-written ones.

---

## 🧑‍💻 Contributors & Work Distribution

| Name                      | Roll No.   | Contribution                                                                 |
|---------------------------|------------|------------------------------------------------------------------------------|
| **Aditya Vilasrao Bhagat** | 2411AI27   | Scraped and collected article data using Selenium and BeautifulSoup. Handled preprocessing and structured storage of the dataset. |
| **Divyanshu Singh**        | 2411AI41   | Led model development: implemented tokenization, prepared data pipeline, set up training and validation. |
| **Vaibhav Shikhar Singh**  | 2411AI48   | Focused on training and evaluation. Handled ROUGE scoring and performance tracking of the model. |

---

## 🛠️ Tech Stack

- **Languages:** Python  
- **Libraries & Tools:**  
  - `Selenium` – Web scraping  
  - `BeautifulSoup` – HTML parsing  
  - `Pandas`, `Numpy` – Data handling  
  - `Transformers`, `Datasets` – Hugging Face NLP models  
  - `Seq2SeqTrainer` – For training T5  
  - `scikit-learn`, `ROUGE` – Model evaluation

---

## 📁 Project Structure

