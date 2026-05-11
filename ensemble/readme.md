## README: Phishing Site Detection using Ensemble Methods

This project implements three core ensemble machine learning architectures from scratch to identify malicious phishing URLs. By combining multiple models, we improve the robustness and accuracy of the detection system compared to using a single classifier.

---

## 🏗️ Implemented Architectures

This assignment focuses on the "Wisdom of the Crowd" principle in Machine Learning:

1. **Bagging (Bootstrap Aggregating):** Reduces **variance** by training multiple Decision Trees on random subsets of the data (with replacement) and using a majority vote for the final prediction.
2. **Boosting (AdaBoost):** Reduces **bias** by training weak learners (stumps) sequentially. Each subsequent model focuses more on the samples that the previous models misclassified.
3. **Stacking (Stacked Generalization):** Uses a **Meta-Learner** (Logistic Regression) to intelligently combine the predictions of diverse base models (Decision Trees and Logistic Regression).

---

## 📁 Project Structure

* `phishing_detection.py`: The main Python script containing the "from scratch" implementations and the evaluation logic.
* `phishing_site_urls.csv`: The dataset containing URLs and their respective labels (`good` or `bad`).
* `venv/`: The Python virtual environment ensuring all dependencies are isolated.

---

## 🚀 Getting Started

### 1. Setup the Environment

Navigate to the project directory and create a virtual environment:

```bash
# Create the environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

```

### 2. Install Dependencies

Install the required libraries for data handling and text vectorization:

```bash
pip install pandas numpy scikit-learn

```

### 3. Run the Detection Model

Execute the script to train the models and view the performance metrics:

```bash
python phishing_detection.py

```

---

## 📊 Methodology & Evaluation

### Text Vectorization

Since URLs are text-based, we use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert strings into numerical features. This allows the models to identify suspicious keywords (e.g., "login", "bank", "verify") and structural patterns common in phishing links.

### Metrics

The project evaluates each ensemble method using:

* **Precision:** How many identified scams were actually scams.
* **Recall:** How many total scams we successfully caught (Critical for security).
* **F1-Score:** The harmonic mean of precision and recall.

---

## 📝 Assignment Notes

* **Data Sampling:** To ensure the "from scratch" logic runs efficiently on standard hardware, the script defaults to a sample of 20,000 records.
* **Manual Implementation:** The `Bagging`, `Boosting`, and `Stacking` logic is written manually without using pre-built ensemble classes from libraries like Scikit-Learn to demonstrate a deep understanding of the underlying math.
