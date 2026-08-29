# 📋 Chest X-Ray Report Generation & Research Overview

## 📌 Project Overview
This project focuses on automated radiology report generation by bridging the semantic gap between visual image data and patient clinical data[cite: 1]. The system uses visual diagnostic inputs, patient vitals, and clinical notes to generate structured, human-readable medical reports while maintaining audit compliance and security standards[cite: 1].

---

## 📖 Research Paper Context: "Beyond images: an integrative multi-modal approach to chest x-ray report generation"

### What Was Accomplished (The "Done")
* Built a novel multi-modal deep neural network combining chest X-rays with 11 supplementary non-imaging features, such as heart rate, temperature, gender, and clinical notes[cite: 1].
* Introduced a "conditioned cross-multi-head attention module" to bridge the gap between visual pixel data and textual/structured patient data[cite: 1].
* Proved through ablation studies that providing all data simultaneously (FullFusion) yields higher accuracy than relying on images alone[cite: 1].
* Achieved the highest ROUGE-L score (0.331) compared to similar state-of-the-art models in the literature[cite: 1].

### Future Work (The "Not Done")
* **Diverse Datasets:** The data solely originates from databases within a single institution, indicating a need to enhance data diversity from various sources to improve the overall robustness of the study[cite: 1].
* **Nuance and Detail Refinement:** The model requires continued refinement to capture more nuanced details and clinical context, as it sometimes fails to capture anomaly variations when the patient is inclined to the right or left[cite: 1].
* **Efficiency for Real-time Use:** The complexity and resource-intensive nature of the multi-modal deep neural network framework may hinder its real-time application in medical settings with limited computational power[cite: 1].

### Merits (Strengths)
* **Mimics Real Clinical Workflows:** It includes data that clinicians typically consider during patient evaluations, such as vital signs and reported symptoms[cite: 1].
* **High Linguistic Coherence:** The order of findings in the generated reports aligns with the reports written by radiologists, making them structurally correct[cite: 1].
* **Strong Baseline Accuracy:** The model reliably notes the presence or absence of abnormalities like pulmonary edema, pleural effusion, pneumonia, and pneumothorax[cite: 1].

### Demerits (Weaknesses)
* **Misses Small Artifacts:** The model often misses surgical materials like catheters and clips, and it lacks sensitivity to bone lesions like scoliosis[cite: 1].
* **Hallucinations:** In some cases, the model hallucinates elements, such as identifying mediastinal clips that are not present in the ground truth or the image[cite: 1].
* **Repetitive Grammar:** Some of the generated reports exhibit repeated words or phrases, and grammatical inconsistencies exist, such as using "and" at the beginning of sentences or concluding paragraphs with "the" or "is"[cite: 1].

---

## 📚 Literature Review & Limitations of Prior Work

| Research Paper | Proposed Solution | Limitations / Challenges |
| :--- | :--- | :--- |
| **Yuan et al. (2019)** | Enriched the decoder by explicitly concatenating medical concepts extracted via Semrep and using a concept-aware attention mechanism[cite: 1]. | Extracting concepts only from previous reports limits the diversity of expressions in generated reports[cite: 1]. |
| **Chen et al. (2020)** | Introduced a medical report generator utilizing a memory-driven Transformer with a relational memory (RM) module to retain knowledge from previous cases[cite: 1]. | Generic memory components may struggle with highly specific contextual details without multi-modal patient context[cite: 1]. |
| **Singh et al. (2021)** | Adopted a two-stage approach that generates the "Findings" section first, and then summarizes it into an "Impression" section based on whether the report is normal or abnormal[cite: 1]. | Primarily utilizes visual features and report texts, disregarding structured patient vitals[cite: 1]. |
| **Nooralahzadeh et al. (2021)** | Proposed a progressive Transformer-based framework using a pre-trained CNN, a mesh-memory Transformer, and BART as the language model[cite: 1]. | Generates high-level context strictly from the given X-ray, lacking integration of demographic data or emergency records[cite: 1]. |
| **Yang et al. (2022)** | Introduced a task-aware framework designed to understand specific diagnostic tasks related to various medical conditions[cite: 1]. | Prioritizes task-awareness for specific imaging types rather than addressing the fusion of asynchronous, heterogenous data sources[cite: 1]. |

---

## 📊 Dataset & Environment Setup
The dataset used in this study was created by leveraging three openly accessible databases[cite: 1]:
* **MIMIC-CXR (v2.0):** Encompasses 377,110 CXR images and 227,835 de-identified radiology reports[cite: 1].
* **MIMIC-IV (v2.0):** Comprises de-identified patient data, including characteristics like age, gender, ethnicity, and marital status[cite: 1].
* **MIMIC-IV-ED (v2.2):** Contains detailed clinical information from emergency department admissions, including diagnosis, medication, triage, and vital signs[cite: 1].
* **Curation:** Record linkage was performed to ensure non-imaging data was collected within the same time frame as the chest X-ray, resulting in a balanced subset of 3,000 samples to minimize biases skewed towards normal cases[cite: 1].

---

## 📈 Evaluation Metrics
The linguistic quality of the generated reports is computed using several automated evaluation metrics[cite: 1]:
* **BLEU (1-4):** Calculated to assess n-gram precision, where higher scores indicate greater local word-level similarity between the generated and reference texts[cite: 1].
* **ROUGE-L:** Used to measure the longest common subsequence, assessing the quality of the generated text in terms of recall and precision[cite: 1].
* **Bio-ClinicalBERT Score:** A domain-adapted evaluation metric that emphasizes clinical conceptual similarity by utilizing embeddings trained on scientific text and clinical notes, allowing for a more nuanced assessment of semantic meaning[cite: 1].

---

## 🧠 Model Selection Rationale
* **EfficientNetB0 CNN:** Utilized as the base model to extract a 1280-length visual feature vector from the images[cite: 1].
* **Transformer Architecture:** Selected because Transformers do not rely on recurrence, enabling faster and more effective learning by including more context in the network via self-attention mechanisms[cite: 1].
* **Cross-Attention Module:** Employed specifically to take image features as the query and the unified patient representation as the key and value, allowing the model to condition each part of the image embedding on relevant semantic patient data[cite: 1].

---

## 🔬 Planned Experiments (Ablation Studies)
To analyze the contribution of each distinct data feature to model performance, an ablation study incrementally presented different features alongside the chest X-ray images[cite: 1]:
* **Baseline:** Employed only chest X-ray images as input to generate corresponding reports, serving as the benchmark reference[cite: 1].
* **Singular:** Incorporated a single additional feature alongside the X-ray, such as evaluating oxygen saturation (O2Sat) individually[cite: 1].
* **TextFusion:** Explored fusing textual features of reported primary symptoms and ICD diagnostic codes with the chest X-rays[cite: 1].
* **ScalarFusion:** Combined multiple predictive scalar features with the images, including O2Sat, diastolic blood pressure, temperature, patient acuity scores, and gender[cite: 1].
* **FullFusion:** Takes a holistic approach by fusing all available and relevant data points, demonstrating substantial enhancements in various metrics[cite: 1].
