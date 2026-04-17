# Cognitive Risk Predictor

A machine learning application for estimating the probability of cognitive impairment using clinical and cognitive test data. This project combines predictive modeling with an interactive web interface to support interpretable and accessible risk assessment.

---

## Overview

This project develops a **Random Forest–based model** to estimate the likelihood of cognitive impairment. The model is calibrated to prioritize **high sensitivity (≥90%)**, reflecting a clinical preference to minimize missed cases.

An interactive web application built with Streamlit allows users to input patient features and receive:
- A predicted **risk probability**
- A **threshold-based classification** (high vs low risk)
- A structured, color-coded interface for improved usability

---

## Features

- Random Forest classifier trained on clinical and cognitive data  
- Sensitivity-calibrated decision threshold  
- Multiple feature configurations:
  - Full feature set  
  - Reduced feature set (with and without demographics)  
- Interactive Streamlit web app  
- Color-coded feature groups:
  - 🔵 Demographics  
  - 🟢 Raw cognitive scores  
  - 🟣 Domain-level (z-score) features  
- Multi-column layout for efficient data entry  
