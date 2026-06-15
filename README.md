# AI Model Risk & Robustness Analysis Platform

An enterprise-grade platform designed to evaluate and demonstrate the operational resilience of machine learning models in credit risk and loan prediction scenarios. This project specifically tests model prediction reliability when faced with various forms of real-world data corruption and adversarial attacks.

## 🚀 Key Features

- 🤖 Models Evaluated
Random Forest (RF)
Logistic Regression (LR)
Support Vector Machine (SVM)
Decision Tree (DT)
<img width="1853" height="922" alt="Screenshot 2026-06-15 161621" src="https://github.com/user-attachments/assets/7c44e6f3-cc3f-4400-80f9-e706b15a3aec" />
<img width="542" height="531" alt="Screenshot 2026-06-15 161208" src="https://github.com/user-attachments/assets/77ba62db-b9e9-458b-ad58-3f908fd0c967" />
<img width="727" height="567" alt="Screenshot 2026-06-15 161356" src="https://github.com/user-attachments/assets/4c320896-e1ff-4caa-aeab-42ac193ea9e2" />
<img width="1257" height="357" alt="Screenshot 2026-06-15 161458" src="https://github.com/user-attachments/assets/c42048c0-dfbb-4138-a2b2-a2548c826c14" />
<img width="1261" height="275" alt="Screenshot 2026-06-15 161428" src="https://github.com/user-attachments/assets/23615074-4b22-4dfd-ae84-9575391b52d4" />

- **Real-time Degradation Simulation**:
  - **Gaussian Noise**: Simulates natural data entry errors and sensor noise.
  - **Missing Data (Dropout)**: Simulates incomplete data applications.
  - **Adversarial Perturbation**: Simulates targeted attacks optimizing for false loan approvals.
- **Dynamic Feature Alignment**: Automatically handles and pads continuous and categorical variable mismatches across 600+ complex features to prevent runtime crashes.
- **Real-Time Analytics Dashboard**: Visualizes performance degradation trajectories compared to a clean baseline.











## 🛠️ System Architecture

1. **Frontend (Dashboard)**:
   - Built with Vanilla JavaScript, HTML5, and CSS3.
   - Professional, data-dense UI with real-time Chart.js degradation visualization.
   - Allows users to interactively scale noise levels ranging from 0% to 100% and view the prediction outcome probabilities immediately.
2. **Backend Engine**:
   - Python-based **Flask REST API**.
   - Handles the ingestion of feature variables, aligns them directly to the exact feature map generated during model training (`columns.json`), and executes inferences.
3. **Machine Learning Pipeline**:
   - Models are serialized (`all_models.pkl`) and securely loaded into the backend.

## ⚡ How to Run Locally

### Prerequisites
- Python 3.9+
- A modern web browser

### Step 1: Install Dependencies
Open your terminal in the root of this repository and run:
```bash
pip install -r requirements.txt
```

### Step 2: Start the Backend Server
Boot up the Flask API that serves the models:
```bash
python app.py
```
*(The server will start locally at `http://127.0.0.1:5000`)*

### Step 3: Launch the Dashboard
Simply double-click the `index.html` file to open it in your default web browser. You can now use the credentials in the mock login screen to access the console, tweak parameters, and observe the robustness decay graphs dynamically!

## 🎓 Academic Value & Objectives
This project was designed for academic exploration of AI Safety and Robust AI methodologies within the FinTech industry. It practically illustrates that while ensemble ML algorithms excel in high-dimensional clean data environments, their decision-making certainty can shift profoundly under specific synthetic stresses.

---
*Created for academic evaluation and demonstration.*
