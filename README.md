
🏦 Loan Approval Prediction System

An intelligent ML-powered web app that predicts whether a loan should be approved or rejected based on applicant details such as income, employment type, credit history, and property area.

Built with Streamlit, RandomForestClassifier, and Pandas, this project demonstrates a complete end-to-end workflow — from data preprocessing and model training to interactive prediction through a modern UI.

🚀 Features

✅ Interactive Web Interface – Enter applicant details and get instant approval results
✅ Machine Learning Model – Trained using Random Forest for high accuracy
✅ Data Preprocessing – Handles missing values, label encodings, and dataset balancing
✅ Confidence Score – Displays the probability of approval or rejection
✅ Dark Mode UI – Clean, minimal design with dark theme aesthetics
✅ Input Summary Table – Displays encoded input features after prediction

🧠 Tech Stack
Layer	Technologies
Frontend	Streamlit
Backend / ML	Python, Scikit-Learn, Pandas, NumPy
Model	RandomForestClassifier
Persistence	joblib
Data Source	Loan dataset (loan_train.csv)
📂 Project Structure
loan-approval-prediction/
│
├── app.py                     # Streamlit UI
├── train_model.py             # Model training + encoding script
├── loan_train.csv             # Training dataset
├── loan_model.pkl             # Trained model
├── label_encoders.pkl         # Saved encoders for categorical data
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation

⚙️ Installation

Clone the repository

git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction


Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate        # For Windows
source venv/bin/activate     # For macOS/Linux


Install dependencies

pip install -r requirements.txt


Train the model (optional if not included)

python train_model.py


Run the Streamlit app

streamlit run app.py


Open your browser → http://localhost:8501

📊 How It Works

The dataset is loaded and cleaned (missing values handled, categorical variables encoded).

A RandomForestClassifier is trained on balanced data.

The trained model and label encoders are saved using joblib.

In the Streamlit app, user inputs are encoded using the same encoders.

The model predicts whether the loan is approved or not and displays a confidence percentage.

💻 Example Input
Feature	Value
Gender	Male
Married	Yes
Dependents	1
Education	Graduate
Self Employed	No
Applicant Income (₹)	5000
Coapplicant Income (₹)	2000
Loan Amount (₹ in thousands)	150
Loan Term (in days)	360
Credit History	1.0
Property Area	Urban

➡️ Output: ✅ Loan Approved (Confidence: 86%)

🧩 Model Accuracy

Displayed during training — example:

RandomForestClassifier Accuracy: 83.5%

🧠 Future Improvements

🔹 Hyperparameter tuning for higher accuracy

🔹 Add SHAP explanations for transparency

🔹 Save user prediction history

🔹 Deploy to Streamlit Cloud or Hugging Face Spaces

🧾 License

This project is licensed under the MIT License — free to use and modify with attribution.

👨‍💻 Author

Utkarsh Pandey
📍 Pune, Maharashtra
💡 Aspiring Entrepreneur | AI & Coding Enthusiast

## Steps to Run

1. Clone this project or extract the zip.
2. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/ninzaami/loan-predication
3. Place the dataset in the `data/` folder as `loan.csv`.
4. Run the training script:
   python train_model.py
5. Launch the Streamlit app:
   streamlit run app/streamlit_app.py
