<<<<<<< HEAD

=======
# Student Placement Prediction Web App

This project is a complete machine learning web application built with Flask.
It predicts whether a student is likely to get placed and, if the prediction is positive, it also estimates the expected salary package in LPA.

The app uses the **Student Placement Prediction Dataset 2026** and includes:

- Placement prediction
- Salary prediction for placed students
- Placement probability
- Input validation
- Responsive modern UI
- Saved trained models using pickle

## Website Overview

The website contains a clean form where the user enters:

- CGPA
- Number of internships
- Number of projects
- Number of certifications
- Aptitude score
- Communication skills score
- Technical skills score
- Number of backlogs

After clicking **Predict Placement**, the app:

1. Validates the input values
2. Creates the required model features
3. Predicts placement status
4. Shows placement probability
5. Predicts salary if the student is likely to be placed

### Output shown on the website

If the student is likely to be placed:

- `Student is likely to be Placed`
- `Expected Salary: INR X LPA`
- `Placement Probability: XX%`

If the student is not likely to be placed:

- `Student is Not likely to be Placed`
- `Placement Probability: XX%`

## Machine Learning Details

### Dataset file

- `dataset.csv`

### Input features used in the model

- `cgpa`
- `internships`
- `projects`
- `certifications`
- `aptitude_score`
- `communication_skills`
- `technical_skills`
- `backlogs`

### Engineered features

The training script also creates:

- `experience_score = internships + projects + certifications`
- `overall_skill_score = average of aptitude, communication, and technical skills`

### Models trained

Classification models:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Regression models:

- Linear Regression
- Random Forest Regressor

### Best models selected

From the training run in this project:

- Best placement model: `Logistic Regression`
- Best salary model: `Linear Regression`

### Evaluation results from the latest training run

Classification:

- Logistic Regression: Accuracy `56.75%`, Precision `57.49%`, Recall `78.93%`
- Random Forest: Accuracy `53.92%`, Precision `56.36%`, Recall `68.17%`
- XGBoost: Accuracy `56.34%`, Precision `57.16%`, Recall `79.15%`

Regression:

- Linear Regression: RMSE `0.9987`
- Random Forest Regressor: RMSE `1.0378`

## Project Structure

```text
placement_project/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- dataset.csv
|-- student_placement_prediction_dataset_2026.csv
|-- placement_model.pkl
|-- salary_model.pkl
|-- templates/
|   |-- index.html
|-- static/
|   |-- styles.css
```

## How To Run

Open a terminal inside:

```powershell
C:\Users\ambika\Documents\Codex\2026-05-06\files-mentioned-by-the-user-archive\placement_project
```

### Option 1: Recommended PowerShell commands without activation

This is the easiest option if PowerShell blocks the activation script.

#### Step 1: Create a virtual environment

```powershell
python -m venv .venv
```

#### Step 2: Install dependencies using the virtual environment Python

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

#### Step 3: Train the models

```powershell
.\.venv\Scripts\python train_model.py
```

This will create or update:

- `placement_model.pkl`
- `salary_model.pkl`

#### Step 4: Run the Flask app

```powershell
.\.venv\Scripts\python app.py
```

#### Step 5: Open the website

Open this URL in your browser:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

### Option 2: Activate the virtual environment in PowerShell

If you want activation to work in PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
python app.py
```

### Option 3: Use Command Prompt

If you prefer Command Prompt instead of PowerShell:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python train_model.py
python app.py
```

## How To Use The Website

1. Open the home page in the browser.
2. Fill in the student details.
3. Use the sliders for CGPA and skill scores.
4. Select the number of backlogs from the dropdown.
5. Click `Predict Placement`.
6. Read the result shown on the page.

## Input Validation Rules

The app validates user inputs before making predictions.

- CGPA must be between `0` and `10`
- Aptitude score must be between `0` and `100`
- Communication skills must be between `0` and `100`
- Technical skills must be between `0` and `100`
- Internships must be between `0` and `20`
- Projects must be between `0` and `20`
- Certifications must be between `0` and `20`
- Backlogs must be between `0` and `20`

## Important Files

- `app.py`: Flask backend and prediction logic
- `train_model.py`: data cleaning, training, evaluation, and pickle export
- `templates/index.html`: web page structure
- `static/styles.css`: website styling
- `requirements.txt`: Python dependencies
- `placement_model.pkl`: trained placement model
- `salary_model.pkl`: trained salary model

## Troubleshooting

### PowerShell says running scripts is disabled

Use one of these:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

or skip activation completely:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python train_model.py
.\.venv\Scripts\python app.py
```

### Model files are missing

If the app shows a model loading error, run:

```powershell
.\.venv\Scripts\python train_model.py
```

### Flask app does not open

Make sure the server is running and then open:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

## Notes

- The project already includes trained model files.
- You can retrain the models anytime using `train_model.py`.
- The salary prediction is shown only when the placement prediction is positive.
- The UI is responsive and works on desktop and mobile screens.
>>>>>>> 03be172 (Initial commit)
