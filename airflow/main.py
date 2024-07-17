from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import git

from mlflow.medical_experiment import run_mlflow_experiment


# Import your function from another file
from cleaning import cleaning
from preprocessing import train_test, scaler

def loading(path):
    df = pd.read_csv(path)
    return df
    
def cleaning_(df):
    return cleaning(df)

def preprocessing_(df):
    return preprocessing(df)

def save(df):
    df.to_csv('../../data/processed/diabetes.csv')

# Define your DAG
dag = DAG(
    'my_dag',
    start_date=datetime(2024, 4, 1),
    schedule_interval='@daily',
)

# Define your Airflow task
task1 = PythonOperator(
    task_id='loading',
    python_callable=loading,  # Call the function from the imported file
    dag=dag,
)

task2 = PythonOperator(
    task_id='cleaning',
    python_callable=cleaning,
    dag=dag,
)

task3 = PythonOperator(
    task_id='preprocessing',
    python_callable=preprocessing,
    dag=dag,
)
task4 = PythonOperator(
    task_id='saving_csv',
    python_callable=save,
    dag=dag,
)

# Function to check if a file exists in a Git repository
def check_file_in_repo():
    repo = git.Repo(#tofill)
    files = [item.a_path for item in repo.index.diff(None)]
    if "your_file.txt" in files:
        print("File found in repository")
    else:
        print("File not found in repository")

check_file_task = PythonOperator(
    task_id='check_file_task',
    python_callable=check_file_in_repo,
    dag=dag,
)

mlflow_task = PythonOperator(
    task_id='mlflow_task',
    python_callable=run_mlflow_experiment, 
    dag=dag,
)

# check que le modèle a été enregistré dans mlflow/mlruns/

task1 >> task2 >> task3 >> task4 >> check_file_task (#si ça passe) mlflow_task
