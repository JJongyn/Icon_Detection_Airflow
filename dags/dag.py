import sys
import os
from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mobis.main import MobisMain
from include.notifications import notify_tasks_failure, notify_tasks_success, notify_dag_success, notify_dag_failure

# DAG 기본 설정
default_args = {
    'owner': 'jongyun',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
with DAG(
    dag_id='Mobis',
    default_args=default_args,
    description='Mobis UI',
    schedule_interval=None,  # 수동 실행
    catchup=False,
    start_date=datetime(2024,1,1),
    on_success_callback=notify_dag_success,
    on_failure_callback=notify_dag_failure,
) as dag:

    mobis = MobisMain()

    ### main 기능 ###
    
    def prepare_data_task(image_path=None, **kwargs):
        ti = kwargs['ti']
        output_path = mobis.prepare_data(image_path=image_path)
        ti.xcom_push(key='output_path', value=output_path)
        return output_path
    
    def process_data_task(**kwargs):
        # XCom에서 이전 태스크의 output_path 가져오기
        ti = kwargs['ti']
        output_path = ti.xcom_pull(task_ids='prepare_data', key='output_path')
        print(f"Processing data from path: {output_path}")
        
    def augment_data_task(**kwargs):
        ti = kwargs['ti']
        output_path = ti.xcom_pull(task_ids='prepare_data', key='output_path')
        mobis.augment_data(task_instance=output_path)
    
    def train_model(**kwargs):
        ti = kwargs['ti']
        results = mobis.train_model()
        ti.xcom_push(key='results', value=results)
    
    def report(**kwargs):
        ti = kwargs['ti']
        results = ti.xcom_pull(task_ids='Step3_Training_Navigation_Model', key='results')
        llm_response = mobis.report_result(results)
        ti.xcom_push(key='llm_response', value=llm_response)

    ### Task ###
    
    # Start
    start = BashOperator(
        task_id="Start_mobis_UI",
        bash_command="echo Start Mobis UI navigator!",
        on_success_callback=notify_tasks_success,
        on_failure_callback=notify_tasks_failure,
    )

    # Step1
    prepare_data_operator = PythonOperator(
        task_id='Step1_Preparing_data',
        python_callable=prepare_data_task,
        op_kwargs={'image_path': ''},
        on_success_callback=notify_tasks_success,
        on_failure_callback=notify_tasks_failure,
    )
    
    # Step2
    augment_data_operator = PythonOperator(
        task_id='Step2_Augmenting_data',
        python_callable=augment_data_task,
        on_success_callback=notify_tasks_success,
        on_failure_callback=notify_tasks_failure,
    )

    # Step3
    train_model_operator = PythonOperator(
        task_id='Step3_Training_Navigation_Model',
        python_callable=train_model,
        task_concurrency=1,
        on_success_callback=notify_tasks_success,
        on_failure_callback=notify_tasks_failure,
        execution_timeout=timedelta(minutes=60),
    )
    
    # Step4
    report_operator = PythonOperator(
        task_id='Training_results',
        python_callable=report,
        on_success_callback=notify_tasks_success,
        on_failure_callback=notify_tasks_failure,
    )
    
    # Final
    end = BashOperator(
        task_id="End_mobis_UI",
        bash_command="echo End Mobis UI navigator!",
        on_success_callback=notify_tasks_success,
        on_failure_callback=notify_tasks_failure,
    )
    
    # 태스크 순서 정의
    start >> prepare_data_operator >> augment_data_operator >> train_model_operator >> report_operator >> end
    