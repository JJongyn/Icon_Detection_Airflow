# Automotive Icon Detection + Airflow
This project focuses on detecting navigation icons in automotive user interfaces using YOLO models, OCR, and Apache Airflow for process automation. The system streamlines data collection, labeling, and retraining, making it adaptable to new versions or updates in automotive applications.

## 📌 Architecture
<img width="1260" alt="process" src="https://github.com/user-attachments/assets/a986a2d4-0828-47c3-81f5-36762516d78b">

## 📌 **Run**

To run airflow server, run this command:

```
airflow webserver --port 8080
```

To run airflow scheduler, run this command:
```
airflow scheduler
```

## 💻 Airflow process 
You can check our airflow tasks in [**`dags/dag.py`**](https://github.com/JJongyn/Icon_Detection_Airflow/blob/86bcebe47a4d865cad199a6ea611686032c8d196/dags/dag.py#L26)

## 💬 Teams notification using airflow
You can check teams message using airflow in [**`include/notification.py`**](https://github.com/JJongyn/Icon_Detection_Airflow/blob/86bcebe47a4d865cad199a6ea611686032c8d196/include/notifications.py#L6)

## 📝 **Project Structure**
**`dags`**: Contains Airflow DAGs to automate the pipeline for data collection, preprocessing, and retraining.

**`include`**: Manages task alerts sent via Teams.

**`src/llm`**: Implements ChatGPT API for handling intelligent system interactions.

**`src/diffusion`**: Contains scripts for data augmentation using diffusion models.

**`src/data`**: Stores preparing and processed data for model training and evaluation.
