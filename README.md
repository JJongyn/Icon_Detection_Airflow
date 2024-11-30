# Automotive Icon Detection + Airflow
This project focuses on detecting navigation icons in automotive user interfaces using YOLO models, OCR, and Apache Airflow for process automation. The system streamlines data collection, labeling, and retraining, making it adaptable to new versions or updates in automotive applications.

## **Run**

To run airflow server, run this command:

```
airflow webserver --port 8080
```

To run airflow scheduler, run this command:
```
airflow scheduler
```


## **Project Structure**
**`dags`**: Contains Airflow DAGs to automate the pipeline for data collection, preprocessing, and retraining.

**`include`**: Manages task alerts sent via Teams.

**`src/llm`**: Implements ChatGPT API for handling intelligent system interactions.

**`src/diffusion`**: Contains scripts for data augmentation using diffusion models.

**`src/data`**: Stores preparing and processed data for model training and evaluation.
