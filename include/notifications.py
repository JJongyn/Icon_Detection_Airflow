import requests
from airflow.models import Variable

mobis_logo = 'https://wiki1.kr/images/9/92/%ED%98%84%EB%8C%80%EB%AA%A8%EB%B9%84%EC%8A%A4%E3%88%9C_%EA%B8%80%EC%9E%90.png'

def send_teams_notification(
    title, summary, theme_color, sections, actions=None, logo_url=mobis_logo
):
    """
    Microsoft Teams로 알림을 보내는 공통 함수.

    Args:
        title (str): 메시지 제목
        summary (str): 메시지 요약
        theme_color (str): 메시지 색상 코드
        sections (list): 메시지 본문에 포함될 섹션들
        actions (list): 추가 버튼이나 링크 등의 액션
    """
    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "title": title,
        "summary": summary,
        "themeColor": theme_color,
        "sections": sections,
        "potentialAction": actions or [],
    }

    if logo_url:
        payload["sections"].insert(0, {
            "activityTitle": f"![Company Logo]({logo_url})",
            "markdown": True
        })
        
        
    teams_webhook_url = Variable.get('teams_webhook_secret')
    headers = {"content-type": "application/json"}

    response = requests.post(teams_webhook_url, json=payload, headers=headers)

    if response.status_code == 200:
        print(f"Teams notification sent: {title}")
    else:
        print(f"Failed to send Teams notification: {response.status_code} - {response.text}")


##### Task 알림 관련 #####
def format_task_section(context, status):
    """Task 알림의 섹션 데이터를 생성."""
    task_id = context['task_instance_key_str']
    dag_id = context['dag'].dag_id
    log_url = context['task_instance'].log_url
    logical_date = context['ds']
    emoji = "✅" if status == "Succeeded" else "❌"

    return {
        "activityTitle": f"[Task] **{task_id}** {status} {emoji}",
        "activitySubtitle": f"[DAG]: **{dag_id}**",
        "facts": [
            {"name": "[Logical Date]", "value": logical_date},
            {"name": "[Log URL]", "value": f"[Click Here]({log_url})"}
        ],
        "markdown": True,
    }

def notify_tasks_failure(context):
    section = format_task_section(context, "Failed")
    send_teams_notification(
        title="❌ Airflow Task Failed",
        summary=f"DAG '{context['dag'].dag_id}' - Task '{context['task_instance_key_str']}' Failed",
        theme_color="FF0000",
        sections=[section]
    )

def notify_tasks_success(context):
    section = format_task_section(context, "Succeeded")
    
    # xcom에서 llm_response 가져오기
    task_instance = context['task_instance']
    llm_response = task_instance.xcom_pull(task_ids='Training_results', key='llm_response')
    
    # llm_response가 있을 경우에만 알림에 포함
    if llm_response:
        section['facts'].append({
            "name": "LLM Response",
            "value": llm_response
        })
        task_instance.xcom_push(key='llm_response', value=None)
        
    # Teams로 알림 전송
    send_teams_notification(
        title="✅ Airflow Task Succeeded 🎉",
        summary=f"DAG '{context['dag'].dag_id}' - Task '{context['task_instance_key_str']}' Succeeded",
        theme_color="28A745",
        sections=[section]
    )

##### DAG 알림 관련 #####
def format_dag_section(context, status):
    dag_id = context['dag'].dag_id
    run_id = context['run_id']
    log_url = context['dag_run'].log_url
    emoji = "🎉" if status == "Succeeded" else "❌"

    return {
        "activityTitle": f"DAG **{dag_id}** {status} {emoji}",
        "activitySubtitle": f"Run ID: {run_id}",
        "facts": [
            {"name": "Execution Date", "value": context['execution_date'].strftime('%Y-%m-%d %H:%M:%S')},
            {"name": "Log URL", "value": f"[Click Here]({log_url})"}
        ],
        "markdown": True,
    }

def notify_dag_success(context):
    section = format_dag_section(context, "Succeeded")
    send_teams_notification(
        title="🎉 DAG Succeeded",
        summary=f"DAG '{context['dag'].dag_id}' Succeeded",
        theme_color="28A745",
        sections=[section]
    )

def notify_dag_failure(context):
    section = format_dag_section(context, "Failed")
    send_teams_notification(
        title="❌ DAG Failed",
        summary=f"DAG '{context['dag'].dag_id}' Failed",
        theme_color="FF0000",
        sections=[section]
    )
