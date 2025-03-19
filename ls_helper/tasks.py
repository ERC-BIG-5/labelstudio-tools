from typing import Optional

from ls_helper.my_labelstudio_client.client import ls_client


def strict_update_project_task_data(task_id: int,
                                    new_data: dict,
                                    existing_data: Optional[dict] = None):
    client = ls_client()
    if existing_data:
        task_resp = client.get_task(task_id)
        if task_resp.status_code != 200:
            print(f"Error, task with id: {task_id} does not exist")
            return
        task_data = task_resp.json()["data"]
        if existing_data["data"] != task_data:
            print(f"Error, task with id: {task_id} does not match existing data")
            return
        send_data = {"data": new_data["data"]}
    resp = client.patch_task(task_id, send_data)
    if resp.status_code != 200:
        raise Exception(f"Error, task with id: {task_id} does not exist or could not be updated")
