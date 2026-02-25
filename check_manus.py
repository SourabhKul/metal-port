#!/usr/bin/env python3
import json
import requests
import sys

# Manus API Configuration
API_KEY = "sk--WPHhbtI2vchXvG_77OEVq9q7ll4GwLZMYSEbHKd4qxiCkD3ZdDXeilh-43iswclETYAV9DYR2xgmDHUPAlULs0Yz_3C"
API_URL = "https://api.manus.ai/v1/tasks"

def get_manus_task(task_id):
    headers = {
        "API_KEY": API_KEY,
    }
    
    url = f"{API_URL}/{task_id}"
    print(f"Checking Manus Task: {task_id}...")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result
    except Exception as e:
        print(f"Error checking Manus task: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return None

if __name__ == "__main__":
    task_id = "A4MQUPyWKBaTiGtxkayh2s"
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
        
    task = get_manus_task(task_id)
    if task:
        print(json.dumps(task, indent=2))
