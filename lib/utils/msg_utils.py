import os
import requests
import json

def send_msg(msg, cfg, config_file='configs/local/dingtalk.txt'):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            webhook = f.readline().strip()
            proxy = f.readline().strip()
    else:
        return False
    if cfg.log_level not in ['INFO', 'WARNINING'] or cfg.local_rank != 0: 
        return False
    header = {
        'Content-Type': 'application/json',
        'Charset': 'UTF-8'
    }
    message ={
        'msgtype': 'text',
        'text': {
            'content': msg
        },
    }
    message_json = json.dumps(message)
    print('send msg begin')
    os.environ['http_proxy'] = proxy
    os.environ['https_proxy'] = proxy
    try:
        info = requests.post(url=webhook, data=message_json,headers=header, timeout=5)
        print('send msg end')
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        return True
    except:
        print('send msg error')
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        return False
    