import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

MAAS_API_URL = os.getenv("MAAS_API_URL", "https://api.modelarts-maas.com/v2/chat/completions")
MAAS_API_KEY = os.getenv("MAAS_API_KEY")
MAAS_MODEL = os.getenv("MAAS_MODEL", "glm-5.1")


def test_maas_chat_basic():
    """基本对话测试：发送一条消息并验证响应"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MAAS_API_KEY}",
    }
    data = {
        "model": MAAS_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好"},
        ],
    }
    response = requests.post(MAAS_API_URL, headers=headers, data=json.dumps(data), verify=False, timeout=30)

    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 200, f"请求失败，状态码: {response.status_code}"

    result = response.json()
    assert "choices" in result, f"响应中缺少 choices 字段: {result}"
    assert len(result["choices"]) > 0, "choices 为空"
    assert "message" in result["choices"][0], "choice 中缺少 message 字段"

    content = result["choices"][0]["message"]["content"]
    print(f"模型回复: {content}")
    assert content, "模型回复内容为空"


def test_maas_chat_multi_turn():
    """多轮对话测试"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MAAS_API_KEY}",
    }
    data = {
        "model": MAAS_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请记住这个数字：42"},
            {"role": "assistant", "content": "好的，我记住了数字42。"},
            {"role": "user", "content": "我刚才让你记住的数字是多少？"},
        ],
    }
    response = requests.post(MAAS_API_URL, headers=headers, data=json.dumps(data), verify=False, timeout=30)

    assert response.status_code == 200, f"请求失败，状态码: {response.status_code}"

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    print(f"多轮对话回复: {content}")
    assert "42" in content, f"模型未正确回忆数字42，回复: {content}"


def test_maas_chat_stream():
    """流式对话测试"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MAAS_API_KEY}",
    }
    data = {
        "model": MAAS_MODEL,
        "messages": [
            {"role": "user", "content": "用一句话介绍Python语言"},
        ],
        "stream": True,
    }
    response = requests.post(MAAS_API_URL, headers=headers, data=json.dumps(data), verify=False, timeout=30)

    assert response.status_code == 200, f"请求失败，状态码: {response.status_code}"

    full_content = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        chunk = json.loads(payload)
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta:
            full_content += delta["content"]

    print(f"流式回复: {full_content}")
    assert full_content, "流式回复内容为空"
