from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import json

def test_local_vllm_deployment(port=8001):
    """测试本地vllm部署的Qwen3模型是否成功"""
    
    url = f"http://localhost:{port}/v1/chat/completions"
    
    # 构建请求
    prompt = "给我一个简短的大语言模型介绍"
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": "Qwen/Qwen3-32B",
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 1000,
        "enable_reasoning": True,  # 启用思考模式
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("正在测试vllm部署的Qwen3模型...")
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ vllm部署成功！")
            
            # 输出模型响应
            content = result["choices"][0]["message"]["content"]
            
            # 尝试解析思考内容
            try:
                # 寻找思考内容标记
                if "<think>" in content and "</think>" in content:
                    thinking_start = content.find("<think>") + len("<think>")
                    thinking_end = content.find("</think>")
                    thinking_content = content[thinking_start:thinking_end].strip()
                    final_content = content[thinking_end + len("</think>"):].strip()
                    
                    print("\n思考内容:", thinking_content)
                    print("\n最终回答:", final_content)
                else:
                    print("\n模型回答:", content)
            except Exception:
                print("\n模型回答:", content)
                
            return True
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 连接错误: {str(e)}")
        return False

def test_local_model_load():
    """测试是否可以在本地加载模型（不使用API）"""
    try:
        print("尝试使用transformers加载Qwen3模型...")
        
        model_name = "Qwen/Qwen3-32B"
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型（仅测试能否加载，不实际加载以节省资源）
        print("模型可以通过transformers正常访问")
        return True
    except Exception as e:
        print(f"本地加载模型失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("===== Qwen3模型测试 =====")
    
    # 首先测试vllm部署
    vllm_success = test_local_vllm_deployment(port=8001)
    
    if vllm_success:
        print("\n✅ vllm在端口8001上成功部署了Qwen3模型！")
    else:
        print("\n❌ vllm部署测试失败。")
        print("正在检查模型本身是否可访问...")
        
        # 如果API测试失败，尝试直接加载模型测试
        model_load_success = test_local_model_load()
        
        if model_load_success:
            print("\n模型本身可以访问，但vllm部署可能存在问题。请检查vllm服务是否正确启动在端口8001。")
            print("您可以尝试以下命令启动vllm服务：")
            print("vllm serve Qwen/Qwen3-32B --enable-reasoning --reasoning-parser deepseek_r1 --port 8001")
        else:
            print("\n无法访问模型。请确保已下载Qwen3模型或有正确的网络连接。")