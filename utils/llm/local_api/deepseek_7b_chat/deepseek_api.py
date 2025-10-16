from flask import Flask, request, jsonify, Response, copy_current_request_context
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer  # 新增 TextStreamer

# 模型路径和设备配置（保持不变）
MODEL_PATH = "D:/NeuroSync/NeuroSync_Player-main/utils/llm/local_api/deepseek_7b_chat/ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Activated device: {device}")

app = Flask(__name__)

# 构建DeepSeek格式的Prompt（保持不变）
def build_deepseek_prompt(messages):
    prompt = "<s>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        prompt += f"{role}: {content}\n</s>"
    prompt += "assistant: "
    return prompt

# 生成配置（保持不变）
generation_config = GenerationConfig(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)

# 加载模型和tokenizer（保持不变）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
)
model.eval()
print("模型加载成功！")

# 新增：测试模型是否能正常生成内容
print("\n===== 测试模型生成能力 =====")
test_prompt = "<s>user: 你好，能看到这句话吗？</s>assistant: "
test_inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
try:
    test_outputs = model.generate(** test_inputs, max_new_tokens=20)
    test_response = tokenizer.decode(test_outputs[0], skip_special_tokens=True)
    print(f"模型测试输出：{test_response}")
except Exception as e:
    print(f"模型测试失败：{str(e)}")
print("===========================\n")

# 非流式接口（保持不变）
@app.route("/generate_llama", methods=["POST"])
def generate_llama():
    if not request.is_json:
        return jsonify({"error": "请求格式必须为JSON"}), 400
    request_data = request.get_json()
    messages = request_data.get("messages", [])
    if not messages:
        return jsonify({"error": "缺少messages参数"}), 400
    
    prompt = build_deepseek_prompt(messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model.generate(** inputs, generation_config=generation_config)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant: ")[-1]
    return jsonify({"response": full_response})

# 修正后的流式接口：预提取参数，避免生成器依赖request
@app.route("/generate_stream", methods=["POST"])
def generate_stream():
    # 第一步：预提取参数（保持不变）
    if not request.is_json:
        return Response(
            f"data: {json.dumps({'error': '请求格式必须为JSON'})}\n\n",
            mimetype="text/event-stream"
        )
    request_data = request.get_json()
    messages = request_data.get("messages", [])
    if not messages:
        return Response(
            f"data: {json.dumps({'error': '缺少messages参数'})}\n\n",
            mimetype="text/event-stream"
        )
    
    # 构建Prompt（保持不变）
    prompt = build_deepseek_prompt(messages)
   # 👇 新增日志放在这里
    print(f"===== 生成的Prompt内容 =====")
    print(prompt)
    print("===========================")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    # 第二步：定义生成器，使用TextStreamer替代as_streamer
    @copy_current_request_context
    def generate():
        try:
            print("\n=== 流式请求已接收 ===")
            print("=== 开始流式生成 ===")
        
            # 生成完整响应
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                output_scores=False
            )
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = full_response.split("assistant: ")[-1].strip()
            
            # 按词语/标点分割（保持自然断句）
            import re
            chunks = re.split(r'([，。, .!?；;])', assistant_response)
            chunks = [c for c in chunks if c.strip()]
            merged_chunks = []
            for i in range(len(chunks)):
                if i % 2 == 1:
                    if merged_chunks:
                        merged_chunks[-1] += chunks[i]
                    else:
                        merged_chunks.append(chunks[i])
                else:
                    merged_chunks.append(chunks[i])
            
            # 直接返回纯文本，不添加任何前缀
            for chunk in merged_chunks:
                if chunk:
                    print(f"生成分块：{chunk}")
                    # 仅返回文本内容，加换行符分隔
                    yield f"{chunk}\n"
            
            print("=== 流式生成结束 ===")
        
        except Exception as e:
            error_msg = f"错误：{str(e)}"
            print(error_msg)
            yield f"{error_msg}\n"  # 错误信息也仅返回纯文本

    # 响应类型设为纯文本
    return Response(generate(), mimetype="text/plain; charset=utf-8")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=False, debug=False)
