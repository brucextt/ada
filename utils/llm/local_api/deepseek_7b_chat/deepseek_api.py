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
    print(f"构建的Prompt：{prompt[:100]}...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    # 第二步：定义生成器，使用TextStreamer替代as_streamer
    @copy_current_request_context
    def generate():
        try:
            print("\n=== 流式请求已接收 ===")
            print("=== 开始流式生成 ===")
            
            # 创建通用的TextStreamer（替代tokenizer.as_streamer）
            streamer = TextStreamer(
                tokenizer,
                skip_prompt=True,  # 跳过输入的prompt
                skip_special_tokens=True,  # 跳过特殊token（如<s>、</s>）
                decode_kwargs={"clean_up_tokenization_spaces": False}
            )
            
            # 模型生成时使用TextStreamer
            for output in model.generate(
                **inputs,
                generation_config=generation_config,
                streamer=streamer,  # 使用新创建的streamer
                pad_token_id=tokenizer.eos_token_id
            ):
                # 解码当前生成的token并返回
                response_chunk = tokenizer.decode(output, skip_special_tokens=True).strip()
                print(f"生成分块：{response_chunk[:50]}...")
                yield f"data: {json.dumps({'response': response_chunk})}\n\n"
            print("=== 流式生成结束 ===")
        
        except Exception as e:
            error_msg = f"流式生成异常：{str(e)}"
            print(error_msg)
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

    # 返回包装后的响应（保持不变）
    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=False, debug=False)
