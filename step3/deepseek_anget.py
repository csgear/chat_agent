'''
MCP 客户端程序
通过 stdio 协议与 MCP 服务器进行无缝通信
精准获取 MCP 服务器的工具清单和调用参数
并灵活调用工具实现 MCP 服务中丰富多样的函数功能
'''

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters  # 从 mcp 模块导入关键类，为通信与会话管理提供支持
from mcp.client.stdio import stdio_client  # 从 mcp.client.stdio 模块导入通信函数，搭建客户端与服务器之间的桥梁
import sys  # 导入 sys 模块，用于处理命令行参数，增强程序的灵活性
import json  # 导入 json 模块，用于高效处理 JSON 数据，实现数据的结构化存储与传输
import os  # 导入 os 模块，用于处理文件路径和环境变量，确保程序在不同环境下的稳定运行


# 定义一个功能强大的函数，用于将 tools 工具中的 JSON 数据转换为清晰易读的格式
def transform_json(tools):
    s = "MCP 服务器提供的工具如下:\n"
    for tool in tools:  # 遍历工具列表，逐一解析每个工具的信息
        s = s + f"""
            工具名称: {tool.name},
            工具描述: {tool.description},
            - 输入参数标题: {tool.inputSchema['title']},
            - 输入参数属性: {json.dumps(tool.inputSchema['properties'], indent=4, ensure_ascii=False)},

            """
    return s


# 定义与 DeepSeek 大模型交互的核心函数，实现用户意图识别与工具调用命令生成
def ask_llm_deepseek(question, tools_list):
    """
    该函数通过精心设计的系统提示（system prompt），引导 DeepSeek 大模型根据用户问题和工具描述，生成符合要求的工具调用命令。

    参数:
        question (str): 用户提出的问题，是模型理解用户需求的关键依据。
        tools_list (str): 包含工具详细描述的字符串，为模型提供工具集的相关信息，辅助其做出正确的工具选择。

    返回:
        tuple: 包含生成的 JSON 格式工具调用命令文本和 OpenAI 客户端对象的元组，便于后续处理与调用。
    """
    system_prompt = tools_list + '\n 根据以上描述，用户要求: %s ，请生成一个工具调用命令，要求以 json 格式输出{"tool": 工具名, "tool_input": 参数字典}，只输出 json，不要输出其他内容' % (
        question)

    # 创建 OpenAI 客户端，配置 API 密钥和基础 URL，确保与 DeepSeek 模型的正常通信
    client = OpenAI(
        api_key="sk-282074c41d594514aee6fd6f179ed292",  # 实际应用中，应从安全的环境变量或配置文件中获取 API 密钥
        base_url="https://api.deepseek.com/beta",
    )

    # 调用 DeepSeek 模型生成响应
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},  # 系统提示，为模型设定任务背景和要求
            {"role": "user", "content": "Hello"},  # 示例用户消息，可根据实际情况调整或移除
        ],
        max_tokens=1024,  # 限制生成文本的最大长度，避免响应过长
        temperature=0.99,  # 控制生成文本的随机性和创造性，数值越高，输出越多样
        stream=False  # 禁用流式输出，一次性获取完整响应
    )

    generated_text = response.choices[0].message.content  # 提取生成的文本内容
    return generated_text, client  # 返回生成的文本和客户端对象，便于后续调用其他函数


# 定义与 DeepSeek 模型进行简单交互的函数，用于获取直接的回答
def llm_deepseek(question, client):
    """
    该函数直接调用 DeepSeek 模型，根据用户问题生成回答。

    参数:
        question (str): 用户提出的问题，是模型生成回答的依据。
        client (OpenAI): OpenAI 客户端对象，用于与 DeepSeek 模型进行通信。

    返回:
        str: 模型生成的回答文本。
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": question},  # 用户消息，包含用户提出的问题
        ],
        max_tokens=1024,  # 限制生成文本的最大长度
        temperature=0.99,  # 控制生成文本的随机性和创造性
        stream=False  # 禁用流式输出
    )
    generated_text = response.choices[0].message.content  # 提取生成的文本内容
    return generated_text


# 获取服务器脚本路径，为启动服务器进程做准备
server_script_path = os.path.join(os.path.dirname(__file__), "base_customer_mcp_tools.py")

# 定义服务器参数，配置服务器进程的运行环境
server_params = StdioServerParameters(
    command="python",  # 运行命令，指定使用 Python 解释器
    args=[server_script_path],  # 服务器脚本路径，指向包含工具实现的脚本文件
    env=None  # 可选的环境变量，用于配置服务器进程的运行环境
)


# 定义异步运行函数，处理用户查询并调用相应工具
async def run(question="你是谁?"):
    """
    该异步函数是整个智能客服流程的核心，负责处理用户查询、调用大模型进行意图识别、调度工具并生成最终回答。

    参数:
        question (str, optional): 用户提出的问题，默认为 "你是谁?"。
    """
    # 建立与 MCP 服务器的连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接，确保客户端与服务器之间的通信正常
            await session.initialize()

            # 列出可用工具，获取工具的详细信息
            tools = await session.list_tools()
            # 将工具信息转换为易读的格式
            s = transform_json(tools.tools) + "\n"
            # 调用大模型进行意图识别和工具调用命令生成
            response, client = ask_llm_deepseek(question, s)
            # 清理生成的 JSON 字符串中的 Markdown 标记，确保格式正确
            response = response.strip().replace('```json', '').replace('```', '').strip()
            # 将清理后的字符串解析为 JSON 对象
            mtools = json.loads(response)  # 现在 response 是纯 JSON 字符串
            print("解析后的工具调用命令: --> ", mtools)

            # 检查生成的命令中是否包含工具名称
            if 'tool' in mtools:
                tool_name = mtools['tool']  # 获取工具名称
                tool_input = mtools['tool_input']  # 获取工具输入参数

                # 调用指定的工具，执行相应的任务
                print("正在调用工具:", tool_name, "输入参数:", tool_input)
                ret = await session.call_tool(tool_name, tool_input)  # 异步调用工具
                if ret:
                    try:
                        # 尝试将工具返回的结果解析为 JSON 对象
                        r = json.loads(ret.content[0].text)
                    except:
                        # 如果解析失败，则直接使用返回的文本内容
                        r = ret.content[0].text
                    print("工具返回结果:", r)
                    # 根据工具返回结果和用户问题，生成最终回答
                    questions = f"用户的问题是{question}，根据{tool_name}的返回结果为：{r}，根据以上信息，回答问题。"
                    r = llm_deepseek(questions, client)  # 调用大模型生成回答
                    print("最终回答:", r)
            else:
                # 如果没有生成有效的工具调用命令，则直接调用大模型回答用户问题
                r = llm_deepseek(question, client)
                print("直接回答:", r)


if __name__ == "__main__":
    import asyncio

    # 定义一组测试问题，用于验证智能客服系统的功能
    questions = [
        '帮我查一下订单，订单编号是 5241368562',
        '我觉得你们家商品价格太高！',
        "手机的质保是多久？",
    ]
    # 遍历测试问题，逐一处理并输出结果
    for question in questions:
        asyncio.run(run(question))