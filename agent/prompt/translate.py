from langchain.messages import HumanMessage, SystemMessage

from agent.hil.breakpoint import State

# def translate_message(state: State):
#     system_prompt = """
#     你是一个翻译助手，负责将用户输入的文本翻译成英文。请确保翻译准确且自然。
#     """

#     messages = state["model_response"]

#     messages = [SystemMessage(content=system_prompt)] + [
#         HumanMessage(content=messages.content)
#     ]
#     response = llm.invoke(messages)
#     return {"model_response": response}
