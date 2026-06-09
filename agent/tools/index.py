"""
tip:
一些工具
"""

from langchain_core.tools import tool
from typing import Union, Optional
from pydantic import BaseModel, Field
import requests
from langgraph.prebuilt import ToolNode
import json


class WeatherLoc(BaseModel):
    location: str = Field(description="这是城市的地址")


class SearchQuery(BaseModel):
    query: str = Field(description="网络搜索")


@tool(args_schema=SearchQuery)
def fetch_real_time_info(query):
    """
    获取真实时间在网络中
    """
    url = "https://f.m.suning.com/api/ct.do"

    response = requests.get(url)
    data = json.loads(response.text)

    # 时间戳
    return json.dumps(data["currentTime"], ensure_ascii=False)


@tool(args_schema=WeatherLoc)
def get_weather(location):
    """
    查询地点天气
    """

    # step1.构建请求
    url = "https://uapis.cn/api/v1/misc/weather"

    # step2.设置查询参数
    params = {"city": location, "lang": "zh"}

    # step3.发送请求
    response = requests.get(url, params=params)

    # step4.解析响应
    data = response.json()
    print(f"\n{location}:{str(data)}\n")
    return json.dumps((data))
