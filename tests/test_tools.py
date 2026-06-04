"""工具单元测试"""

import pytest

from agent.tools import calculator, get_current_time, text_analyzer, unit_converter


class TestCalculator:
    """计算器工具测试"""

    def test_basic_addition(self):
        result = calculator.invoke({"expression": "2 + 3"})
        assert "5" in result

    def test_complex_expression(self):
        result = calculator.invoke({"expression": "(15 * 23 + 47) / 8"})
        assert "49.25" in result

    def test_math_functions(self):
        result = calculator.invoke({"expression": "sqrt(16)"})
        assert "4" in result

    def test_invalid_expression(self):
        result = calculator.invoke({"expression": "invalid"})
        assert "错误" in result


class TestTextAnalyzer:
    """文本分析工具测试"""

    def test_basic_analysis(self):
        result = text_analyzer.invoke({"text": "Hello World"})
        assert "单词数: 2" in result

    def test_chinese_text(self):
        result = text_analyzer.invoke({"text": "你好世界"})
        assert "中文字符数: 4" in result


class TestUnitConverter:
    """单位转换工具测试"""

    def test_length_conversion(self):
        result = unit_converter.invoke(
            {"value": 1, "from_unit": "km", "to_unit": "m"}
        )
        assert "1000" in result

    def test_temperature_conversion(self):
        result = unit_converter.invoke(
            {"value": 100, "from_unit": "fahrenheit", "to_unit": "celsius"}
        )
        assert "37.78" in result

    def test_weight_conversion(self):
        result = unit_converter.invoke(
            {"value": 1, "from_unit": "kg", "to_unit": "g"}
        )
        assert "1000" in result


class TestGetCurrentTime:
    """时间工具测试"""

    def test_returns_time(self):
        result = get_current_time.invoke({})
        assert "当前时间" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
