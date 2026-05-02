"""内置的经典因子表达式。"""

from .. import config


def get_all_factors() -> list[tuple[str, str]]:
    """返回所有经典因子，以 (名称, 表达式) 元组列表的形式。"""
    return list(config.CLASSIC_FACTORS.items())
