# -*- coding: utf-8 -*-
import os
from typing import Dict, Optional, List
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from metpy.calc import wind_shear, geopotential_to_height
from metpy.units import units
import plotly.express as px
import dash
from dash import dcc, html
import jinja2
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> Optional[xr.Dataset]:
    """加载NetCDF格式的气象数据文件。

    Args:
        file_path (str): NetCDF文件路径。

    Returns:
        Optional[xr.Dataset]: 加载的数据集，若失败则返回None。
    """
    try:
        ds = xr.open_dataset(file_path)
        logger.info(f"成功加载数据文件：{file_path}")
        return ds
    except Exception as e:
        logger.error(f"加载数据文件失败：{e}")
        return None

def quality_control(ds: xr.Dataset, temp_range: tuple = (-100, 60), wind_max: float = 100.0) -> xr.Dataset:
    """数据质量控制：剔除异常值并插值补全缺失数据。

    Args:
        ds (xr.Dataset): 输入数据集。
        temp_range (tuple, optional): 温度有效范围，默认(-100, 60)。
        wind_max (float, optional): 风速分量最大绝对值，默认100.0。

    Returns:
        xr.Dataset: 经过质量控制后的数据集。
    """
    ds_clean = ds.copy()
    for var, condition in [
        ('temperature', lambda x: (x > temp_range[0]) & (x < temp_range[1])),
        ('u_wind', lambda x: np.abs(x) < wind_max),
        ('v_wind', lambda x: np.abs(x) < wind_max)
    ]:
        if var in ds_clean:
            ds_clean[var] = ds_clean[var].where(condition(ds_clean[var]))
            ds_clean[var] = ds_clean[var].interpolate_na(dim='time', method='linear', fill_value="extrapolate")
    logger.info("数据质量控制完成")
    return ds_clean

def compute_indicators(ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """计算关键气象指标，如风切变和位势高度。

    Args:
        ds (xr.Dataset): 输入数据集。

    Returns:
        Dict[str, xr.DataArray]: 包含计算结果的字典。
    """
    indicators = {}
    try:
        if all(var in ds for var in ['u_wind', 'v_wind', 'pressure']):
            u_low, u_high = ds['u_wind'].isel(pressure=0).metpy.quantify(), ds['u_wind'].isel(pressure=-1).metpy.quantify()
            v_low, v_high = ds['v_wind'].isel(pressure=0).metpy.quantify(), ds['v_wind'].isel(pressure=-1).metpy.quantify()
            p_low, p_high = ds['pressure'].isel(pressure=0).metpy.quantify(), ds['pressure'].isel(pressure=-1).metpy.quantify()
            shear = np.sqrt((u_high - u_low)**2 + (v_high - v_low)**2) / (p_low - p_high)
            indicators['wind_shear'] = shear

        if 'geopotential' in ds:
            height = geopotential_to_height(ds['geopotential'].metpy.quantify())
            indicators['geopotential_height'] = height
        logger.info("气象指标计算完成")
    except Exception as e:
        logger.error(f"气象指标计算失败：{e}")
    return indicators

# 其他函数如 plot_weather_map, create_dash_app, generate_report 等可类似优化

def main(data_file: str = "data.nc") -> None:
    """主函数：执行气象数据分析流程。

    Args:
        data_file (str, optional): 数据文件路径，默认"data.nc"。
    """
    if not os.path.exists(data_file):
        logger.error(f"数据文件 {data_file} 不存在，程序退出。")
        return

    ds = load_data(data_file)
    if ds is None:
        return

    ds = quality_control(ds)
    indicators = compute_indicators(ds)
    summary = {k: str(float(v.mean())) if v.size > 1 else str(v.item()) for k, v in indicators.items()}

    weather_map_path = "weather_map.png"
    plot_weather_map(ds, weather_map_path)
    generate_report(summary, [weather_map_path])

    app = create_dash_app(ds)
    if app:
        app.run_server(debug=True)

if __name__ == "__main__":
    main()
