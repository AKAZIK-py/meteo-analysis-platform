import os
from typing import Dict, Optional, List
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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
        ('t2m', lambda x: (x > temp_range[0] + 273.15) & (x < temp_range[1] + 273.15)),  # ERA5温度单位为K
        ('u10', lambda x: np.abs(x) < wind_max),
        ('v10', lambda x: np.abs(x) < wind_max)
    ]:
        if var in ds_clean:
            ds_clean[var] = ds_clean[var].where(condition(ds_clean[var]))
            ds_clean[var] = ds_clean[var].interpolate_na(dim='valid_time', method='linear', fill_value="extrapolate")
    logger.info("数据质量控制完成")
    return ds_clean

def compute_indicators(ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """计算关键气象指标。由于ERA5单层数据无压力层，仅计算风速。

    Args:
        ds (xr.Dataset): 输入数据集。

    Returns:
        Dict[str, xr.DataArray]: 包含计算结果的字典。
    """
    indicators = {}
    try:
        if 'u10' in ds and 'v10' in ds:
            wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
            indicators['wind_speed'] = wind_speed * units('m/s')
        logger.info("气象指标计算完成")
    except Exception as e:
        logger.error(f"气象指标计算失败：{e}")
    return indicators

def plot_weather_map(ds: xr.Dataset, output_path: str = "weather_map.png") -> None:
    """使用Cartopy绘制温度等值线图。

    Args:
        ds (xr.Dataset): 输入数据集。
        output_path (str): 输出图片路径。
    """
    if 't2m' not in ds or 'latitude' not in ds or 'longitude' not in ds:
        logger.error("缺少绘制天气图所需变量")
        return

    temp = ds['t2m'].isel(valid_time=0) - 273.15  # 转换为摄氏度，使用 valid_time
    lats = ds['latitude']
    lons = ds['longitude']

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    contour = ax.contourf(lons, lats, temp, 60, transform=ccrs.PlateCarree(), cmap='coolwarm')
    plt.colorbar(contour, ax=ax, orientation='vertical', label='Temperature (°C)')
    plt.title("2m Temperature Contour Map")
    plt.savefig(output_path)
    plt.close(fig)
    logger.info(f"天气图已保存至 {output_path}")

def create_dash_app(ds: xr.Dataset) -> Optional[dash.Dash]:
    """创建交互式可视化模块，展示风玫瑰图和温度热力图。

    Args:
        ds (xr.Dataset): 输入数据集。

    Returns:
        Optional[dash.Dash]: Dash应用实例，若失败则返回None。
    """
    required_vars = ['u10', 'v10', 't2m', 'valid_time']
    if not all(var in ds.variables or var in ds.dims for var in required_vars):
        missing_vars = [var for var in required_vars if var not in ds.variables and var not in ds.dims]
        logger.error(f"缺少生成交互式图表所需变量：{missing_vars}")
        return None

    ds = ds.copy()
    ds['wind_speed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
    ds['wind_direction'] = (np.arctan2(ds['v10'], ds['u10']) * 180 / np.pi) % 360
    ds['t2m'] = ds['t2m'] - 273.15  # 转换为摄氏度

    df = ds.to_dataframe().reset_index()

    wind_rose_fig = px.bar_polar(df, r="wind_speed", theta="wind_direction",
                                 color="wind_speed", template="plotly_dark",
                                 title="Wind Rose Diagram")
    heatmap_fig = px.density_heatmap(df, x="valid_time", y="t2m", nbinsx=50, nbinsy=50,
                                     title="2m Temperature Heatmap")

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("ERA5 Data Analysis Dashboard"),
        dcc.Graph(id="wind-rose", figure=wind_rose_fig),
        dcc.Graph(id="temp-heatmap", figure=heatmap_fig)
    ])
    return app

def generate_report(summary: Dict[str, str], figures: List[str], output_path: str = "report.md") -> None:
    """生成Markdown格式报告。

    Args:
        summary (Dict[str, str]): 数据摘要。
        figures (List[str]): 图表路径列表。
        output_path (str): 输出文件路径。
    """
    template_str = """
# ERA5 数据分析报告

## 数据摘要
{% for key, value in summary.items() %}
- **{{ key }}**: {{ value }}
{% endfor %}

## 图表
{% for fig in figures %}
![图表]({{ fig }})
{% endfor %}

生成时间：{{ timestamp }}
"""
    template = jinja2.Template(template_str)
    report_content = template.render(summary=summary, figures=figures, timestamp=pd.Timestamp.now())

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    logger.info(f"报告已生成：{output_path}")

def main(data_file: str = "era5_sample.nc") -> None:
    """主函数：执行ERA5数据分析。

    Args:
        data_file (str, optional): 数据文件路径，默认"era5_sample.nc"。
    """
    if not os.path.exists(data_file):
        logger.error(f"数据文件 {data_file} 不存在，请先下载ERA5数据")
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

