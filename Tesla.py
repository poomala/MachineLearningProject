{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8d6d2523",
   "metadata": {
    "papermill": {
     "duration": 0.018072,
     "end_time": "2023-12-31T15:11:28.907533",
     "exception": false,
     "start_time": "2023-12-31T15:11:28.889461",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Tesla Stock Forecasting\n",
    "\n",
    "This Kaggle repository encompasses notebooks and datasets for a comprehensive project focused on forecasting Tesla (TSLA) stock prices. The predictive models employed include a single-step LSTM for immediate forecasting and a multi-stacked LSTM for extended predictions, spanning multiple days into the future (equivalent to a month in business days). The documentation not only presents the practical implementation but also provides an insightful exploration of the intuitive and mathematical underpinnings of LSTM networks. Dive into the code and datasets to enhance your understanding of LSTM dynamics, stacked architectures, and multistep forecasting in the realm of financial time series analysis.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "75deb519",
   "metadata": {
    "papermill": {
     "duration": 0.017355,
     "end_time": "2023-12-31T15:11:28.943153",
     "exception": false,
     "start_time": "2023-12-31T15:11:28.925798",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "In this notebook, we will explore the application of Long Short-Term Memory (LSTM) networks, a special kind of Recurrent Neural Network (RNN), for forecasting Tesla's stock prices.\n",
    "\n",
    "LSTMs, or Long Short-Term Memory networks, are specifically crafted to identify intricate patterns within sequential data. This makes them particularly effective for tasks involving time series data, like predicting stock prices, or processing linguistic sequences, such as phrases and sentences. Unlike standard feedforward neural networks, LSTMs have feedback connections that allow them to process not just single data points, but entire sequences of data. For a stock price prediction model, this allows the LSTM to consider the historical sequence of stock prices in making predictions about future prices.\n",
    "\n",
    "### LSTM Intuitive Explanation\n",
    "    \n",
    "An LSTM unit is designed to keep track of dependencies in input sequences. It does this through structures called gates which regulate the flow of information. These gates decide what to retain in memory (cell state) and what to forget based on the current input and previous state.\n",
    "\n",
    "- **Forget Gate $f_t$**: This gate decides which information from the cell state should be thrown away or kept. It looks at the previous output $h_{t-1}$ and the current input $x_t$ and applies a sigmoid function to output numbers between 0 and 1 for each number in the cell state $C_{t-1}$. A 1 represents “completely keep this” while a 0 represents “completely get rid of this.”\n",
    "\n",
    "- **Input Gate $i_t$**: This gate updates the cell state with new information. It first decides which values to update using a sigmoid function, and then creates a new vector of candidate values, $\\nu(C_t)$, that could be added to the state.\n",
    "\n",
    "- **Cell State $C_t$**: This is the memory of the LSTM. It is updated by forgetting the things deemed unnecessary and adding new candidate values scaled by their importance.\n",
    "\n",
    "- **Output Gate $o_t$**: Finally, the output is computed. The cell state is passed through `tanh` (to push the values to be between -1 and 1) and then multiplied by the output of the sigmoid gate, so that we only output the parts we decided to [1, 2].\n",
    "\n",
    "### LSTM Mathematical Explanation \n",
    "    \n",
    "The LSTM cell computes the following functions at each step `t` of the sequence:\n",
    "\n",
    "- **Forget Gate $f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$**: This gate decides which information is irrelevant from the cell state by looking at the previous output and the current input.\n",
    "\n",
    "- **Input Gate $i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$**: It decides which values will be updated in the cell state. \n",
    "\n",
    "- **Cell Input $\\tilde{C_t} = \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C)$**: It creates a vector of new candidate values that could be added to the cell state.\n",
    "\n",
    "- **Cell State Update $C_t = f_t * C_{t-1} + i_t * \\tilde{C}_t$**: The cell state is updated by forgetting things (multiplying by $f_t$) and adding new candidate values (multiplying by $i_t$).\n",
    "\n",
    "- **Output Gate $o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$**: It decides what the next hidden state ($h_t$) should be.\n",
    "\n",
    "- **Hidden State Output $h_t = o_t * \\tanh(C_t)$**: The output is based on the cell state but only certain parts are allowed to be outputted (controlled by $o_t$). \n",
    "\n",
    "Each of these steps involves a combination of weights ($W$), biases ($b$), and the previous hidden state and current input, which are trained during the learning process. The LSTM's ability to maintain a long-term memory is largely due to the cell state $C_t$ and how the gates interact to modify this memory.\n",
    "\n",
    "We will start by examining the Tesla stock data set, which includes daily trading information such as open, high, low, close prices, and trading volume[1, 2].\n",
    "    \n",
    "### Tesla Stock Dataset Overview\n",
    "    \n",
    "In this analysis, we will be delving into the stock data for Tesla, Inc. (TSLA). Tesla is renowned for its innovative approach to electric vehicles, energy storage, and solar panel manufacturing. As a company at the forefront of the transition to sustainable energy, Tesla's stock market performance is of significant interest to investors, market analysts, and enthusiasts of technology and sustainability. The dataset comprises daily trading information captured in several key financial metrics:\n",
    "\n",
    "- `open`: The price at which the stock started trading when the market opened on a given day.\n",
    "- `high`: The highest price at which the stock traded during the day.\n",
    "- `low`: The lowest price at which the stock traded during the day.\n",
    "- `close`: The last price at which the stock traded during the day. This is the figure most commonly reported in the financial news.\n",
    "- `volume`: The number of shares or contracts traded in a security or an entire market during a given period. It is a measure of the total demand for and supply of the stock.\n",
    "- `dividends`: The distribution of reward from a portion of the company's earnings and is paid to a class of its shareholders.\n",
    "- `stock splits`: An action taken by a company to divide its existing shares into multiple shares to boost the liquidity of the shares. Although the number of shares outstanding increases by a specific multiple, the total dollar value of the shares remains the same compared to pre-split amounts, because the split does not add any real value.\n",
    "\n",
    "The historical stock data for Tesla provides insights into the company's stock performance over time. By analyzing patterns in this data, we can attempt to forecast future stock prices using machine learning techniques such as Long Short-Term Memory (LSTM) networks."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "874ee22d",
   "metadata": {
    "papermill": {
     "duration": 0.020276,
     "end_time": "2023-12-31T15:11:28.984697",
     "exception": false,
     "start_time": "2023-12-31T15:11:28.964421",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Background-Imports and Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "042ff9bb",
   "metadata": {
    "_kg_hide-input": true,
    "_kg_hide-output": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:11:29.026149Z",
     "iopub.status.busy": "2023-12-31T15:11:29.025609Z",
     "iopub.status.idle": "2023-12-31T15:12:14.682514Z",
     "shell.execute_reply": "2023-12-31T15:12:14.680554Z"
    },
    "papermill": {
     "duration": 45.681054,
     "end_time": "2023-12-31T15:12:14.685895",
     "exception": false,
     "start_time": "2023-12-31T15:11:29.004841",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "!pip install yfinance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2aebf3f6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:14.725945Z",
     "iopub.status.busy": "2023-12-31T15:12:14.725476Z",
     "iopub.status.idle": "2023-12-31T15:12:35.953277Z",
     "shell.execute_reply": "2023-12-31T15:12:35.951852Z"
    },
    "papermill": {
     "duration": 21.251632,
     "end_time": "2023-12-31T15:12:35.955952",
     "exception": false,
     "start_time": "2023-12-31T15:12:14.704320",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Data Imports\n",
    "import yfinance as yf\n",
    "import pandas as pd \n",
    "import numpy as np \n",
    "import random\n",
    "\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from pandas.tseries.offsets import BDay\n",
    "\n",
    "# Visualization Imports\n",
    "from plotly.subplots import make_subplots\n",
    "import plotly.graph_objects as go\n",
    "import plotly.express as px\n",
    "import plotly.io as pio\n",
    "import scipy.stats as stats\n",
    "\n",
    "# Neural Network Imports\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras import models\n",
    "from tensorflow.keras import regularizers\n",
    "from tensorflow.keras.layers import LSTM\n",
    "from tensorflow.keras.layers import Dense\n",
    "from tensorflow.keras.layers import Dropout\n",
    "from tensorflow.keras.layers import Bidirectional\n",
    "from tensorflow.keras.callbacks import ModelCheckpoint\n",
    "\n",
    "# Setting seed\n",
    "SEED = 0\n",
    "random.seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "tf.random.set_seed(SEED)\n",
    "\n",
    "# Visualization Configurations\n",
    "pio.templates.default = \"plotly_dark\"\n",
    "%config InlineBackend.figure_format = 'retina'"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "baa3adaf",
   "metadata": {
    "papermill": {
     "duration": 0.019466,
     "end_time": "2023-12-31T15:12:35.995027",
     "exception": false,
     "start_time": "2023-12-31T15:12:35.975561",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Tesla Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f527e3a0",
   "metadata": {
    "_kg_hide-input": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:36.035714Z",
     "iopub.status.busy": "2023-12-31T15:12:36.034611Z",
     "iopub.status.idle": "2023-12-31T15:12:36.043399Z",
     "shell.execute_reply": "2023-12-31T15:12:36.041992Z"
    },
    "papermill": {
     "duration": 0.03227,
     "end_time": "2023-12-31T15:12:36.046124",
     "exception": false,
     "start_time": "2023-12-31T15:12:36.013854",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def show(data: pd.DataFrame):\n",
    "    df = data.copy()\n",
    "    df = df.style.format(precision=3)\n",
    "    df = df.background_gradient(cmap='Reds', axis=0)\n",
    "    display(df)\n",
    "\n",
    "def highlight_half(data: pd.DataFrame, axis=1, precision=3):\n",
    "    \n",
    "    s = data.shape[1] if axis else data.shape[0]\n",
    "    data_style = data.style.format(precision=precision)\n",
    "\n",
    "    def apply_style(val):\n",
    "        style1 = 'background-color: red; color: white'\n",
    "        style2 = 'background-color: blue; color: white'\n",
    "        return [style1 if x < s//2 else style2 for x in range(s)]\n",
    "\n",
    "    display(data_style.apply(apply_style, axis=axis))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "580cbaaa",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:36.086313Z",
     "iopub.status.busy": "2023-12-31T15:12:36.085877Z",
     "iopub.status.idle": "2023-12-31T15:12:53.771797Z",
     "shell.execute_reply": "2023-12-31T15:12:53.770564Z"
    },
    "papermill": {
     "duration": 17.709221,
     "end_time": "2023-12-31T15:12:53.774673",
     "exception": false,
     "start_time": "2023-12-31T15:12:36.065452",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Target stock & columns for modeling\n",
    "SYMBOL = \"TSLA\"\n",
    "columns = ['open', 'high', 'low', 'close', 'volume']\n",
    "\n",
    "# Getting Tesla (TSLA) stock data\n",
    "ticker = yf.Ticker(SYMBOL)\n",
    "\n",
    "# End stock dates\n",
    "end_date = \"2023-12-01\"\n",
    "\n",
    "# Pulling stock data \n",
    "df = ticker.history(start=\"2017-01-01\", end=end_date)\n",
    "df.columns = df.columns.str.lower()\n",
    "\n",
    "# Showing data\n",
    "show(df.tail())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "745d79f1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:53.816525Z",
     "iopub.status.busy": "2023-12-31T15:12:53.815424Z",
     "iopub.status.idle": "2023-12-31T15:12:53.837454Z",
     "shell.execute_reply": "2023-12-31T15:12:53.835669Z"
    },
    "papermill": {
     "duration": 0.045606,
     "end_time": "2023-12-31T15:12:53.840878",
     "exception": false,
     "start_time": "2023-12-31T15:12:53.795272",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Data info\n",
    "print('Data Info:')\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37c6309a",
   "metadata": {
    "papermill": {
     "duration": 0.02053,
     "end_time": "2023-12-31T15:12:53.880925",
     "exception": false,
     "start_time": "2023-12-31T15:12:53.860395",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Data Visualization \n",
    "### Candle Stick Plots \n",
    "    \n",
    "\n",
    "Candlestick charts are a visual tool for market analysis, used to describe price movements of a security, derivative, or currency. Each \"candlestick\" typically represents one day of trading and is composed of a body and wicks.\n",
    "- **Body**: The wider section of the candlestick which indicates the opening and closing prices. If the body is filled or dark, the security closed lower than it opened. If the body is empty or light, it closed higher than it opened.\n",
    "- **Wicks**: Lines that extend from the top and bottom of the body representing the high and low prices during the period.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc6e1c46",
   "metadata": {
    "_kg_hide-input": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:53.927374Z",
     "iopub.status.busy": "2023-12-31T15:12:53.926659Z",
     "iopub.status.idle": "2023-12-31T15:12:53.956413Z",
     "shell.execute_reply": "2023-12-31T15:12:53.954914Z"
    },
    "papermill": {
     "duration": 0.058751,
     "end_time": "2023-12-31T15:12:53.960761",
     "exception": false,
     "start_time": "2023-12-31T15:12:53.902010",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def plot_candlestick(stock_df, name='', rolling_avg=None, fig_size=(1100, 700)):\n",
    "\n",
    "    # Copy df to avoid modifying the original data\n",
    "    stock_data = stock_df.copy()\n",
    "    \n",
    "    # Creating plot\n",
    "    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,\n",
    "        close=stock_data['close'], open=stock_data['open'], high=stock_data['high'], low=stock_data['low'], \n",
    "        name=\"Candlesticks\", increasing_line_color='green', decreasing_line_color='red', line=dict(width=1)\n",
    "                                        )])\n",
    "    # Rolling averages if specified\n",
    "    if rolling_avg:\n",
    "        colors = ['rgba(0, 255, 255, 0.5)',   # cyan\n",
    "                  'rgba(255, 255, 0, 0.5)',   # yellow\n",
    "                  'rgba(255, 165, 0, 0.5)',   # orange\n",
    "                  'rgba(255, 105, 180, 0.5)', # pink\n",
    "                  'rgba(165, 42, 42, 0.5)',   # brown\n",
    "                  'rgba(128, 128, 128, 0.5)', # gray\n",
    "                  'rgba(128, 128, 0, 0.5)',   # olive\n",
    "                  'rgba(0, 0, 255, 0.5)']     # blue\n",
    "        \n",
    "        for i, avg in enumerate(rolling_avg):\n",
    "            color = colors[i % len(colors)]\n",
    "            ma_column = f'{avg}-day MA'\n",
    "            stock_data[ma_column] = stock_data['close'].rolling(window=avg).mean()\n",
    "\n",
    "            # Moving average trace\n",
    "            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[ma_column],\n",
    "                    mode='lines', name=f'{avg}-day Moving Average', line=dict(color=color)))\n",
    "\n",
    "    # Layout updates\n",
    "    fig.update_layout(title=f\"{name} Stock Price - Candlestick Chart\",\n",
    "                      xaxis_title=\"Date\", yaxis_title=\"Price\",\n",
    "                      width=fig_size[0], height=fig_size[1],\n",
    "                      xaxis=dict(\n",
    "                          rangeselector=dict(\n",
    "                              buttons=list([\n",
    "                                  dict(count=14, label=\"2w\", step=\"day\", stepmode=\"backward\"),\n",
    "                                  dict(count=1, label=\"1m\", step=\"month\", stepmode=\"backward\"),\n",
    "                                  dict(count=3, label=\"3m\", step=\"month\", stepmode=\"backward\"),\n",
    "                                  dict(count=6, label=\"6m\", step=\"month\", stepmode=\"backward\"),\n",
    "                                  dict(count=1, label=\"YTD\", step=\"year\", stepmode=\"todate\"),\n",
    "                                  dict(count=1, label=\"1y\", step=\"year\", stepmode=\"backward\"),\n",
    "                                  dict(count=2, label=\"2y\", step=\"year\", stepmode=\"backward\"),\n",
    "                                  dict(count=3, label=\"3y\", step=\"year\", stepmode=\"backward\"),\n",
    "                                  dict(count=5, label=\"5y\", step=\"year\", stepmode=\"backward\"),\n",
    "                                  dict(step=\"all\")]),\n",
    "                              bgcolor='pink',\n",
    "                              font=dict(color='black'),\n",
    "                              activecolor='lightgreen'))\n",
    "                     )\n",
    "    fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "31b73ec8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:54.089955Z",
     "iopub.status.busy": "2023-12-31T15:12:54.089494Z",
     "iopub.status.idle": "2023-12-31T15:12:54.795398Z",
     "shell.execute_reply": "2023-12-31T15:12:54.794194Z"
    },
    "papermill": {
     "duration": 0.736777,
     "end_time": "2023-12-31T15:12:54.803863",
     "exception": false,
     "start_time": "2023-12-31T15:12:54.067086",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# General Tesla stocks\n",
    "plot_candlestick(df, name=SYMBOL)\n",
    "\n",
    "# With Moving averages\n",
    "plot_candlestick(df, name=SYMBOL, rolling_avg=[20, 50, 200])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c992f7d3",
   "metadata": {
    "papermill": {
     "duration": 0.028136,
     "end_time": "2023-12-31T15:12:54.860464",
     "exception": false,
     "start_time": "2023-12-31T15:12:54.832328",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Tesla Stock Splits  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a3572f59",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:54.921197Z",
     "iopub.status.busy": "2023-12-31T15:12:54.920211Z",
     "iopub.status.idle": "2023-12-31T15:12:55.353536Z",
     "shell.execute_reply": "2023-12-31T15:12:55.352150Z"
    },
    "papermill": {
     "duration": 0.467589,
     "end_time": "2023-12-31T15:12:55.357198",
     "exception": false,
     "start_time": "2023-12-31T15:12:54.889609",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Plotting stock splits\n",
    "fig = px.line(x=df.index, y=df['stock splits'], title=f'{SYMBOL} Stock Splits Over Time')\n",
    "fig.update_layout(width=1100, height=500)\n",
    "fig.update_traces(line=dict(color='cyan', width=3))\n",
    "fig.update_xaxes(title_text='Date')\n",
    "fig.update_yaxes(title_text='Stock Splits')\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bf3744b7",
   "metadata": {
    "papermill": {
     "duration": 0.028482,
     "end_time": "2023-12-31T15:12:55.416881",
     "exception": false,
     "start_time": "2023-12-31T15:12:55.388399",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Percent Change in Stock Prices \n",
    "    \n",
    "The percentage change in stock prices is a measure used to express the change in price over time as a proportion of the previous price. This metric is useful for comparing the performance of a stock across different time frames or against other stocks. The formula to calculate the daily percentage change is:\n",
    "\n",
    "$$\\text{Percentage Change} = \\left( \\frac{\\text{Current Price} - \\text{Previous Price}}{\\text{Previous Price}} \\right) \\times 100\n",
    "$$\n",
    " **Current Price** is the price of the stock at the end of the current period (e.g., end of the day).\n",
    " **Previous Price** is the price of the stock at the end of the previous period.\n",
    "\n",
    "For the first period in a time series data set (e.g., the first day of available stock data), the percentage change is not defined as there is no previous price to compare to. In such cases, it is common to set the percentage change to zero or to omit the value. When analyzing stock data over a longer period, such as a month or year, the percentage change is calculated using the price at the beginning and the end of the period. This metric is commonly used in financial analysis to assess the volatility and performance of stocks. It is also a key indicator for investors making decisions about buying or selling securities.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a7314ae7",
   "metadata": {
    "_kg_hide-input": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:55.477876Z",
     "iopub.status.busy": "2023-12-31T15:12:55.476955Z",
     "iopub.status.idle": "2023-12-31T15:12:55.947414Z",
     "shell.execute_reply": "2023-12-31T15:12:55.945696Z"
    },
    "papermill": {
     "duration": 0.505862,
     "end_time": "2023-12-31T15:12:55.952381",
     "exception": false,
     "start_time": "2023-12-31T15:12:55.446519",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Creating subplot\n",
    "fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3],\n",
    "                    vertical_spacing=0.1, horizontal_spacing=0.05,\n",
    "                    subplot_titles=(f\"{SYMBOL} - Percent Change over Time\", f\"{SYMBOL} Percent Change - Histogram\",\n",
    "                                    f\"{SYMBOL} - Stock Volume over Time\", f\"{SYMBOL} Stock Volume - Histogram\",))\n",
    "# Percent Change Plot\n",
    "percent_change = df['close'].pct_change() * 100\n",
    "fig.add_trace(go.Scatter(x=df.index, y=percent_change, name='Percent Change', marker_color='darkorchid'), row=1, col=1)\n",
    "fig.add_trace(go.Histogram(x=percent_change, nbinsx=50, name='Percent Change', marker_color='darkorchid'),  row=1, col=2)\n",
    "fig.add_annotation(text=f\"Mean: {percent_change.mean():.2f}%<br>Std Dev: {percent_change.std():.2f}%\",\n",
    "                   xref='x2', yref='y2', x=percent_change.mean(), y=5, showarrow=True)\n",
    "# Volume Plot\n",
    "fig.add_trace(go.Scatter(x=df.index, y=df['volume'], name='Volume', marker_color='darkcyan'), row=2, col=1)\n",
    "fig.add_trace(go.Histogram(x=df['volume'], nbinsx=50, name='Daily Volume', marker_color='darkcyan'),  row=2, col=2)\n",
    "fig.add_annotation(text=f\"Mean: {df['volume'].mean():.2f}<br>Std Dev: {df['volume'].std():.2f}\",\n",
    "                   xref='x4', yref='y4', x=df['volume'].mean(), y=5, showarrow=True)\n",
    "\n",
    "fig.update_layout(height=700, width=1100)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6a404acc",
   "metadata": {
    "papermill": {
     "duration": 0.034989,
     "end_time": "2023-12-31T15:12:56.023784",
     "exception": false,
     "start_time": "2023-12-31T15:12:55.988795",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "- **Tesla Percent Change over Time:** plot displays the daily percentage change of Tesla's stock price, revealing the volatility over the observed period. The fluctuations are captured by sharp spikes and dips, indicating days with significant price movements. This could be reflective of market reactions to news events, earnings reports, or broader economic conditions.\n",
    "\n",
    "- **Tesla Percent Change - Histogram:** plot on the top right presents the distribution of these daily percentage changes. Most changes cluster around the mean, suggesting a normal distribution of returns, which is typical for stock prices over time. The mean close to zero implies stable average growth, while the standard deviation indicates the extent of variation from the average.\n",
    "\n",
    "- **Tesla Stock Volume over Time:** showing the traded volume of Tesla's stock. Peaks in this plot could correspond to specific events or the release of significant news affecting investor sentiment and trading behavior.\n",
    "\n",
    "- **Tesla Stock Volume - Histogram:** illustrates the distribution of trading volume, indicating how often certain volumes occur. The mean and standard deviation provide a summary of the typical volume and its variability. The concentration of data on the lower end suggests that high-volume days are less frequent but can be associated with key market or company-specific events."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55d3d2d1",
   "metadata": {
    "papermill": {
     "duration": 0.034514,
     "end_time": "2023-12-31T15:12:56.094604",
     "exception": false,
     "start_time": "2023-12-31T15:12:56.060090",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## LSTM Data Prep\n",
    "\n",
    "In machine learning, it's often crucial, especialy when using neural networks, to normalize data before feeding it into a model. This process adjusts values measured on different scales to a notionally common scale, often prior to averaging. Here, we use the `MinMaxScaler` from the `sklearn.preprocessing` package, which scales each feature by its maximum and minimum values. This scaler transforms each value `v` in a feature column to `v'` in the range [0, 1] using the following formula:\n",
    "$$v' = \\frac{v - \\text{min}(v)}{\\text{max}(v) - \\text{min}(v)}$$\n",
    "- `v` is the original value.\n",
    "- `min(v)` is the minimum value in the feature column.\n",
    "- `max(v)` is the maximum value in the feature column.\n",
    "The columns `['open', 'high', 'low', 'close', 'volume']` from the Tesla stock dataset are normalized, which includes the opening, high, low, and closing prices along with the trading volume. Normalizing these features allows for a more stable and faster convergence during the training of neural networks, like the LSTM model we'll be using for stock price forecasting.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b7248f6a",
   "metadata": {
    "_kg_hide-input": false,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:56.170456Z",
     "iopub.status.busy": "2023-12-31T15:12:56.168276Z",
     "iopub.status.idle": "2023-12-31T15:12:56.189161Z",
     "shell.execute_reply": "2023-12-31T15:12:56.188198Z"
    },
    "papermill": {
     "duration": 0.062534,
     "end_time": "2023-12-31T15:12:56.191819",
     "exception": false,
     "start_time": "2023-12-31T15:12:56.129285",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "class RNNFormater:\n",
    "    \n",
    "    def __init__(self, data: pd.DataFrame, mapping_steps=10):\n",
    "        \"\"\"\n",
    "        Initialize the RNNFormater with a DataFrame and steps to map for data.\n",
    "        \n",
    "        Args:\n",
    "            data (pd.DataFrame): Input DataFrame containing time series data.\n",
    "            mapping_steps (int): Number of time steps for each input sequence to be mapped to output.\n",
    "        \"\"\"\n",
    "        # Storing data\n",
    "        self.df = data.copy()\n",
    "        self.data = self.df.values\n",
    "\n",
    "        # Scaler stored for usage later\n",
    "        self.scaler = MinMaxScaler()\n",
    "        self.normalized_data = self.scaler.fit_transform(self.data)\n",
    "        \n",
    "        self.time_steps = data.shape[0]\n",
    "        self.n_columns = data.shape[1]\n",
    "\n",
    "        # Number of mapping steps\n",
    "        self.mapping_steps = mapping_steps\n",
    "\n",
    "    def data_mapping(self):\n",
    "        \"\"\"\n",
    "        Maps a 2D array into a 3D array for RNNs input, with each sequence having mapping_steps time steps.\n",
    "    \n",
    "        Args:\n",
    "            mapping_steps (int): Number of time steps for each sequence.\n",
    "    \n",
    "        Returns:\n",
    "            np.array: A 3D array suitable for RNN inputs.\n",
    "        \"\"\"\n",
    "        mapping_steps = self.mapping_steps + 1\n",
    "        \n",
    "        mapping_iterations = self.time_steps - mapping_steps + 1\n",
    "        self.normalized_data_mapped = np.empty((mapping_iterations, mapping_steps, self.n_columns))\n",
    "        \n",
    "        for i in range(mapping_iterations):\n",
    "            self.normalized_data_mapped[i, :, :] = self.normalized_data[i:i + mapping_steps, :]\n",
    "        \n",
    "        return self.normalized_data_mapped\n",
    "    \n",
    "    def rnn_train_test_split(self, test_percent=0.1):\n",
    "        \"\"\"\n",
    "        Splits the 3D mapped data into training and testing sets for an RNN.\n",
    "        \n",
    "        Args:\n",
    "            test_percent (float): The fraction of data to be used for testing.\n",
    "        \n",
    "        Returns:\n",
    "            tuple: X_train, X_test, y_train, y_test\n",
    "        \"\"\"\n",
    "        self.test_size = int(np.round(self.normalized_data_mapped.shape[0] * test_percent))\n",
    "        self.train_size = self.normalized_data_mapped.shape[0] - self.test_size\n",
    "        \n",
    "        X_train = self.normalized_data_mapped[:self.train_size, :-1, :]\n",
    "        y_train = self.normalized_data_mapped[:self.train_size, -1, :]\n",
    "        \n",
    "        X_test = self.normalized_data_mapped[self.train_size:, :-1, :]\n",
    "        y_test = self.normalized_data_mapped[self.train_size:, -1, :]\n",
    "        \n",
    "        return X_train, X_test, y_train, y_test  \n",
    "\n",
    "    def forecast_n_steps(self, model, data: pd.DataFrame, n_forecast_steps=30):\n",
    "        \"\"\"\n",
    "        Forecast multiple steps ahead using the LSTM model.\n",
    "    \n",
    "        Args:\n",
    "            model (tf.keras.Model): Trained LSTM model for prediction.\n",
    "            data (pd.DataFrame): Input DataFrame containing the latest time series data.\n",
    "            n_forecast_steps (int): Number of future steps to forecast.\n",
    "    \n",
    "        Returns:\n",
    "            np.array: Forecasted values for n_forecast_steps.\n",
    "        \"\"\"\n",
    "        # Scaling the latest 'mapping_steps' data for mapping\n",
    "        last_steps = self.scaler.transform(data.values)[-self.mapping_steps:]\n",
    "    \n",
    "        # Initialize normalized_data_mapped array\n",
    "        normalized_data_mapped = np.empty((n_forecast_steps, self.mapping_steps, self.n_columns))\n",
    "    \n",
    "        # Initialize predictions array\n",
    "        predictions = np.empty((n_forecast_steps, self.n_columns))\n",
    "    \n",
    "        # Predict the first step\n",
    "        normalized_data_mapped[0, :, :] = last_steps\n",
    "        predictions[0, :] = model.predict(\n",
    "            normalized_data_mapped[0, :, :].reshape(1, self.mapping_steps, self.n_columns),\n",
    "            verbose=False\n",
    "        )\n",
    "        # Generate predictions and update normalized_data_mapped for each subsequent step\n",
    "        for i in range(1, n_forecast_steps):\n",
    "            # Shift the window and insert new prediction at end\n",
    "            normalized_data_mapped[i, :-1, :] = normalized_data_mapped[i - 1, 1:, :]\n",
    "            normalized_data_mapped[i, -1, :] = predictions[i - 1, :]\n",
    "    \n",
    "            # Predicting next step\n",
    "            norm_data = normalized_data_mapped[i, :, :].reshape(1, self.mapping_steps, self.n_columns)\n",
    "            predictions[i, :] = model.predict(norm_data, verbose=False)\n",
    "    \n",
    "        # Inverse transform the predictions to original scale\n",
    "        predictions = self.scaler.inverse_transform(predictions)\n",
    "        return predictions\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e75f7667",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:56.270741Z",
     "iopub.status.busy": "2023-12-31T15:12:56.269760Z",
     "iopub.status.idle": "2023-12-31T15:12:56.286648Z",
     "shell.execute_reply": "2023-12-31T15:12:56.284823Z"
    },
    "papermill": {
     "duration": 0.059343,
     "end_time": "2023-12-31T15:12:56.290209",
     "exception": false,
     "start_time": "2023-12-31T15:12:56.230866",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Initializing class\n",
    "mapping_steps = 32 # ~ 1 months in buisness days\n",
    "rnn_formater = RNNFormater(df[columns], mapping_steps=mapping_steps)\n",
    "\n",
    "# Mapping steps\n",
    "norm_data_mapped = rnn_formater.data_mapping() # n_steps -> y\n",
    "# print(f'Mapped Normalized data step 0:\\n{norm_data_mapped[0]}')\n",
    "print(f'Normalized data shape: {norm_data_mapped[0].shape}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ef48e55",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:56.366699Z",
     "iopub.status.busy": "2023-12-31T15:12:56.366276Z",
     "iopub.status.idle": "2023-12-31T15:12:56.374815Z",
     "shell.execute_reply": "2023-12-31T15:12:56.372902Z"
    },
    "papermill": {
     "duration": 0.04918,
     "end_time": "2023-12-31T15:12:56.378446",
     "exception": false,
     "start_time": "2023-12-31T15:12:56.329266",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Train Test Split\n",
    "X_train, X_test, y_train, y_test = rnn_formater.rnn_train_test_split(test_percent=0.05)\n",
    "print(f'Number of time steps for test set: {rnn_formater.test_size}')\n",
    "\n",
    "print(f'X shape: {X_train.shape}')\n",
    "# print(X_train[0])\n",
    "\n",
    "print(f'y shape: {y_train.shape}')\n",
    "# print(y_train[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dfbba8b8",
   "metadata": {
    "papermill": {
     "duration": 0.036715,
     "end_time": "2023-12-31T15:12:56.453561",
     "exception": false,
     "start_time": "2023-12-31T15:12:56.416846",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "###  Vanilla LSTM Model Building"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "015c115e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:56.528152Z",
     "iopub.status.busy": "2023-12-31T15:12:56.527700Z",
     "iopub.status.idle": "2023-12-31T15:12:57.122386Z",
     "shell.execute_reply": "2023-12-31T15:12:57.121263Z"
    },
    "papermill": {
     "duration": 0.641243,
     "end_time": "2023-12-31T15:12:57.130208",
     "exception": false,
     "start_time": "2023-12-31T15:12:56.488965",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# For consistant results\n",
    "random.seed(0)\n",
    "np.random.seed(0)\n",
    "tf.random.set_seed(0)\n",
    "\n",
    "# Vanilla LSTM\n",
    "model = models.Sequential([\n",
    "    LSTM(units=80, input_shape=(mapping_steps, len(columns))),\n",
    "    Dropout(0.05),\n",
    "    Dense(units=len(columns))   \n",
    "])\n",
    "\n",
    "# Compiling model\n",
    "model.compile(optimizer='adam', loss='mae')\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4151d1ce",
   "metadata": {
    "_kg_hide-output": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:12:57.208065Z",
     "iopub.status.busy": "2023-12-31T15:12:57.207648Z",
     "iopub.status.idle": "2023-12-31T15:20:30.351151Z",
     "shell.execute_reply": "2023-12-31T15:20:30.349930Z"
    },
    "papermill": {
     "duration": 453.185852,
     "end_time": "2023-12-31T15:20:30.354222",
     "exception": false,
     "start_time": "2023-12-31T15:12:57.168370",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Callback to save model weights\n",
    "model_checkpoint = ModelCheckpoint('LSTM_Tesla_model.h5', monitor='val_loss', save_best_only=True)\n",
    "\n",
    "# Fitting the model\n",
    "history = model.fit(X_train, y_train, \n",
    "                    batch_size=256, \n",
    "                    epochs=1_000, \n",
    "                    validation_data=(X_test, y_test), \n",
    "                    callbacks=[model_checkpoint],\n",
    "                    shuffle=False,\n",
    "                    verbose=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0da4cc8b",
   "metadata": {
    "_kg_hide-input": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:30.431033Z",
     "iopub.status.busy": "2023-12-31T15:20:30.430618Z",
     "iopub.status.idle": "2023-12-31T15:20:30.445035Z",
     "shell.execute_reply": "2023-12-31T15:20:30.443308Z"
    },
    "papermill": {
     "duration": 0.05604,
     "end_time": "2023-12-31T15:20:30.447981",
     "exception": false,
     "start_time": "2023-12-31T15:20:30.391941",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def plot_training_history(history, plot_title='Training Performance', plot_legends=None, color0=0):\n",
    "    \"\"\"\n",
    "    Plots the training history of a model using Plotly.\n",
    "\n",
    "    Args:\n",
    "        history (dict): A dictionary containing the training history metrics.\n",
    "        plot_title (str): Title of the plot.\n",
    "        plot_legends (list): List of legends for the plot. If None, it uses the keys from the history dictionary.\n",
    "\n",
    "    Returns:\n",
    "        None: Displays the plot.\n",
    "    \"\"\"\n",
    "    # Extracting metrics from the history object\n",
    "    epochs = np.arange(1, len(next(iter(history.values()))) + 1)\n",
    "    colors = ['blue', 'gold', 'violet', 'lime', 'blue', 'pink', 'yellow']\n",
    "    data = []\n",
    "\n",
    "    # If no legends are provided, use keys from the history\n",
    "    if not plot_legends:\n",
    "        plot_legends = list(history.keys())\n",
    "\n",
    "    # Prepare data for each metric in the history\n",
    "    for i, (key, legend) in enumerate(zip(history.keys(), plot_legends)):\n",
    "        color_index = i % len(colors) + color0\n",
    "        data.append(go.Scatter(x=epochs, y=history[key], mode='lines+markers', name=legend, line=dict(color=colors[color_index])))\n",
    "\n",
    "    # Add error for minimum epoch value\n",
    "    min_epoch = np.argmin(history['val_loss']) + 1 \n",
    "    loss_str = f\"Train Loss: {history['loss'][min_epoch-1]:.3e}<br>Test Loss: {history['val_loss'][min_epoch - 1]:.3e}\"\n",
    "\n",
    "    # Creating the layout\n",
    "    layout = go.Layout(title=plot_title, xaxis=dict(title='Epochs'), yaxis=dict(title='Value'), width=1100, height=600)\n",
    "    fig = go.Figure(data=data, layout=layout)\n",
    "\n",
    "    # Annotate the minimum loss with an arrow\n",
    "    fig.add_annotation(\n",
    "        go.layout.Annotation(\n",
    "            x=min_epoch,\n",
    "            y=history['loss'][min_epoch - 1],\n",
    "            xref=\"x\",\n",
    "            yref=\"y\",\n",
    "            text=loss_str,\n",
    "            showarrow=True,\n",
    "            arrowhead=7,\n",
    "            arrowcolor='green',\n",
    "            arrowsize=2,\n",
    "            bordercolor='green',\n",
    "            borderwidth=2,\n",
    "            ax=0,\n",
    "            ay=-40\n",
    "        )\n",
    "    )\n",
    "    fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd78bcee",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:30.525355Z",
     "iopub.status.busy": "2023-12-31T15:20:30.524601Z",
     "iopub.status.idle": "2023-12-31T15:20:30.568611Z",
     "shell.execute_reply": "2023-12-31T15:20:30.567362Z"
    },
    "papermill": {
     "duration": 0.084902,
     "end_time": "2023-12-31T15:20:30.571190",
     "exception": false,
     "start_time": "2023-12-31T15:20:30.486288",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Plotting LSTM model loss\n",
    "plot_training_history(history.history, plot_title='LSTM Model Loss')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d3ca690",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:30.650635Z",
     "iopub.status.busy": "2023-12-31T15:20:30.650173Z",
     "iopub.status.idle": "2023-12-31T15:20:31.621312Z",
     "shell.execute_reply": "2023-12-31T15:20:31.620407Z"
    },
    "papermill": {
     "duration": 1.014742,
     "end_time": "2023-12-31T15:20:31.624677",
     "exception": false,
     "start_time": "2023-12-31T15:20:30.609935",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Loading best wieghts during training\n",
    "model = models.load_model(f'LSTM_Tesla_model.h5')\n",
    "\n",
    "# Predicting\n",
    "predictions = model.predict(X_test, verbose=False)\n",
    "predictions = rnn_formater.scaler.inverse_transform(predictions)\n",
    "\n",
    "# Showing predictions and data\n",
    "index_1 = y_test.shape[0]\n",
    "df_y_test = df[columns].iloc[-index_1:]\n",
    "df_predictions = pd.DataFrame(predictions, index=df_y_test.index, columns=[f'pred_{col}' for col in columns])\n",
    "df_test_pred = pd.concat([df_y_test, df_predictions], axis=1)\n",
    "                         \n",
    "# Shwoing outputs\n",
    "highlight_half(df_test_pred.tail(), axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8b74a7a8",
   "metadata": {
    "papermill": {
     "duration": 0.036541,
     "end_time": "2023-12-31T15:20:31.701303",
     "exception": false,
     "start_time": "2023-12-31T15:20:31.664762",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### STM Resdiual Analysis Building "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e8e8ceea",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:31.776521Z",
     "iopub.status.busy": "2023-12-31T15:20:31.776089Z",
     "iopub.status.idle": "2023-12-31T15:20:31.786286Z",
     "shell.execute_reply": "2023-12-31T15:20:31.784811Z"
    },
    "papermill": {
     "duration": 0.051768,
     "end_time": "2023-12-31T15:20:31.789818",
     "exception": false,
     "start_time": "2023-12-31T15:20:31.738050",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Error dataframe\n",
    "df_error = pd.DataFrame(df_predictions.values - df_y_test.values, index=df.index[-index_1:], columns=[f'error_{col}' for col in columns])\n",
    "print('RMSE Per Column')\n",
    "print((df_error**2).mean()**(1/2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6fd79377",
   "metadata": {
    "_kg_hide-input": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:31.866918Z",
     "iopub.status.busy": "2023-12-31T15:20:31.866408Z",
     "iopub.status.idle": "2023-12-31T15:20:31.891006Z",
     "shell.execute_reply": "2023-12-31T15:20:31.889665Z"
    },
    "papermill": {
     "duration": 0.066402,
     "end_time": "2023-12-31T15:20:31.893505",
     "exception": false,
     "start_time": "2023-12-31T15:20:31.827103",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def plotly_residual_analysis(df, title_add=''):\n",
    "    \"\"\"\n",
    "    Perform residual analysis for multiple features in a DataFrame.\n",
    "    The DataFrame should contain actual and predicted columns for each feature.\n",
    "    \n",
    "    Args:\n",
    "        df (pd.DataFrame): DataFrame containing actual and predicted columns.\n",
    "        title_add (str, optional): Additional title for the subplots.\n",
    "    \"\"\"\n",
    "    # Number of columns\n",
    "    columns = [col for col in df.columns if not col.startswith('pred_')]\n",
    "    num_features = len(columns)\n",
    "\n",
    "    # Color per column\n",
    "    colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow']\n",
    "\n",
    "    # Subplots per columns\n",
    "    fig = make_subplots(rows=num_features, cols=4, vertical_spacing=0.035, horizontal_spacing=0.035,\n",
    "                        subplot_titles=(\"Histogram\", \"QQ-Normal Plot\", \"Residuals vs. Predicted Values\", \"Residuals vs Index\"))\n",
    "\n",
    "    for i, col in enumerate(columns):\n",
    "        actual = df[col]\n",
    "        predicted = df[f'pred_{col}']\n",
    "        residuals = actual - predicted\n",
    "        mean_residuals = np.mean(residuals)\n",
    "        sd_residuals = np.std(residuals)\n",
    "        rmse = np.sqrt(np.mean(residuals**2))\n",
    "        index = df.index\n",
    "\n",
    "        # Assign color for each feature\n",
    "        color = colors[i % len(colors)]\n",
    "\n",
    "        # Histogram of residuals\n",
    "        fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name=f'{col.title()} Residuals', marker_color=color), row=i+1, col=1)\n",
    "        # Add lines for mean and standard deviation\n",
    "        fig.add_vline(x=mean_residuals, line=dict(color='black', width=2), row=i+1, col=1)\n",
    "        fig.add_vline(x=mean_residuals + sd_residuals, line=dict(color='grey', width=2, dash='dash'), row=i+1, col=1)\n",
    "        fig.add_vline(x=mean_residuals - sd_residuals, line=dict(color='grey', width=2, dash='dash'), row=i+1, col=1)\n",
    "        fig.add_annotation(x=mean_residuals, y=5, text=f\"Mean: {mean_residuals:.2f}\", showarrow=True, row=i+1, col=1)\n",
    "        fig.add_annotation(x=sd_residuals + mean_residuals, y=5, text=f\"SD: {sd_residuals:.2f}\", showarrow=False, row=i+1, col=1)\n",
    "        \n",
    "        # QQ-Normal of residuals\n",
    "        qq = stats.probplot(residuals, dist=\"norm\", plot=None)\n",
    "        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0], mode='lines',  showlegend=False), row=i+1, col=2)\n",
    "        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker_color=color, name=f'{col.title()} QQ'), row=i+1, col=2)\n",
    "\n",
    "        # Residuals vs. predicted values\n",
    "        fig.add_trace(go.Scatter(x=predicted, y=residuals, mode='markers', marker_color=color, name=f'{col.title()} Resid Pred'), row=i+1, col=3)\n",
    "        fig.add_hline(y=0, line=dict(color='red'), row=i+1, col=3)\n",
    "        fig.add_hline(y=2 * rmse, line=dict(color='red', dash='dash'), row=i+1, col=3)\n",
    "        fig.add_hline(y=-2 * rmse, line=dict(color='red', dash='dash'), row=i+1, col=3)\n",
    "\n",
    "        # Residuals vs. index\n",
    "        fig.add_trace(go.Scatter(x=index, y=residuals, mode='markers', marker_color=color, name=f'{col.title()} Resid Index'), row=i+1, col=4)\n",
    "        fig.add_hline(y=0, line=dict(color='red'), row=i+1, col=4)\n",
    "        fig.add_hline(y=2 * rmse, line=dict(color='red', dash='dash'), row=i+1, col=4)\n",
    "        fig.add_hline(y=-2 * rmse, line=dict(color='red', dash='dash'), row=i+1, col=4)\n",
    "\n",
    "    # Update layout\n",
    "    fig.update_layout(height=250*num_features, width=1400, title_text=\"Residual Analysis \" + title_add)\n",
    "    fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5f8bb66b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:31.967260Z",
     "iopub.status.busy": "2023-12-31T15:20:31.966846Z",
     "iopub.status.idle": "2023-12-31T15:20:34.487024Z",
     "shell.execute_reply": "2023-12-31T15:20:34.485803Z"
    },
    "papermill": {
     "duration": 2.561503,
     "end_time": "2023-12-31T15:20:34.490542",
     "exception": false,
     "start_time": "2023-12-31T15:20:31.929039",
     "status": "completed"
    },
    "scrolled": true,
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Residual Analysis Plot\n",
    "plotly_residual_analysis(df_test_pred, \n",
    "                         title_add=f'- {SYMBOL} Vanilla LSTM')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "02f916f8",
   "metadata": {
    "papermill": {
     "duration": 0.03719,
     "end_time": "2023-12-31T15:20:34.569725",
     "exception": false,
     "start_time": "2023-12-31T15:20:34.532535",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### LSTM Predictions "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa8c78f4",
   "metadata": {
    "_kg_hide-input": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:34.654226Z",
     "iopub.status.busy": "2023-12-31T15:20:34.652837Z",
     "iopub.status.idle": "2023-12-31T15:20:34.667825Z",
     "shell.execute_reply": "2023-12-31T15:20:34.666014Z"
    },
    "papermill": {
     "duration": 0.060204,
     "end_time": "2023-12-31T15:20:34.670672",
     "exception": false,
     "start_time": "2023-12-31T15:20:34.610468",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def plot_predictions(y_values_df, predictions_df, title_add=''):\n",
    "    \"\"\"\n",
    "    Plots actual values and predictions for each feature in separate subplots.\n",
    "    \n",
    "    Args:\n",
    "        y_values_df (pd.DataFrame): DataFrame containing actual values.\n",
    "        predictions_df (pd.DataFrame): DataFrame containing predicted values.\n",
    "        title_add (str, optional): Additional title for the subplots.\n",
    "    \"\"\"\n",
    "    # Number/color per features \n",
    "    columns = [col for col in y_values_df.columns]\n",
    "    num_features = len(columns)\n",
    "    actual_colors = ['cyan', 'lime', 'yellow', 'violet', 'gold', 'pink']\n",
    "\n",
    "    # Creating subplots\n",
    "    fig = make_subplots(rows=num_features, cols=1, vertical_spacing=0.03, subplot_titles=[col.title() for col in columns])\n",
    "\n",
    "    for i, col in enumerate(columns):\n",
    "        # Actual values trace\n",
    "        fig.add_trace(go.Scatter(x=y_values_df.index, y=y_values_df[col], mode='lines', name=col.title(),\n",
    "                                 line=dict(color=actual_colors[i % len(actual_colors)])), row=i+1, col=1)\n",
    "        \n",
    "        # Predicted values trace\n",
    "        pred_col = f'pred_{col}'\n",
    "        if pred_col in predictions_df.columns:\n",
    "            fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df[pred_col], \n",
    "                                     mode='lines', name=f'Predicted {col.title()}', line=dict(color='red')), row=i+1, col=1)\n",
    "            \n",
    "            # Calculate RMSE and add as an annotation\n",
    "            rmse = np.sqrt(np.mean((y_values_df[col] - predictions_df[pred_col]) ** 2))\n",
    "            fig.add_annotation(xref='x domain', yref='y domain', x=1, y=0.05, showarrow=False,\n",
    "                               text=f'RMSE: {rmse:.2f}', row=i+1, col=1, font=dict(color='red'))\n",
    "    fig.update_layout(height=350*num_features, width=1100, title_text=\"Data & Predictions \" + title_add)\n",
    "    fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64fbfc95",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:34.755518Z",
     "iopub.status.busy": "2023-12-31T15:20:34.755110Z",
     "iopub.status.idle": "2023-12-31T15:20:34.875902Z",
     "shell.execute_reply": "2023-12-31T15:20:34.874763Z"
    },
    "papermill": {
     "duration": 0.165731,
     "end_time": "2023-12-31T15:20:34.878732",
     "exception": false,
     "start_time": "2023-12-31T15:20:34.713001",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Plotting prediction and data\n",
    "plot_predictions(df_y_test, df_predictions, title_add=f'- {SYMBOL} Vanilla LSTM')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9ea1678",
   "metadata": {
    "papermill": {
     "duration": 0.04094,
     "end_time": "2023-12-31T15:20:34.961484",
     "exception": false,
     "start_time": "2023-12-31T15:20:34.920544",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Forecasting with LSTM Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1dc612f3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:35.047119Z",
     "iopub.status.busy": "2023-12-31T15:20:35.046730Z",
     "iopub.status.idle": "2023-12-31T15:20:35.755779Z",
     "shell.execute_reply": "2023-12-31T15:20:35.754548Z"
    },
    "papermill": {
     "duration": 0.754522,
     "end_time": "2023-12-31T15:20:35.758445",
     "exception": false,
     "start_time": "2023-12-31T15:20:35.003923",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Getting last steps for LSTM forecast\n",
    "last_steps = df[columns].iloc[-mapping_steps:, :]\n",
    "\n",
    "# Number of steps to forecast\n",
    "n_forecast_steps = 10 # ~ 2 weeks in business days\n",
    "\n",
    "# Forming date index\n",
    "date_index = pd.date_range(start=end_date, periods=n_forecast_steps, freq=BDay())\n",
    "\n",
    "# Forecasting n-steps\n",
    "forecast_array = rnn_formater.forecast_n_steps(model, last_steps, n_forecast_steps)\n",
    "forecast = pd.DataFrame(forecast_array, index=date_index, columns=[f'forecast_{col}' for col in columns])\n",
    "show(forecast)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5a12e41",
   "metadata": {
    "_kg_hide-input": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:35.849486Z",
     "iopub.status.busy": "2023-12-31T15:20:35.849027Z",
     "iopub.status.idle": "2023-12-31T15:20:35.860938Z",
     "shell.execute_reply": "2023-12-31T15:20:35.859769Z"
    },
    "papermill": {
     "duration": 0.060801,
     "end_time": "2023-12-31T15:20:35.863549",
     "exception": false,
     "start_time": "2023-12-31T15:20:35.802748",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Plotting function\n",
    "def plot_stock_data(df, previous_data=None, test_data=None, title_add=''):\n",
    "    colors = ['blue', 'green', 'purple', 'orange', 'cyan']\n",
    "\n",
    "    fig = make_subplots(rows=df.shape[1], cols=1, shared_xaxes=True, vertical_spacing=0.02,\n",
    "                        subplot_titles=df.columns)\n",
    "\n",
    "    for i, col in enumerate(df.columns):\n",
    "        fig.add_trace(\n",
    "            go.Scatter(x=df.index, y=df[col], mode='lines', name=col, line=dict(dash='dashdot', color='red'),\n",
    "                      ), row=i+1, col=1)\n",
    "\n",
    "        if previous_data is not None:\n",
    "            column = list(previous_data.columns)[i]\n",
    "            fig.add_trace(go.Scatter(x=previous_data.index, y=previous_data[column], mode='lines', name=column,\n",
    "                                     line=dict(color=colors[i % len(colors)])), row=i+1, col=1)\n",
    "\n",
    "        if test_data is not None:\n",
    "            column = list(test_data.columns)[i]\n",
    "            fig.add_trace(\n",
    "                go.Scatter(x=test_data.index, y=test_data[column], mode='lines', name=f'Unseen {col.title()}',\n",
    "                           line=dict(color='floralwhite')), row=i+1, col=1)\n",
    "            \n",
    "\n",
    "    fig.update_layout(height=1200, width=1100, title_text=f\"Stock Data Over Time {title_add}\")\n",
    "    fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "875d58d8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:35.952380Z",
     "iopub.status.busy": "2023-12-31T15:20:35.951770Z",
     "iopub.status.idle": "2023-12-31T15:20:36.124757Z",
     "shell.execute_reply": "2023-12-31T15:20:36.123369Z"
    },
    "papermill": {
     "duration": 0.219562,
     "end_time": "2023-12-31T15:20:36.127332",
     "exception": false,
     "start_time": "2023-12-31T15:20:35.907770",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Actual stock values to test forecast\n",
    "df_test = ticker.history(start=end_date, end='2023-12-15').iloc[-n_forecast_steps:, :]\n",
    "df_test.columns = df_test.columns.str.lower()\n",
    "\n",
    "# Plotting forecast\n",
    "plot_stock_data(forecast, last_steps, df_test, title_add=f'- {SYMBOL} Vanilla LSTM')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2da3b401",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:36.218031Z",
     "iopub.status.busy": "2023-12-31T15:20:36.217599Z",
     "iopub.status.idle": "2023-12-31T15:20:36.250935Z",
     "shell.execute_reply": "2023-12-31T15:20:36.249653Z"
    },
    "papermill": {
     "duration": 0.082011,
     "end_time": "2023-12-31T15:20:36.253563",
     "exception": false,
     "start_time": "2023-12-31T15:20:36.171552",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Error per column on New Data\n",
    "error_array = forecast.values - df_test[columns].values\n",
    "errors = pd.DataFrame(error_array, index=df_test.index, columns=[f'error_{col}' for col in columns])\n",
    "show(errors)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e923e86",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:36.345019Z",
     "iopub.status.busy": "2023-12-31T15:20:36.344610Z",
     "iopub.status.idle": "2023-12-31T15:20:36.368411Z",
     "shell.execute_reply": "2023-12-31T15:20:36.367086Z"
    },
    "papermill": {
     "duration": 0.072187,
     "end_time": "2023-12-31T15:20:36.370990",
     "exception": false,
     "start_time": "2023-12-31T15:20:36.298803",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# RMSE per column on New Data\n",
    "root_mean_squared_error = np.sqrt((errors**2).mean())\n",
    "root_mean_squared_error = pd.DataFrame(root_mean_squared_error, columns=['RMSE New TLSA Data'])\n",
    "show(root_mean_squared_error.T)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8befd3fc",
   "metadata": {
    "papermill": {
     "duration": 0.043893,
     "end_time": "2023-12-31T15:20:36.459642",
     "exception": false,
     "start_time": "2023-12-31T15:20:36.415749",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "- This is very simimilar to the error on the test data used in the validation of the LSTM above!"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dfefcdb8",
   "metadata": {
    "papermill": {
     "duration": 0.043616,
     "end_time": "2023-12-31T15:20:36.635707",
     "exception": false,
     "start_time": "2023-12-31T15:20:36.592091",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Understanding Stacked LSTM \n",
    "   \n",
    "In time series forecasting, such as predicting stock prices, we often use Long Short-Term Memory (LSTM) networks due to their ability to capture temporal dependencies. Two common architectures of LSTM models are single-layer and stacked (multi-layer) LSTMs. Each has unique characteristics suited for different complexities in time series data.\n",
    "\n",
    "#### Single-Layer LSTM\n",
    "A single-layer LSTM consists of one LSTM layer. It is relatively simpler and can efficiently model time series data where relationships between time steps are not overly complex. This architecture is particularly effective for shorter sequences or less volatile data.\n",
    "\n",
    " **Simplicity:** Easier to train and less prone to overfitting on smaller datasets.\n",
    " \n",
    " **Speed:** Generally faster to train due to fewer parameters.\n",
    " \n",
    " **Use Case:** Ideal for more straightforward time series problems where the relationship between past and future data points is more linear or less volatile.\n",
    " \n",
    "**Mathematical Process:** Each time step's input is processed through a series of gates (forget, input, and output) within this single layer, which influences the hidden state and cell state.\n",
    "\n",
    " **Equations:** The LSTM unit updates are based on the current input and the previous hidden state. The mathematical operations are confined within a single series of transformations.\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "    \n",
    "# Mutli-Step LSTM Data Preperation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94651f40",
   "metadata": {
    "_kg_hide-input": false,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:36.725692Z",
     "iopub.status.busy": "2023-12-31T15:20:36.724490Z",
     "iopub.status.idle": "2023-12-31T15:20:36.741130Z",
     "shell.execute_reply": "2023-12-31T15:20:36.740210Z"
    },
    "papermill": {
     "duration": 0.064193,
     "end_time": "2023-12-31T15:20:36.743694",
     "exception": false,
     "start_time": "2023-12-31T15:20:36.679501",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "class RNNFormaterMultiStep:\n",
    "    def __init__(self, data: pd.DataFrame, n_steps_in, n_steps_out):\n",
    "        \"\"\"\n",
    "        Initialize the RNNFormater with a DataFrame, number of input steps, and number of output steps.\n",
    "        \n",
    "        Args:\n",
    "            data (pd.DataFrame): Input DataFrame containing time series data.\n",
    "            n_steps_in (int): Number of time steps for each input sequence.\n",
    "            n_steps_out (int): Number of time steps for each output sequence.\n",
    "        \"\"\"\n",
    "        self.df = data.copy()\n",
    "        self.n_steps_in = n_steps_in\n",
    "        self.n_steps_out = n_steps_out\n",
    "        self.n_columns = self.df.shape[1]\n",
    "        \n",
    "        self.scaler = MinMaxScaler()\n",
    "        self.normalized_data = self.scaler.fit_transform(self.df.values)\n",
    "\n",
    "    def data_mapping(self):\n",
    "        \"\"\"\n",
    "        Maps a 2D array into a 3D array for RNN input, with each sequence having n_steps_in time steps\n",
    "        and each target having n_steps_out time steps.\n",
    "        \n",
    "        Returns:\n",
    "            X (np.array): A 3D array of input sequences.\n",
    "            y (np.array): A 3D array of target sequences.\n",
    "        \"\"\"\n",
    "        num_samples = len(self.normalized_data) - self.n_steps_in - self.n_steps_out + 1\n",
    "\n",
    "        X = np.empty((num_samples, self.n_steps_in, self.n_columns))\n",
    "        y = np.empty((num_samples, self.n_steps_out, self.n_columns))\n",
    "\n",
    "        for i in range(num_samples):\n",
    "            X[i, :, :] = self.normalized_data[i:i + self.n_steps_in, :]\n",
    "            y[i, :, :] = self.normalized_data[i + self.n_steps_in:i + self.n_steps_in + self.n_steps_out, :]\n",
    "\n",
    "        return X, y\n",
    "\n",
    "    def rnn_train_test_split(self, X, y, test_percent=0.1):\n",
    "        \"\"\"\n",
    "        Splits the 3D mapped data into training and testing sets for an RNN.\n",
    "        \n",
    "        Args:\n",
    "            X (np.array): The input data sequences.\n",
    "            y (np.array): The target data sequences.\n",
    "            test_percent (float): The fraction of data to be used for testing.\n",
    "        \n",
    "        Returns:\n",
    "            X_train, X_test, y_train, y_test (tuple): Split data into training and testing sets.\n",
    "        \"\"\"\n",
    "        test_size = int(len(X) * test_percent)\n",
    "        X_train, y_train = X[:-test_size], y[:-test_size]\n",
    "        X_test, y_test = X[-test_size:], y[-test_size:]\n",
    "\n",
    "        return X_train, X_test, y_train, y_test\n",
    "\n",
    "    def multi_step_forecast(self, model, data: np.array):\n",
    "        \"\"\"\n",
    "        Forecast multiple steps ahead using the LSTM model.\n",
    "        \n",
    "        Args:\n",
    "            model (tf.keras.Model): Trained LSTM model for prediction.\n",
    "            data (pd.DataFrame): Input DataFrame containing the latest time series data.\n",
    "        \n",
    "        Returns:\n",
    "            pd.DataFrame: Forecasted values for n_steps_out steps.\n",
    "        \"\"\"    \n",
    "        # Normalizing latest data\n",
    "        last_steps_normalized = self.scaler.transform(data)\n",
    "        last_steps_normalized = last_steps_normalized.reshape(1, self.n_steps_in, self.n_columns)\n",
    "        \n",
    "        # Predicting using model\n",
    "        forecast = model.predict(last_steps_normalized)\n",
    "        forecast = forecast.reshape(self.n_steps_out, self.n_columns)\n",
    "\n",
    "        # Inverse transforming to original scale\n",
    "        forecast = self.scaler.inverse_transform(forecast)\n",
    "        forecast = pd.DataFrame(forecast, columns=[f'forecast_{col}' for col in self.df.columns])\n",
    "        \n",
    "        return forecast\n",
    "        "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1388608b",
   "metadata": {
    "papermill": {
     "duration": 0.04336,
     "end_time": "2023-12-31T15:20:36.830521",
     "exception": false,
     "start_time": "2023-12-31T15:20:36.787161",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Building Stacked Mutli-Step LSTM \n",
    "    \n",
    "Here, we use again use the `MinMaxScaler` from the `sklearn.preprocessing`, this is nicely done in the class `RNNFormaterMultiStep`\n",
    "\n",
    "**In our example we have our Tesla stock price prediction model:**\n",
    "- **Input (X):** We map `252` time steps of past stock prices, representing about `1 year` of stock data in business days.\n",
    "- **Output (y):** The model forecasts the next `21` time steps, equivalent to `1 month` ahead in business days.\n",
    "\n",
    "--- \n",
    "\n",
    "**What It Is**: `kernel_initializer` in neural networks, such as those in Keras/TensorFlow, is a parameter that sets the method to initialize the weights in a layer.\n",
    "\n",
    "**Mathematical Basis**: \n",
    "- Orthogonal Initialization: This method sets weights as orthogonal matrices, preserving the variance from input to output of the layer. Mathematically, for a weight matrix $ W $, it ensures $ W^T W = I $ or $ W W^T = I $ with $ I $ being the identity matrix.\n",
    "\n",
    "**Why It Matters**: \n",
    "- Proper weight initialization is crucial for efficient training. It can impact the speed of convergence and the final performance of the network.\n",
    "- Orthogonal initialization is beneficial in deep networks as it avoids vanishing and exploding gradients.\n",
    "\n",
    "# Bidirectional LSTM \n",
    "    \n",
    "A **bidirectional LSTM** is a type of RNN architecture designed to process and analyze sequential data. Unlike traditional LSTMs that read sequences in one direction, bidirectional LSTMs process data in both forward and backward directions simultaneously. This bidirectional processing allows the model to capture dependencies from both past and future contexts.\n",
    "\n",
    "During the training phase, the bidirectional LSTM considers information from earlier time steps to later ones, and vice versa. This helps the model understand patterns that may exist across various points in the sequence. The bidirectional LSTM consists of two separate layers – one processing the sequence from the beginning to the end, and the other from the end to the beginning. The outputs from both directions are typically concatenated or combined to provide a more comprehensive representation of the input sequence.\n",
    "This architecture is useful in tasks where understanding both past and future information is essential, such as time-series analysis and natural language processing. By incorporating bidirectional processing, the LSTM becomes more adept at capturing complex relationships within sequential data.\n",
    "\n",
    "**Bidirectional Model Example**\n",
    "\n",
    "```python\n",
    "# Multi-Step LSTM\n",
    "ms_model = models.Sequential([\n",
    "    LSTM(units=40, return_sequences=True, input_shape=(n_steps_in, len(columns)), kernel_initializer='orthogonal'),\n",
    "    Bidirectional(LSTM(units=20, return_sequences=False)),\n",
    "    Dense(units=n_steps_out * len(columns))   \n",
    "])\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbbcef76",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:36.920072Z",
     "iopub.status.busy": "2023-12-31T15:20:36.919396Z",
     "iopub.status.idle": "2023-12-31T15:20:36.947147Z",
     "shell.execute_reply": "2023-12-31T15:20:36.946032Z"
    },
    "papermill": {
     "duration": 0.075574,
     "end_time": "2023-12-31T15:20:36.949743",
     "exception": false,
     "start_time": "2023-12-31T15:20:36.874169",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "n_steps_in = 252  # ~ 1 year in business days mapping\n",
    "n_steps_out = 21  # ~ 1 month forecast in business days\n",
    "\n",
    "# Intializing created RNN class\n",
    "ms_rnn_formatter = RNNFormaterMultiStep(df[columns], n_steps_in, n_steps_out)\n",
    "\n",
    "# Normalizing and data mappings for input & output\n",
    "X, y = ms_rnn_formatter.data_mapping()\n",
    "\n",
    "# Train Test split\n",
    "X_train, X_test, y_train, y_test = ms_rnn_formatter.rnn_train_test_split(X, y)\n",
    "\n",
    "print(f'X_train shape: {X_train.shape}\\ny_train shape: {y_train.shape}') # (time mapes, time steps, columns)\n",
    "print(f'\\nX_test shape: {X_test.shape}\\ny_test shape: {y_test.shape}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "674ed02e",
   "metadata": {
    "papermill": {
     "duration": 0.044409,
     "end_time": "2023-12-31T15:20:37.039438",
     "exception": false,
     "start_time": "2023-12-31T15:20:36.995029",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Training Stacked Mutli-Step LSTM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a88a6378",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:37.131156Z",
     "iopub.status.busy": "2023-12-31T15:20:37.130339Z",
     "iopub.status.idle": "2023-12-31T15:20:37.819597Z",
     "shell.execute_reply": "2023-12-31T15:20:37.818376Z"
    },
    "papermill": {
     "duration": 0.738824,
     "end_time": "2023-12-31T15:20:37.823299",
     "exception": false,
     "start_time": "2023-12-31T15:20:37.084475",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# For consistant results\n",
    "random.seed(0)\n",
    "np.random.seed(0)\n",
    "tf.random.set_seed(0)\n",
    "\n",
    "# Multi-Step LSTM\n",
    "ms_model = models.Sequential([\n",
    "    \n",
    "    # Layer 1\n",
    "    LSTM(units=40, return_sequences=True, input_shape=(n_steps_in, len(columns)), kernel_initializer='orthogonal'),\n",
    "\n",
    "    # Layer 2\n",
    "    LSTM(units=20, return_sequences=False),\n",
    "    # Bidirectional(LSTM(units=20, return_sequences=False)),\n",
    "\n",
    "    # Output layer\n",
    "    Dense(units=n_steps_out * len(columns))   \n",
    "])\n",
    "\n",
    "ms_model.compile(optimizer='adam', loss='mse')\n",
    "ms_model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36e09b97",
   "metadata": {
    "_kg_hide-output": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:20:37.920353Z",
     "iopub.status.busy": "2023-12-31T15:20:37.919961Z",
     "iopub.status.idle": "2023-12-31T15:24:57.948812Z",
     "shell.execute_reply": "2023-12-31T15:24:57.947680Z"
    },
    "papermill": {
     "duration": 260.079153,
     "end_time": "2023-12-31T15:24:57.951871",
     "exception": false,
     "start_time": "2023-12-31T15:20:37.872718",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Callback to save model weights\n",
    "model_checkpoint = ModelCheckpoint('Multi_LSTM_Tesla_model.h5', monitor='val_loss', save_best_only=True)\n",
    "\n",
    "# Fitting the model\n",
    "ms_history = ms_model.fit(X_train, y_train.reshape(-1, n_steps_out * len(columns)), \n",
    "                          batch_size=1024,\n",
    "                          epochs=150, \n",
    "                          validation_data=(X_test, y_test.reshape(-1, n_steps_out * len(columns))), \n",
    "                          callbacks=[model_checkpoint],\n",
    "                          shuffle=False, \n",
    "                          verbose=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9bc96c1c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:24:58.048904Z",
     "iopub.status.busy": "2023-12-31T15:24:58.048136Z",
     "iopub.status.idle": "2023-12-31T15:24:58.067380Z",
     "shell.execute_reply": "2023-12-31T15:24:58.065892Z"
    },
    "papermill": {
     "duration": 0.07066,
     "end_time": "2023-12-31T15:24:58.070184",
     "exception": false,
     "start_time": "2023-12-31T15:24:57.999524",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Plotting stakced multi-step model loss\n",
    "plot_training_history(ms_history.history,\n",
    "                      plot_title='Stacked Multi-Step LSTM Model Loss')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe9d6f63",
   "metadata": {
    "papermill": {
     "duration": 0.046993,
     "end_time": "2023-12-31T15:24:58.163676",
     "exception": false,
     "start_time": "2023-12-31T15:24:58.116683",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Mutli-Step Forecasting with Stacked Multi-Step LSTM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8871bcad",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:24:58.263525Z",
     "iopub.status.busy": "2023-12-31T15:24:58.263086Z",
     "iopub.status.idle": "2023-12-31T15:24:58.293084Z",
     "shell.execute_reply": "2023-12-31T15:24:58.291927Z"
    },
    "papermill": {
     "duration": 0.085026,
     "end_time": "2023-12-31T15:24:58.296760",
     "exception": false,
     "start_time": "2023-12-31T15:24:58.211734",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Getting stock data for forecasting\n",
    "data_to_test = df[columns].iloc[-n_steps_in:]\n",
    "\n",
    "# Showing data\n",
    "print(f'Test Data Shape: {data_to_test.shape}')\n",
    "show(data_to_test.tail())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "96a578cc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:24:58.395706Z",
     "iopub.status.busy": "2023-12-31T15:24:58.395230Z",
     "iopub.status.idle": "2023-12-31T15:25:00.173505Z",
     "shell.execute_reply": "2023-12-31T15:25:00.172153Z"
    },
    "papermill": {
     "duration": 1.831261,
     "end_time": "2023-12-31T15:25:00.176309",
     "exception": false,
     "start_time": "2023-12-31T15:24:58.345048",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Loading best wieghts during training\n",
    "ms_model = models.load_model('Multi_LSTM_Tesla_model.h5')\n",
    "\n",
    "# Forecasting using rnn\n",
    "forecast = ms_rnn_formatter.multi_step_forecast(ms_model, data_to_test.values)\n",
    "\n",
    "# Setting index as datetime corresponding to forecast period\n",
    "forecast.index = pd.date_range(start=end_date, periods=n_steps_out, freq=BDay())\n",
    "print(f'Forecast shape: {forecast.shape}')\n",
    "show(forecast)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2a2fcc8",
   "metadata": {
    "_kg_hide-input": true,
    "execution": {
     "iopub.execute_input": "2023-12-31T15:25:00.279209Z",
     "iopub.status.busy": "2023-12-31T15:25:00.278772Z",
     "iopub.status.idle": "2023-12-31T15:25:00.292783Z",
     "shell.execute_reply": "2023-12-31T15:25:00.291660Z"
    },
    "papermill": {
     "duration": 0.068203,
     "end_time": "2023-12-31T15:25:00.296446",
     "exception": false,
     "start_time": "2023-12-31T15:25:00.228243",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def plot_forecast(previous_data, forecast, test_data=None, title_add=''):\n",
    "    \"\"\"\n",
    "    Plots actual values and predictions for each feature in separate subplots.\n",
    "    \n",
    "    Args:\n",
    "        previous_data (pd.DataFrame): DataFrame containing actual values.\n",
    "        forecast (pd.DataFrame): DataFrame containing predicted values.\n",
    "        title_add (str, optional): Additional title for the subplots.\n",
    "    \"\"\"\n",
    "    # Number/color per features \n",
    "    columns = [col for col in previous_data.columns]\n",
    "    num_features = len(columns)\n",
    "    actual_colors = ['cyan', 'gold', 'violet', 'lime', 'blue', 'pink', 'yellow']\n",
    "\n",
    "    # Creating subplots\n",
    "    fig = make_subplots(rows=num_features, cols=1, vertical_spacing=0.03, subplot_titles=[col.title() for col in columns])\n",
    "\n",
    "    for i, col in enumerate(columns):\n",
    "        # Actual values trace\n",
    "        fig.add_trace(\n",
    "            go.Scatter(x=previous_data.index, y=previous_data[col], mode='lines', name=col.title(),\n",
    "                       line=dict(color=actual_colors[i % len(actual_colors)])), row=i+1, col=1)\n",
    "        \n",
    "        # Predicted values trace\n",
    "        pred_col = f'forecast_{col}'\n",
    "        if pred_col in forecast.columns:\n",
    "            fig.add_trace(\n",
    "                go.Scatter(x=forecast.index, y=forecast[pred_col], \n",
    "                           mode='lines', name=f'Forecast {col.title()}', line=dict(color='red')), row=i+1, col=1)\n",
    "\n",
    "        if test_data is not None:\n",
    "            fig.add_trace(\n",
    "                go.Scatter(x=test_data.index, y=test_data[col], mode='lines', name=f'Unseen {col.title()}',\n",
    "                           line=dict(color='floralwhite')), row=i+1, col=1)\n",
    "            \n",
    "            \n",
    "    fig.update_layout(height=350*num_features, width=1100, title_text=\"Data and Forecast \" + title_add)\n",
    "    fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e51246e2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-31T15:25:00.401803Z",
     "iopub.status.busy": "2023-12-31T15:25:00.400520Z",
     "iopub.status.idle": "2023-12-31T15:25:00.605479Z",
     "shell.execute_reply": "2023-12-31T15:25:00.603661Z"
    },
    "papermill": {
     "duration": 0.260911,
     "end_time": "2023-12-31T15:25:00.608221",
     "exception": false,
     "start_time": "2023-12-31T15:25:00.347310",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Actual stock values to test forecast\n",
    "df_test = ticker.history(start=end_date).iloc[-n_steps_out:, :]\n",
    "df_test.columns = df_test.columns.str.lower()\n",
    "\n",
    "# Plotting stacked mutli-step LSTM forecast\n",
    "plot_forecast(data_to_test[columns], forecast, df_test, title_add='- TSLA Stock')"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 1007,
     "sourceId": 1814,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 500872,
     "sourceId": 927894,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 1436765,
     "sourceId": 3358622,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 1971512,
     "sourceId": 3441564,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 3626523,
     "sourceId": 7306405,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30626,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 818.159251,
   "end_time": "2023-12-31T15:25:03.092786",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-12-31T15:11:24.933535",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
