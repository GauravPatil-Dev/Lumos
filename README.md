Before you start, do `pip install -r requirements.txt`. Then you can refer to `demo.ipynb` for our deployment and evaluation script.


# LoRA Model Training Lumos

This project involves data preparation, training a model using LoRA (Low-Rank Adaptation), and evaluation. The project utilizes DeepSpeed for efficient distributed training. Lumos takes market news and optional basic financials related to the specified company from the past few weeks as input and responds with the company's **positive developments** and **potential concerns**. Then it gives out a **prediction** of stock price movement for the coming week and its **analysis** summary. Lumos is finetuned on Llama-2-7b-chat-hf with LoRA on the past year's DOW30 market data. But also has shown great generalization ability on other ticker symbols.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview
This project aims to train a model using the LoRA technique, which optimizes model parameters efficiently. The training process is enhanced with DeepSpeed for better performance on GPUs.

## Requirements
- Python 3.7 or higher
- Git
- Virtualenv (recommended)

### Python Packages
- torch==2.0.1
- transformers==4.32.0
- peft==0.5.0
- pandas
- yfinance
- finnhub-python
- nvidia-ml-py3

## Setup

1. **Clone the Repository**:
   git clone git@github.com:GauravPatil-Dev/Lumos.git
   cd your-repo-name

## Create a Virtual Environment:

    `python -m venv venv`
    `source venv/bin/activate  # On Windows: venv\Scripts\activate`

## Install Dependencies:
    `pip install -r requirements.txt`


## Data Preparation
Company profile & Market news & Basic financials & Stock prices are retrieved using **yfinance & finnhub**.

Prompts used are organized as below:

```
SYSTEM_PROMPT = "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n"

prompt = """
[Company Introduction]:

{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding. {name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry.

From {startDate} to {endDate}, {name}'s stock price {increase/decrease} from {startPrice} to {endPrice}. Company news during this period are listed below:

[Headline]: ...
[Summary]: ...

[Headline]: ...
[Summary]: ...

Some recent basic financials of {name}, reported at {date}, are presented below:

[Basic Financials]:
{attr1}: {value1}
{attr2}: {value2}
...

Based on all the information before {curday}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company-related news. Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction.

"""
```

## Configure Training:

Ensure the `config.json` file is correctly set up for your training parameters.

Run Training Script:
Execute the train.sh script to start training:
    `sh train.sh`