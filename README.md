<!-- Temporary fix to not upset page formatting.
In future, we should move most of the information in the README to
a dedicated docs page, where we can control the size of images etc.
-- >
<!-- markdownlint-disable MD033 -->

# TimeLLM

> [!IMPORTANT]
> This is a fork of the original [Time-LLM repository](https://github.com/KimMeen/Time-LLM).
> Please credit their work using the citation information below.
>
> This fork turns the original work into an installable package.
> See [the installation instructions](#installation) below for how to get started.

## [(ICLR'24) Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://github.com/KimMeen/Time-LLM)

![Last commit badge](https://img.shields.io/github/last-commit/KimMeen/Time-LLM?color=green)
![Number of stars badge](https://img.shields.io/github/stars/KimMeen/Time-LLM?color=yellow)
![Number of forks badge](https://img.shields.io/github/forks/KimMeen/Time-LLM?color=lightblue)
![PRs welcome badge](https://img.shields.io/badge/PRs-Welcome-green)

| **[Paper Page](https://arxiv.org/abs/2310.01728)**
| **[YouTube Talk](https://www.youtube.com/watch?v=6sFiNExS3nI)**
| **[YouTube Talk 2](https://www.youtube.com/watch?v=L-hRexVa32k)**
| **[Medium Blog](https://medium.com/towards-data-science/time-llm-reprogram-an-llm-for-time-series-forecasting-e2558087b8ac)**

| **[机器之心中文解读](https://www.jiqizhixin.com/articles/2024-04-15?from=synced&keyword=TIME-LLM)**
| **[量子位中文解读](https://mp.weixin.qq.com/s/UL_Kl0PzgfYHOnq7d3vM8Q)**
| **[时序人中文解读](https://mp.weixin.qq.com/s/FSxUdvPI713J2LiHnNaFCw)**
| **[AI算法厨房中文解读](https://mp.weixin.qq.com/s/nUiQGnHOkWznoBPqM0KHXg)**
| **[知乎中文解读](https://zhuanlan.zhihu.com/p/676256783)**

<p align="center">
<img src="./figures/logo.png" width="70" alt="TimeLLM logo">
</p>

---

> 🙋 Please let us know if you find out a mistake or have any suggestions!
>
> 🌟 If you find this resource helpful, please consider to star this repository and cite the original author's research:

```tex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Updates/News

🚩 **News** (Oct. 2025): Time-LLM has been cited 1,000 times in the past two years! 🎉 We are deeply grateful to the community for the incredible support along the journey.

🚩 **News** (Aug. 2024): Time-LLM has been adopted by XiMou Optimization Technology Co., Ltd. (XMO) for Solar, Wind, and Weather Forecasting.

🚩 **News** (Oct. 2024): Time-LLM has been included in [PyPOTS](https://pypots.com/). Many thanks to the PyPOTS team!

🚩 **News** (May 2024): Time-LLM has been included in [NeuralForecast](https://github.com/Nixtla/neuralforecast). Special thanks to the contributor @[JQGoh](https://github.com/JQGoh) and @[marcopeix](https://github.com/marcopeix)!

🚩 **News** (Mar. 2024): Time-LLM has been upgraded to serve as a general framework for repurposing a wide range of language models to time series forecasting. It now defaults to supporting Llama-7B and includes compatibility with two additional smaller PLMs (GPT-2 and BERT). Simply adjust `--llm_model` and `--llm_dim` to switch backbones.

## Introduction

Time-LLM is a reprogramming framework to repurpose LLMs for general time series forecasting with the backbone language models kept intact.
Notably, we show that time series analysis (e.g., forecasting) can be cast as yet another "language task" that can be effectively tackled by an off-the-shelf LLM.

<p align="center">
<img src="./figures/framework.png" height = "360" alt="" align=center />
</p>

- Time-LLM comprises two key components: (1) reprogramming the input time series into text prototype representations that are more natural for the LLM, and (2) augmenting the input context with declarative prompts (e.g., domain expert knowledge and task instructions) to guide LLM reasoning.

<p align="center">
<img src="./figures/method-detailed-illustration.png" height = "190" alt="" align=center />
</p>

## Installation

We recommend that you create a virtual environment (for example with `conda` or `uv`) to install `timellm` into, or install it into the existing virtual environment for the project that you want to use it with.

Once you have created a virtual environment, you can install the package either directly from GitHub (recommended) or by cloning the repository and locally installing (recommended for developers / contributors).

To install from GitHub, in your chosen virtual environment, run

```sh
pip install git+https://github.com/NSs-FLF-RHUL/Time-LLM.git
```

To clone and install locally, in your chosen virtual environment, run

```sh
cd path/to/where/you/want/your/local/copy/to/be
git clone https://github.com/NSs-FLF-RHUL/Time-LLM.git
cd Time-LLM
# To install normally, run
pip install .
# To create an editable installation and fetch additional
# developer dependences (recommended for developers), run
pip install -e .[dev]
```

## Datasets

You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), then place the downloaded contents under `./dataset`

## Quick Demos

1. Download datasets and place them under `./dataset`
2. Tune the model. We provide five experiment scripts for demonstration purpose under the folder `./scripts`. For example, you can evaluate on ETT datasets by:

```bash
bash ./scripts/TimeLLM_ETTh1.sh
```

```bash
bash ./scripts/TimeLLM_ETTh2.sh
```

```bash
bash ./scripts/TimeLLM_ETTm1.sh
```

```bash
bash ./scripts/TimeLLM_ETTm2.sh
```

## Detailed usage

Please refer to `run_main.py`, `run_m4.py` and `run_pretrain.py` for the detailed description of each hyperparameter.

## Further Reading

As one of the earliest works exploring the intersection of large language models and time series, we sincerely thank the open-source community for supporting our research. While we do not plan to make major updates to the main Time-LLM codebase, we still welcome **constructive pull requests** to help maintain and improve it.

🌟 Please check out our team’s latest research projects listed below.

1, [**TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models**](https://arxiv.org/pdf/2509.24803), _arXiv_ 2025.

**Authors**: Tong Guan, Zijie Meng, Dianqi Li, Shiyu Wang, Chao-Han Huck Yang, Qingsong Wen, Zuozhu Liu, Sabato Marco Siniscalchi, Ming Jin, Shirui Pan

```bibtex
@article{guan2025timeomni,
  title={TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models},
  author={Guan, Tong and Meng, Zijie and Li, Dianqi and Wang, Shiyu and Yang, Chao-Han Huck and Wen, Qingsong and Liu, Zuozhu and Siniscalchi, Sabato Marco and Jin, Ming and Pan, Shirui},
  journal={arXiv preprint arXiv:2509.24803},
  year={2025}
}
```

2, [**Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement**](https://arxiv.org/pdf/2503.01875), in _ACL_ 2025.
[\[HuggingFace\]](https://huggingface.co/Time-MQA)

**Authors**: Yaxuan Kong, Yiyuan Yang, Yoontae Hwang, Wenjie Du, Stefan Zohren, Zhangyang Wang, Ming Jin, Qingsong Wen

```bibtex
@inproceedings{kong2025time,
  title={Time-mqa: Time series multi-task question answering with context enhancement},
  author={Kong, Yaxuan and Yang, Yiyuan and Hwang, Yoontae and Du, Wenjie and Zohren, Stefan and Wang, Zhangyang and Jin, Ming and Wen, Qingsong},
  booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)},
  year={2025}
}
```

3, [**Towards Neural Scaling Laws for Time Series Foundation Models**](https://arxiv.org/pdf/2410.12360), in _ICLR_ 2025.
[\[GitHub Repo\]](https://github.com/Qingrenn/TSFM-ScalingLaws)

**Authors**: Qingren Yao, Chao-Han Huck Yang, Renhe Jiang, Yuxuan Liang, Ming Jin, Shirui Pan

```bibtex
@inproceedings{yaotowards,
  title={Towards Neural Scaling Laws for Time Series Foundation Models},
  author={Yao, Qingren and Yang, Chao-Han Huck and Jiang, Renhe and Liang, Yuxuan and Jin, Ming and Pan, Shirui},
  booktitle={International Conference on Learning Representations (ICLR)}
  year={2025}
}
```

4, [**Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts**](https://arxiv.org/pdf/2409.16040), in _ICLR_ 2025.
[\[GitHub Repo\]](https://github.com/Time-MoE/Time-MoE)

**Authors**: Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, Ming Jin

```bibtex
@inproceedings{shi2024time,
  title={Time-moe: Billion-scale time series foundation models with mixture of experts},
  author={Shi, Xiaoming and Wang, Shiyu and Nie, Yuqi and Li, Dianqi and Ye, Zhou and Wen, Qingsong and Jin, Ming},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

5, [**TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis**](https://arxiv.org/abs/2410.16032), in _ICLR_ 2025.
[\[GitHub Repo\]](https://github.com/kwuking/TimeMixer/blob/main/README.md)

**Authors**: Shiyu Wang, Jiawei Li, Xiaoming Shi, Zhou Ye, Baichuan Mo, Wenze Lin, Shengtong Ju, Zhixuan Chu, Ming Jin

```bibtex
@inproceedings{wang2024timemixer++,
  title={TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis},
  author={Wang, Shiyu and Li, Jiawei and Shi, Xiaoming and Ye, Zhou and Mo, Baichuan and Lin, Wenze and Ju, Shengtong and Chu, Zhixuan and Jin, Ming},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Acknowledgement

Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [OFA (GPT4TS)](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) as the code base and have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.

## Legal Disclaimer

> [!IMPORTANT]
> [This content was moved to the README from `LEGAL.md`](https://github.com/KimMeen/Time-LLM/blob/main/LEGAL.md).

Within this source code, the comments in Chinese shall be the original, governing version. Any comment in other languages are for reference only. In the event of any conflict between the Chinese language version comments and other language version comments, the Chinese language version shall prevail.

法律免责声明

关于代码注释部分，中文注释为官方版本，其它语言注释仅做参考。中文注释可能与其它语言注释存在不一致，当中文注释与其它语言注释存在不一致时，请以中文注释为准。
