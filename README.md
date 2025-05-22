# MPO

> The official implementation for the ACL 2025 paper *MPO: Multilingual Safety Alignment via Reward Gap Optimization*.

<img src="https://img.shields.io/badge/Venue-ACL--25-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red">

## Requirement & Installation

This repository is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and follow the same requirements and installation procedures.

| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.9     | 3.10      |
| torch        | 2.0.0   | 2.6.0     |
| torchvision  | 0.15.0  | 0.21.0    |
| transformers | 4.45.0  | 4.50.0    |
| datasets     | 2.16.0  | 3.2.0     |
| accelerate   | 0.34.0  | 1.2.1     |
| peft         | 0.14.0  | 0.15.1    |
| trl          | 0.8.6   | 0.9.6     |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.16.4    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.3   | 0.8.2     |
| flash-attn   | 2.5.6   | 2.7.2     |

```bash
pip install -e ".[torch,metrics]" --no-build-isolation
```

## Dataset

The data has been placed in the `/data` directory and registered in `data_info.json`, including `gemma_mpo_data.json`, `llama_mpo_data.json`, and `qwen_mpo_data.json`, respectively.

## Training

Please run the following command to start the training process.

```sh
llamafactory-cli train examples/train_mpo/{model}_mpo.yaml
```

model = gemma2 / llama3.1 / qwen2.5

<!-- ## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@inproceedings{zhao2024sapt,
  title={Sapt: A shared attention framework for parameter-efficient continual learning of large language models},
  author={Zhao, Weixiang and Wang, Shilong and Hu, Yulin and Zhao, Yanyan and Qin, Bing and Zhang, Xuanyu and Yang, Qing and Xu, Dongliang and Che, Wanxiang},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={11641--11661},
  year={2024}
}
``` -->

## Credits
The code of this repository relies on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and we would like to show the sincere gratitude to authors of it.
