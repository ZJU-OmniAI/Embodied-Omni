# <img src="./assets/embodied-reasoner-logo.png" height="50">Embodied-Reasoner 
✨This is the official implementation of paper 
<a href="https://arxiv.org/abs/2503.21696">Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks</a>

> **📦 This repository has moved.** It was previously hosted at `github.com/zwq2018/embodied_reasoner` — the URL cited in our paper — and now lives under the [ZJU-OmniAI](https://github.com/ZJU-OmniAI) organization. GitHub redirects the old URL here automatically, so existing links, clones, and remotes keep working.

<table>
  <tr>
    <td width="30%">
      <p>1. 💫 Embodied Task</p>  
      <p>2. 💫 Deep Reasoning Model</p>
      <p>3. 💫 Multimodal Scene</p>
      <p>4. 💫 Long-horizon Decision</p>
      <p>5. 💫 Multi-turn Interaction</p>
    </td>
    <td width="70%">
      <p align="center">
        <img src="./assets/embodied_reasoner.JPG" height="300"/>
      </p>
    </td>
  </tr>
</table>

<p align="center">
          🤗 <a href="https://huggingface.co/datasets/zwq2018/embodied_reasoner">Hugging Face</a>
          <!-- &nbsp&nbsp | &nbsp&nbsp 🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a> -->
          <!-- &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.21696">Arxiv</a>  -->
          &nbsp&nbsp | &nbsp&nbsp <img src="assets/arxiv_logo.png" alt="arXiv" height="14"> <a href="https://arxiv.org/abs/2503.21696">Arxiv</a> 
          &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://embodied-reasoner.github.io">WebPage</a> 
          &nbsp&nbsp | &nbsp&nbsp  📺  <a href="https://www.bilibili.com/video/BV1Cs7Hz4ETk?t=1623.2">Bilibili</a>
<br>
</p>

## Video 📷 📷
<!-- <p align="center">
   <video width="640" controls autoplay muted>
    <source src="./assets/video_v2_en.mp4">
  </video> 
</p> -->
<!-- [![Watch the demo](./assets/video_thumbnail.png)](https://github.com/ZJU-OmniAI/embodied_reasoner/blob/main/assets/video_v2_en.mov?raw=true) -->


<!--https://github.com/user-attachments/assets/677497aa-2b8e-4ed2-a8c7-249bbe0b2206-->


https://github.com/user-attachments/assets/da9c5b42-ab8e-4101-9ec0-a226590d23fc


🎙️ **Paper Sharing**: https://www.bilibili.com/video/BV1Cs7Hz4ETk?t=1623.2



## News 🔥🔥
- **2026.04:** Embodied-Reasoner Accepted by ACL 2026 Main
- **2026.01:** [Invited Talk@视觉语言导航](https://www.bilibili.com/video/BV149cjz5Es5/?spm_id_from=333.337.search-card.all.click&vd_source=818fc4816fcb6a1d8c82455cd7851b48)
- **2025.05:** [Invited Talk@智猩猩](https://www.bilibili.com/video/BV1Cs7Hz4ETk?spm_id_from=333.788.videopod.sections&vd_source=818fc4816fcb6a1d8c82455cd7851b48)
- **2025.03:** We release our paper and dataset.
## Contents 🌳🌳
- [Overview](#Overview)
- [Performance](#preformance)
- [Examples](#examples)
- [Training](#training)
    - [Step 1. Install Requirements](#step-1-install-requirements)
    - [Step 2. Data prepare](#step-2-data-prepare)
    - [Step 3. Run Training Scripts](#step-3-run-training-scripts)
- [Evaluation](#evaluation)
    - [Step 1. Install Requirements](#step-1-install-requirements-1)
    - [Step 2. Run Evaluation Scripts](#step-2-run-evaluation-scripts)
- [Task and Trajectory Engine](#Overview)
- [Citation](#citation)
- [License](#license)

## Overview 🦾🦾
In this paper, we present **Embodied-Reasoner**, a multimodal embodied model that extends o1-style deep-reasoning capabilities to embodied interactive tasks. It can perform complex tasks in AI2THOR such as searching for hidden objects, manipulating and transporting items with several impressive features:
<div style="display: flex; align-items: center;">
  <div style="flex: 0.33;">
    <ul>
      <li><strong>👉 Deep Reasoning</strong> abilities, e.g., <strong>analysis</strong>, <strong>spatial reasoning</strong>, <strong>reflection</strong>, <strong>planning</strong>.</li>
      <li><strong>👉 Interleaved Multimodal Processing</strong> capabilities, especially handling long sequences of interleaved image-text context</li>
      <li><strong>👉 Environmental Interaction</strong> abilities, enabling it to autonomously observe the environment, explore rooms, and find hidden objects</li>
      <li><strong>👉 Open-source Models</strong> released in 7B/2B sizes</li>
      <li><strong>👉 Open-source Dataset</strong> 🤗 <a href="https://huggingface.co/datasets/zwq2018/embodied_reasoner">Hugging Face</a>: 9.3k interleaved observation-reasoning-action trajectories, including 64K images and 8M thought tokens.</li>
    </ul>
  </div>
  <div style="flex: 0.67; display: flex; justify-content: center; align-items: center;">
    <img src="./assets/fig0.png" height="400"/>
  </div>    
</div>           




Our contributions can be summarized as follows:   

> **Task and Trajectory Engine:** Automatically synthesizes coherent **Observation-Thought-Action** trajectories, spaning 107 diverse indoor scenes, e.g., kitchens and living rooms, and covers 2,100 interactive objects (e.g., eggs, laptops) and 2,600 containers (e.g., refrigerators, drawers), *64K* a first-person perspective image from interaction and *8M* thought tokens. 
<p align="center">
    <img src="./assets/table.jpg" height="180" style="display:inline-block; margin-right:10px;"/>
<p>  

> **Long CoT with Diverse Thinking Pattern:** analysis, spatial reasoning, reflection, planning, and verification. These coherent, image-text interleaved trajectories boost its spatial, temporal reasoning capabilities.
<p align="center">
    <img src="./assets/state_trans.jpg" height="350" style="display:inline-block;"/>
<p>  

> **Iterative Training Pipeline:** A three-stage iterative training pipeline that combines **imitation learning**, **self-exploration tunning**, and **self-correction tunning**.   


> **Interactive Evaluation Framework:** 809 test cases across 12 novel scenarios: `<Instruction, Key Action, Final state >`

<p align="center">
    <img src="./assets/dataset.jpg" height="370" style="display:inline-block; margin-right:10px;"/>
    <img src="./assets/object.jpg" height="370" style="display:inline-block;"/>
<p>



## Performance 🌿🌿
We compare the performance of Embodied-Reasoner against advanced VLMs and visual reasoning models. 
- Success Rate (%) measures whether a task is successfully completed. 
- Search Efficiency (%) evaluates task efficiency—more steps indicate lower efficiency. 
- Task Completeness (%) computes the proportion of predicted actions that belong to the set of key actions.
<p align="center">
    <img src="./assets/performance.jpg"/>
<p>

## Examples 👀 👀
### Simulator Experiments
Embodied-Reasoner exhibits spontaneous thinking behaviors, e.g., analyzing environmental states (#1,3), reflecting on missed details (#4), reasoning based on the latest observations (#5), and recalling cues for efficient planning (#9). These thoughts remain coherent and logically consistent despite spanning multiple rounds. In contrast, general VLMs lacking thinking abilities struggle with long-horizon interactive tasks and produce unreasonable actions, e.g., forget tasks or repetitive searching.

<p align="center">
    <img src="./assets/example.jpg"/>
<p>

### Real-World Experiments
To evaluate the generalization of our reasoning model, we design a real-world experiment. Our model rules out the countertop and dining table after two explorations (steps 1,2), ultimately locating the coffee (#7) in the cabinet and placing it in the microwave for heating (#11). However, we observe that OpenAI o3-mini fails to formulate a reasonable plan, heading to the microwave first instead of searching for the coffee.

<p align="center">
    <img src="./assets/real_example.jpg" height="400"/>
<p>

## QuickStart 🎯🎯
### Training

#### Step 1. Install Requirements
```shell
conda create -n llama-factory python=3.11
conda activate llama-factory
git clone -b embodied-reasoner https://github.com/iGangao/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install wandb accelerate deepspeed importlib-metadata
```

#### Step 2. Data prepare
Please refer to `data/README.md` for checking the details about the format of dataset files.

#### Step 3. Run training scripts
Run the training scripts:
```shell
bash scripts/train.sh
```

### Evaluation

#### Step 1. Install Requirements

```shell
conda create -n embodied-reasoner python=3.9
conda activate embodied-reasoner
pip install -r requirements.txt
```

#### Step 2. Run evaluation scripts
Run the evaluation scripts:
```shell
bash scripts/eval.sh
```

## Task and Trajectory Engine ⛲⛲

You can navigate to the data_engine folder to synthesize tasks and trajectories. Below are the key files within the data_engine:

```plaintext
data_engine/
├── taskgenerate/               # Item information and room metadata for task generation
│   ├── bathrooms/
│   ├── bedrooms/
│   ├── kitchens/
│   ├── living_rooms/
│   └── pick_up_and_put.json
├── TaskGenerate.py             # Task synthesis script
├── o1StyleGenerate.py          # Trajectory synthesis script
├── o1StyleGenerate_ordered.py  # Complex task trajectory synthesis script
├── vlmCall.py                  # Script to call the VLM
└── vlmCallapi_keys.py          # Please Set your API keys here

```


#### Step 1. Generate Task
`TaskGenerate.py` can synthesize task templates and corresponding key actions. The generated task-related data will be stored in the `<tasktype>_metadata` folder under data_engine.

You can run the following Python script to perform the task generation, and can modify parameters like task types within this Python file.
```shell
python TaskGenerate.py
```

For example, one generated task data entry is shown below, where actions contains a list of key actions for the task.
```json
{
    "taskname": "Locate the Apple in the room.",
    "tasktype": "single_search",
    "metadatapath": "taskgenerate/kitchens/FloorPlan1/metadata.json",
    "actions": [
        {
            "action": "navigate to",
            "objectId": "CounterTop|-00.08|+01.15|00.00",
            "objectType": "CounterTop",
            "baseaction": "",
            "reward": 1,
            "relatedObject": [
                "CounterTop|-00.08|+01.15|00.00",
                "Apple|-00.47|+01.15|+00.48"
            ]
        },
        {
            "action": "end",
            "objectId": "",
            "objectType": "",
            "baseaction": "",
            "reward": 1,
            "relatedObject": [
                "CounterTop|-00.08|+01.15|00.00",
                "Apple|-00.47|+01.15|+00.48"
            ]
        }
    ],
    "totalreward": 2
}
```

#### Step 2. Generate O1-style Trajectory
`o1StyleGenerate.py` and `o1StyleGenerate_ordered.py` can synthesize trajectories for 10 different sub-task types. Specifically, o1StyleGenerate_ordered.py is designed to synthesize more complex sequential object transfer tasks.

You can run the following Python script to perform the trajectory generation. Additionally, you can set the task type and trajectory type within the script (typically, 'b' is shortest, 'a' is longer, and 'c' is the longest).

```shell
python o1StyleGenerate.py
python o1StyleGenerate_ordered.py
```

Below is an example folder of a generated trajectory, including the JSON file and associated images for the trajectory.
<p align="center">
    <img src="./assets/trajectory.png"/ height=180>
<p>


Below is an example of the JSON file contents:

```json
{
    "scene": "FloorPlan1",
    "tasktype": "...",
    "taskname": "Locate the Apple in the room.",
    "trajectory": [
        "<...>...</...>",
        "<...>...</...>",
        "..."
    ],
    "images": [
        ".../init_observe.png",
        "..."
    ],
    "flag": "",
    "time": "...",
    "task_metadata": {
        "..."
    }
}
```
- **scene:** the scene where the task is performed.
- **tasktype:** the type of the task.
- **taskname:** the name of the task.
- **trajectory:** reasoning and decision-making content of the trajectory
- **images:** paths to corresponding images (the first image represents the initial state; each subsequent image corresponds to the state after performing each action listed in trajectory).
- **time and flag:** records the generation timestamp and exceptions encountered during trajectory generation.
- **task_metadata:** task information generated during Step 1.



To view our complete trajectory dataset, please visit our <a href="https://huggingface.co/datasets/zwq2018/embodied_reasoner">Hugging Face</a> Page.

Please refer to `data_endine/README.md` for checking the details about the data engine.

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{embodied-reasoner,
    title   = {Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks}, 
    author  = {Wenqi Zhang and Mengna Wang and Gangao Liu and Huixin Xu and Yiwei Jiang and Yongliang Shen and Guiyang Hou and Zhe Zheng and Hang Zhang and Xin Li and Weiming Lu and Peng Li and Yueting Zhuang},
    journal = {arXiv preprint arXiv:2503.21696},
    year    = {2025}
}

```

## License
[![Code License](https://img.shields.io/badge/Code_License-%E6%9C%A8%E5%85%B0-yellow)](LICENSE)

The codebase is licensed under 木兰. 

## Contact Us
If you have any questions, please contact us by email: 
zhangwenqi@zju.edu.cn, lipeng@iscas.ac.cn

## Acknowledgements
Our training code uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and uses the Simulator with [Ai2-THOR](https://github.com/allenai/ai2thor). Thanks for their wonderful works.
