# Training Math Agent with Data-Augment Strategies

This example demonstrates how to use **AgentScope-Tuner** to enhance a math problem-solving agent. We will focus on leveraging **Data-Centric** features, such as the `difficulty_based` task selector, to improve data utility and training efficiency.

## Task Setting

We use the foundational [math-agent example](../react_agent/main.py) as our baseline to demonstrate the data enhancement capabilities. Notably, these data-centric techniques are generic and customizable, making them adaptable to other agent workflows.

### Agent Goal and Type
The agent's objective is to solve mathematical reasoning problems, learning to produce a correct final answer through a step-by-step thought process. The agent is implemented as a **`ReActAgent`**, which follows a reasoning-acting loop to solve tasks iteratively.

### Environment
Each task is a question-answer pair from a math dataset. The agent's performance is evaluated based on its final answer.

### Objective of the Data-Centric Approach

Training can be inefficient if tasks are too easy or too hard. This example addresses this by providing **selectors** to dynamically select tasks using **data feedback**. This empowers users to explore and implement their own data-centric strategies, such as focusing on "productively challenging" samples, to maximize training efficiency.

## Dataset Preparation

To enable difficulty-based sampling, our training data needs to include features that represent the "difficulty" of each task.

1.  **Base Dataset**: You can use any standard math problem dataset. A good example is the math data in [LLM360/guru-RL-92k](https://huggingface.co/datasets/LLM360/guru-RL-92k), which comes pre-annotated with pass rates from different LLMs, serving as direct difficulty features.
2.  **Build Your Own Features**: If you use your own dataset, you can generate these features by pre-running several models of varying capabilities and recording their pass rates. This can be done within the [Trinity](https://github.com/modelscope/Trinity-RFT/pull/440) framework.
3.  **Data Format**: The final dataset should be in HuggingFace format. In this example, data will be transferred to *GSM8K format* according to the [workflow](../react_agent/main.py). Besides the task content, it must include the difficulty feature columns you've defined (e.g., `qwen_7b_pass_rate`, `qwen_30b_pass_rate`).
4.  **Example Data Preparation**: We provide a script for this example. Simply execute `python prepare_data.py` to generate the required dataset.

## Code Implementation

### Agent Workflow & Judge Function

This example follows the foundational [math-agent example](../react_agent/main.py), adopting its `run_react_agent` and `gsm8k_judge` as the `workflow_func` and `judge_func`, respectively. This highlights a key benefit: you can apply training strategies without altering your core agent logic.

### Design of Data-Centric Features

Leveraging the powerful data processing capabilities of **`Trinity`**, **AgentScope-Tuner** provides interfaces for advanced operations like task selection and experience processing.

#### Task Selector

The `Task Selector` determines how samples are selected from a dataset. It is configured within the `Dataset` object in your Python script.

- **Built-in Selectors**:
  - `sequential`: Samples are selected in a fixed order.
  - `shuffle`: The dataset is shuffled at the beginning of each epoch.
  - `random`: Samples are randomly chosen with replacement for each batch.
  - `offline_easy2hard`: Samples are sorted by a predefined feature for curriculum learning.
  - `difficulty_based` (Customized): An adaptive sampler based on task difficulty.

> For more details on `Task Selector`, including how to implement a custom selector based on feedback signals, please refer to Trinity's **[Selector Development Guide](https://github.com/modelscope/Trinity-RFT/blob/main/docs/sphinx_doc/source/tutorial/develop_selector.md)**.

#### Data Processor

The `Data Processor` allows for real-time processing of **Task** and **Experience** during training, enabling operations like calculating feedback metrics, data augmentation, or filtering.

For example, the `difficulty_based` selector requires a `pass_rate_calculator` operator to compute the agent's success rate for each task. This feedback is then used to adjust the sampling strategy.

> For more details on `Data Processor`, please refer to Trinity's **[Operator Development Guide](https://github.com/modelscope/Trinity-RFT/blob/main/docs/sphinx_doc/source/tutorial/develop_operator.md)**.


### Configuring the Experiments

We demonstrate how to set up two experiments to compare the baseline `random` selector against the `difficulty_based` selector.

**Experiment 1: Baseline with Random Selector**

In `main_random.py`, we configure the `task_selector` for random sampling.

```python
# In main_random.py
train_dataset = Dataset(
    path="path/to/your/augmented/math_data",
    split="train",
    task_selector={'selector_type': 'random'},
)

tune(
    workflow_func=run_react_agent,
    judge_func=gsm8k_judge,
    config_path="config_random.yaml",
    train_dataset=train_dataset,
    ...
)
```

**Experiment 2: Advanced Training with Difficulty-Based Selector**

In `main_difficulty.py`, we switch the selector to `difficulty_based` and provide initial feature keys.

```python
# In main_difficulty.py
train_dataset = Dataset(
    path="path/to/your/augmented/math_data",
    split="train",
    task_selector={
        'selector_type': 'difficulty_based',
        'feature_keys': ["qwen_7b_pass_rate", "qwen_30b_pass_rate"],
        'kwargs': {...}, # Hyper-parameters for the selection algorithm
    },
)

tune(
    workflow_func=run_react_agent,
    judge_func=gsm8k_judge,
    config_path="config_difficulty.yaml",
    train_dataset=train_dataset,
    ...
)
```

> The `difficulty_based` selector in this example is an implementation of the ***BOTS*** algorithm. For details on its inner workings, please refer to the [***BOTS paper***](https://arxiv.org/abs/2510.26374) and its [***tutorials***](https://github.com/modelscope/Trinity-RFT/blob/main/examples/bots/README.md).

The `config_difficulty.yaml` must enable the `pass_rate_calculator` to provide real-time feedback.

```yaml
# Enable the calculator to provide feedback for the selector
data_processor:
  experience_pipeline:
    operators:
      - name: pass_rate_calculator
```

## How to Run

### Step 1: Prerequisites

Ensure you have installed AgentScope and Trinity-RFT with [the guidance](../react_agent/README.md).

### Step 2: Prepare the Dataset

Run the data preparation script. Make sure to update the dataset paths in `main_random.py` and `main_difficulty.py` afterward.

```bash
python prepare_data.py
```

### Step 3: Start Ray Cluster

For distributed training, start a Ray cluster.

```bash
# For single node
ray start --head
```

### Step 4: Run Training

You can now run either the baseline or the difficulty-based training experiment.

- **To run the baseline experiment with a random selector:**

```bash
python main_random.py
```

- **To run the experiment with the difficulty-based selector:**
```bash
python main_difficulty.py
```

## Experimental Results 

The following results compare the performance of the `difficulty-based` selection strategy (red line, bots) against a standard `random` selection strategy (black line, random).

![Training Result Image](./training_result.jpg)

### Training Reward Curve

The chart on the left shows the rollout accuracy during training. As can be seen, the tasks sampled by the random strategy appear to be difficult for the model, with the accuracy remaining below 0.2. In contrast, using the difficulty selector results in a higher mean accuracy, indicating that the agent is engaging with more tasks that it can successfully solve.

### Evaluation on AIME-24

For comparison, we evaluated both selection strategies on the AIME-24 benchmark. The chart on the right shows that the difficulty-based method demonstrates a better upward trend in performance over time.
