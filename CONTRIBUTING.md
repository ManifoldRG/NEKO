Welcome to the contributing guide for NEKO, and thank you for investing your time in contributing to our project!

Join our Discord [here](https://discord.gg/a8uDbxzEbM).

In this guide, you will get an overview of the contribution workflow for the NEKO project.

## New contributor guide

To get an overview of the project, read the [README](https://github.com/ManifoldRG/NEKO#readme). Our project roadmap is also available at [NEKO Roadmap](https://docs.google.com/document/d/e/2PACX-1vQ2JVJvSiYmwjDFnppj0_38NCUEdLG8pAdj0Q2tSy1yy4wwQxJOAAzNFwz2Is4TONhgUVnvJzuu5o85/pub).

## Solve an issue

See [Issues](https://github.com/ManifoldRG/NEKO/issues) for more information. If you find an issue to work on, you are welcome to look into it and engage in discourse in the comments on the issue.

### Contributing to the Codebase

Fork the repository, create a new branch, make your changes, and then submit a pull request (PR).

So far, most of the code for model training is complete, contribution to that part of the codebase is mostly fixing issues or improving efficiency.

The part that needs the most new code is to add a new modality or task. As of now, we have implemented some tasks such as control, text, image-caption and VQA (the last two tasks are for single images only). If you want to implement a new task, here are the majority of the work:
- Define a new task type in `gato/tasks/task.py`
- Define your new task under the `gato/tasks` folder (refer to the tasks already implemented here, you should have a good idea about what you need to do):
    - Open and load the datasets for the task, process the data into the format to be accepted by model training
    - Define the function to sample data from this task for training
    - Define a function to evaluate the performance of this task
- Define the function for inference/predicton specific to this task in `gato/policy/gato_policy.py`. The rest of this file defines the training model for NEKO, and most likely you do not need to touch that part
- Add a few lines of code in `gato/training/trainer.py` to sample data for the new task and add to each batch for training
- Add a few lines of code in `train.py` to input the command-line args specific to your task and to instantiate an instance of your task

You can take a look of a PR and find out more concrete details to match what is described above, for example, https://github.com/ManifoldRG/NEKO/pull/30

As a final note, please be respectful when commenting on these issues as they are publicly viewable. Thank you!
