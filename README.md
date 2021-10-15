# Simple pytorch integration with pudb


This is a simple stringifier viewing tensor attributes when debugging pytorch code with pudb.
It shows tensor shapes, devices and trainable parameters for modules.

A simple example of the usage is shown below (notice the tensors and modules in the Variables pane)

![Alt text](img/torch_pudb.png?raw=true "Example Usage")

Notice the tensors and network layers are presented with accompanying debugging information.

## How to install

Follow the instructions in the [example stringifier](https://github.com/inducer/pudb/blob/main/example-stringifier.py) to install.
