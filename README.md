# wandb-model-observer

You can observe weghts/gradients/outputs of modules in your model!

It is extremely useful to discover where an error happens.

## Usage

```python
from wandb_model_observer import ModelObserver  # import `ModelObserver`

class MyModel(nn.Module):
    ...


model = MyModel()
observer = ModelObserver(model, key="wandb api key", project="your project name", group="your group name in the project")  # ready for observing your model

# training loop
...

# log weights/gradients/outputs on W&B
observer.commit()

```

## Details

`ModelObserver` can monitor weights/gradients of each module in the model when `.backward()` is called, also outputs of each module are monitored when `.forward` is called