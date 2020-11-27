# ML Project Code
## Setup
Make sure to run this first to avoid dependency issues:
```py
pip install -e MLPrimitives/
pip install -e Orion/
pip install -e mSSA/
pip install -r requirements.txt
```

## Adding a method
Suppose your method is called `MyMethod` and will be implemented in the file `my_method.py`.

* Go to `Orion/orion/primitives`. Create the file `my_method.py`. Taking inspiration from `orbit.py` implement a `fit` and `predict` method of the `MyMethod` class.
* Go to `Orion/orion/primitives/json`. Create the file `orion.primitives.my_method.MyMethod.json`.
* Copy/Paste its content from `orion.primitives.orbit.OrbitTAD.json`. Make sure the rename the references to `orbit` and to `OrbitTAD` by `my_method` and `MyMethod` and to modify parameters.
* Go to `Orion/orion/pipelines/verified/`. Create a `my_method` folder, with a `my_method.json` file.
* Again, take inspiration from the `Orion/orion/pipelines/verified/orbit/orbit.json`. Make sure to again update the names and parameters.


## Testing a method
* Open `essai.py`, and replace `orbit` by `my_method`.
* Put in the datasets you want.
* Run.
