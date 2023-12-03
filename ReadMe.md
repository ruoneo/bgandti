# A bidirectional GAN framework for predicting unknown drug-target interactions. 

## usage

```commandline
python ./run.py
```
## description
you can train the model by `run.py`

## requirement
```text
TensorFlow==1.15.4  
dppy==0.2.0  
munkres  
scikit-learn  
cvxopt  
```
> `cvxopt` requires `Numpy-MKL`, you can get the windows binary [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy). And install the `cvxopt` binary from [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#cvxopt)

## acknowledge
Thanks for the projects **GAT：** [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)