To set up autoreload we much alter our ipython config file located at `~/.ipython/profile_default/ipython_config.py`.  **NOTE**: If you don't have this file, run `ipython profile create` from the TERMINAL (not from an ipython session!)

We will then alter the following line from...
```
# lines of code to run at IPython startup.
# c.InteractiveShellApp.exec_lines = []
```
to...
```
# lines of code to run at IPython startup.
c.InteractiveShellApp.exec_lines = [
                                    '%load_ext autoreload',
                                    '%autoreload 2'
]
```

This will enable autoreload everytime you start up an ipython session.

There are, however, some subtleties about how autoreload works.  Let's say I have a file called `test.py` with the following code...
```python
def my_func(num):
    return num + 4

if __name__=='__main__':
    print my_func(4)
```
If we ran `run test.py` from a ipython session it would print out 8.  Let's say we then alter the file to be the following...
```python
def my_func(num):
    print 'Am I working?'
    return num + 4

if __name__=='__main__':
    print my_func(4)
```
And then execute the command `my_func(4)` the output will still just be 8.

The issue is that we didn't actually direct ipython to track any changes made to this file.  We can solve this by saying `from test import *` or `from test import my_func`.  After we do that, any changes that we make will be automatically reflected in our ipython shell!
