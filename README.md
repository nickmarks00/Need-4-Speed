# Penguin Pi

A branch for development work on the Penguin Pi.

## Setting up the Penguin Pi.

Before connecting to the Penguin Pi, run the following lines to install dependencies. That is, theses commands should be executed locally.
```
sudo apt install python-opencv python-pygame python-numpy python-matplotlib
```

Note that if you run into installation issues that cannot find packages, try each of these packages but `python3-*` instead.

## Connecting

```
python3 operate.py --ip=192.168.50.1 --port=8080
```
And you should be good to go!


## Notes
- It is weird that we are installing with `apt`. Perhaps it would be better to try setting up a virtual environment and installing the packages from there.
