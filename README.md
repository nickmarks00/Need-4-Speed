# Need-4-Speed

## Demos

I've added a demo Frozen Lake simulation that uses basic Q-learning. It can be used to easily check that your Open Gymnasium environment is set-up correctly.

### Usage

Firstly, make sure that `pipenv` is installed using `pip`. Then:
1. Change into the `gym-frozen-lake` directory.
2. Run `pipenv sync` or `pipenv install` to install the dependencies from the `Pipfile.lock` file.
3. You might need to run `pipenv shell` before this. In either case, run it now to activate the virtual environment.
4. You can then execute the Frozen Lake demo using `pipenv run python3 frozen_lake.py`. It should print the state space and action space, then the rewards from training. Finally it plays 3 games and prints the results of each.
5. Change the `render_mode` to `human` to activate the simulation and see the training games being played "live".
