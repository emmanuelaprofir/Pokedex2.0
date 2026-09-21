I made this project to understand and create a neural network that learns and recognize 4 differents Pokemons.
Pikachu, Elektek, Lucario, and Cosmog.

The process is functionnal and we can see the unfolding of how the AI learns, the number of epochs and the differents percentages.

I added a early stop because in the way that I made it, there are not enough data and to many epochs. 
This is called overfitting, it learns to many details and cannot recognize a Pokemon simply. 
So when it reaches 100% of accuracy, it tolerates 4 more epochs where there's no more improvements and stops by itself.

In order to make it work,you need a virtual environnement of python in 3.11, activate it (venv\Scripts\activate.bat)
And you need to install tensorflow 2.15 
I used vscode, so I needed to install every libraries with pip install.

In the source code, you need to change the paths.
