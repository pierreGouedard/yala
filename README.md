# Yet Another Learning Algorithm - YALA


### Onboarding

In the above section, we present the different steps needed to be able to run the notebooks. Steps that includes command 
line operation will be described on Ubunutu
 
  * **Get authorization**
  
  Clone repository *yala* and *firing_graph* from *https://github.com/PierreGouedard*
    
  * **Create a virtual env to install requirement**
  
  Create virtual env using the tool of your choice. A list of python requirements can be installed from requirements.txt 
  and a conda environment config file can be found  at the root of the ml-analysis project (with ubuntu / conda):
    
    cd ~/yala
    conda env create -f dev-env.yaml
  
  * **Install firing_graph package on your local computer**
  
  activate the virtual env and install firing_graph module (with ubuntu / conda):
    
    conda activate yala-env
    cd ~/firing_graph
    pip install --no-cache-dir -e .


### TODO:

Test if this is working properly. Then make it a bit more generalizable.
  - today only continuous / complete features is allowed (simply try to enlarge bounds)
    what if the continuous / complete is not true (i.e categorical value - days of week)
    There is a simple solution to that => instead of looking at enlargind neighbour bit of a bound blindly 
    Do something that will reveal what are the close bits => vertext at |I| - 1 projected on bound will give you 
    the closest bit of that bound are the one that receive more signals.
  - Ohter improvement, make the algorithm more robust when it comes to selecting / removing bits => prevent too noisy
    deletion by adding a notion of smoothing of bounds.

After those two steps (that runs correctly) test shapes mining => usefull to see if algorithm still converge properly
and add test for 1) noisy shape recovery 2) when a features values as no continuousity in euclidean space. Then its 
probably good to go on testing on famous tabular dataset prediction.

  
