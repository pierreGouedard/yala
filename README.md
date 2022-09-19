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
  
