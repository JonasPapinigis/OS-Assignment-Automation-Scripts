import numpy as np
import subprocess
import os
from typing import Any
import itertools
import random







"""
numberOfProcesses=4
staticPriority=0
meanInterArrival=50
meanCpuBurst=15.0
meanIOBurst=15.0
meanNumberBursts=2.0
seed=270826029269605
"""


def create_input_parameters(input_data, name: str, loc=None):
    folder_name = loc if loc is not None else 'Data'
    
    os.makedirs(folder_name, exist_ok=True)
    
    file_path = os.path.join(folder_name, name + ".prp")
    
    contents = ""
    for field_name, value in input_data:  # Fixed iteration
        if (field_name == "seed" and value == ""):
            contents += f"{field_name}={random.randint(10**14,10**15)}"
            continue
        contents += f"{field_name}={value}\n"
    
    with open(file_path, 'w') as file:
        file.write(contents)


def list_to_dict(values):

    keys = [
        "numberOfProcesses",
        "staticPriority",
        "meanInterArrival",
        "meanCpuBurst",
        "meanIOBurst",
        "meanNumberBursts",
        "seed"
    ]
    
    result_dict = {key: value for key, value in zip(keys, values)}
    
    return result_dict
def create_experiment(init_values, to_test, test_values, trial_name):
    keys = [
        "numberOfProcesses",
        "staticPriority",
        "meanInterArrival",
        "meanCpuBurst",
        "meanIOBurst",
        "meanNumberBursts",
        "seed"
    ]
    #Allow tuples and single ints
    #Typecheck
    _type = type(to_test)
    for i in test_values:
        if type(i) != _type:
            raise ValueError(f"Ivalid Test Values {i}, {_type}")
    
    if _type == "tuple":
        #Check that to_test in same format as test_values
        test_dimensions = len(to_test)
        for tuple in test_values:
            if len(tuple) != test_dimensions:
                raise ValueError("Test values do not match test dimensions")

    os.makedirs(trial_name, exist_ok=True)
    if _type == "tuple":
        for num,tuple in enumerate(test_values):
            values = init_values.copy()  # Create a copy to avoid mutation
            #cycles to test parameters
            for test_ptr,param in enumerate(to_test):
                #changes that parameter to the value supplied by the test_values
                values[param] = tuple[test_ptr]
                create_input_parameters(zip(keys,values), f"inpt{num}", trial_name)
    else: 
        for num, value in enumerate(test_values):
            values = init_values.copy()  # Create a copy to avoid mutation
            values[to_test] = value  # Set the parameter at index `to_test` to the current test value

            # Assuming create_input_parameters function exists and works as expected
            create_input_parameters(zip(keys, values), f"inpt{num}", trial_name)
        

        












#run = subprocess.run(['cmd', '/c', 'java','-cp', 'target/os-coursework1-1.0-SNAPSHOT.jar', 'InputGenerator','experiment1/input_parameters.prp','experiment1/inputs.in'],stdout=True,stdin=True)

def generate_inputs(trial_name):
    cmds = ""
    if not os.path.exists(trial_name) or not os.path.isdir(trial_name):
        raise ValueError("Input Folder Not found")

    input_files = [(os.path.join(trial_name,f)).replace('\\','/') for f in os.listdir(trial_name) if f.endswith(".prp")]

    for file in input_files:
        base_name = os.path.basename(file).replace(".prp","")
        output_destination = os.path.join(trial_name,base_name) + ".in"
        args = ['cmd', '/c', 'java','-cp', 'target/os-coursework1-1.0-SNAPSHOT.jar', 'InputGenerator', file, output_destination]
        subprocess.run(args)
        cmds += " ".join(args)+"\n"
    cmds += "\n"
    return cmds
        














class SchedulerParams:

    def __init__(self,scheduler: str, timeLimit: int, periodic: bool, interruptTime: int, 
                 timeQuantum: int, initBurstEst: float, alphaBurstEst: float):

        if (scheduler not in ["FcfsScheduler","IdealSJFScheduler","RRScheduler","FeedbackRRScheduler","SJFScheduler"]):
            raise ValueError("Not a Real scheduler (FcfsScheduler,IdealSJFScheduler,RRScheduler,FeedbackRRScheduler,SJFScheduler")
        self.scheduler = scheduler
        self.timeLimit = timeLimit
        self.periodic = periodic
        self.interruptTime = interruptTime
        self.timeQuantum = timeQuantum
        self.initBurstEst = initBurstEst
        self.alphaBurstEst = alphaBurstEst

    def to_dict(self):

        return {
            'scheduler': self.scheduler,
            'timeLimit': self.timeLimit,
            'periodic': self.periodic,
            'interruptTime': self.interruptTime,
            'timeQuantum': self.timeQuantum,
            'initialBurstEstimate': self.initBurstEst,
            'alphaBurstEstimate': self.alphaBurstEst
        }

    def to_list(self):

        return [self.scheduler, self.timeLimit, self.periodic, 
                self.interruptTime, self.timeQuantum, self.initBurstEst, self.alphaBurstEst]




def create_input_file(input_data, name: str, trial_name):
    
    if not os.path.exists(trial_name):
        raise ValueError("Experiment directory not found")
    
    file_path = os.path.join(trial_name, f"{name}.prp")
    
    contents = ""
    for field_name, value in input_data.to_dict().items():
        contents += f"{field_name}={value}\n"
    
    with open(file_path, 'w') as file:
        file.write(contents)

def create_schedulers(init_values, to_test, test_values, trial_name):
    # Define all possible schedulers
    all_schedulers = ["FcfsScheduler", "IdealSJFScheduler", "RRScheduler", "FeedbackRRScheduler", "SJFScheduler"]
    
    # Determine which schedulers to test
    mask = init_values[0]
    if mask:
        schedulers_to_test = [scheduler for scheduler, to_include in zip(all_schedulers, mask) if to_include]
    else:
        schedulers_to_test = all_schedulers
    
    

    for scheduler in schedulers_to_test:
        scheduler_short_name = scheduler.replace("Scheduler", "")  
        
        for num, value in enumerate(test_values):
            values = init_values[1:].copy()  
            
            if to_test > 0:  
                
                values.insert(0, scheduler)
                
                values[to_test] = value
            _input_data = SchedulerParams(*values)
            
            file_name = f"{scheduler_short_name}{num}_"
            create_input_file(_input_data, file_name, trial_name)













#run = subprocess.run(['cmd', '/c', 'java','-cp', 'target/os-coursework1-1.0-SNAPSHOT.jar', 'InputGenerator','experiment1/input_parameters.prp','experiment1/inputs.in'],stdout=True,stdin=True)

def generate_outputs(trial_name):
    cmds = ""

    #print(f"Input,Scheduler,Output Directories:\n{input_path}\n{scheduler_path}\n{output_path}")

    if not os.path.exists(trial_name) or not os.path.isdir(trial_name):
        raise ValueError("Input Folder(s) Not found")

    input_files = [f for f in os.listdir(trial_name) if f.endswith(".in")]
    
    schedulers = [f for f in os.listdir(trial_name) if f.endswith("_.prp")]

    for scheduler in schedulers: 
        for input in input_files:
            scheduler_base = os.path.basename(scheduler)
            input_base = os.path.basename(input)
            
            output_filename = scheduler_base.replace("_.prp","")+input_base.replace(".in","")+".out"
            final_output = os.path.join(trial_name,output_filename).replace("\\","/")
            
            args = ['cmd', '/c', 'java','-cp', 'target/os-coursework1-1.0-SNAPSHOT.jar', 'Simulator', f"{trial_name}/{scheduler}", final_output, f"{trial_name}/{input}"]
            subprocess.run(args)
            cmds += " ".join(args)+"\n"
    cmds += "\n"
    return cmds










experiment_name = "experiment2"
commands = ""
"""numberOfProcesses
staticPriority
meanInterArrival
meanCpuBurst=15.0
meanIOBurst=15.0
meanNumberBursts=3.0
seed=270826029269605
"""
example_params = [5,0,30,15,15,5,""]
_to_test = 0
_test_values = [5,5,5,5,5]
_test_name = experiment_name
create_experiment(example_params,_to_test,_test_values,_test_name)


"""
scheduler=SJFScheduler
timeLimit=10000
periodic=false
interruptTime=0
timeQuantum=20
initialBurstEstimate=10
alphaBurstEstimate=0.5
"""

commands += generate_inputs(experiment_name)
_init_values = [[True,True,True,True,True], 10000, False, 0, 20, 10.0, 0.5]  
_to_test = 3
_test_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51]
_trial_name = experiment_name

create_schedulers(_init_values, _to_test, _test_values, _trial_name)

commands += generate_outputs(experiment_name)


if (os.path.isfile("run.bat")):
    with open("run.bat",'r') as file:
        lines = file.readlines()
        if lines and lines[-1].strip().lower() == 'pause':
            lines = lines[:-1]
    
    lines.append(commands + '\n')
    lines.append('pause\n')

    with open("run.bat", 'w') as file:
        file.writelines(lines)
    

else :
    
    with open("run.bat","w") as file:
        file.write(f"@echo off\n Automated Experiment Recreation\n {commands}\n pause")