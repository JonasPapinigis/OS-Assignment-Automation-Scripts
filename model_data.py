import numpy as np
import matplotlib.pyplot as plt
from io import BufferedReader
import os
import re
from mpl_toolkits.mplot3d import axes3d

#id	priority	createdTime	startedTime	terminatedTime	cpuTime	blockedTime	turnaroundTime	waitingTime	responseTime

def convert_doc(output_doc):
    contents = ""
    processes = []
    attribute_list = ["id",	"priority", "createdTime", "startedTime", "terminatedTime", "cpuTime", "blockedTime", "turnaroundTime","waitingTime", "responseTime"]
    with open(output_doc, 'r') as f:
        contents = f.read()
    all_lines = contents.split("\n")
    raw_processes = [f.replace("\t"," ") for f in all_lines[1:-2]]
    for proc in raw_processes:
        process_dict = {}
        val_array = proc.split(" ")
        for num,attr in enumerate(attribute_list):
            process_dict[attr] = int(val_array[num])
        processes.append(process_dict)

    return processes

def aggregate_data(experiment):
    scheduler_types = ["Fcfs","IdealSJF", "FeedbackRR","RR","SJF"]
    outputs = []
    #Need this to be (Type, Data)
    output_paths = [f"{experiment}/{f}" for f in os.listdir(experiment) if f.endswith(".out")]
    for path in output_paths:
        print(path)
        scheduler = "ERROR"
        input = -1
        #Want this to be in order so RR check comes after FeedbackRR, as it is its subsrtring
        for type in scheduler_types:
            if type in path:
                scheduler = type
                break
        match = re.search(r'inpt(\d+)', path)
        input = int(match.group(1))
        if (input < -1 or scheduler == "ERROR"):
            raise RuntimeError(f"Error Extracting Data:{path}")
        outputs.append((scheduler,input,convert_doc(path)))
    return outputs


exp1_out = aggregate_data("experiment1")
prp_files =  [f"inpt{num}.prp" for num in range(0,225)]
#inpt,cpu,io, log(cpu/io)
inpt_cpuio_vals = {}
for num,filename in enumerate(prp_files):
    with open(os.path.join("experiment1",filename),'r') as file:
        lines = file.readlines()
        cpu_burst = int(lines[3].strip().replace("meanCpuBurst=",""))
        io_burst = int(lines[4].strip().replace("meanIOBurst=",""))
        inpt_cpuio_vals[num] = cpu_burst,io_burst,np.log(cpu_burst/io_burst)

for key,value in inpt_cpuio_vals.items():
    print(f"{key},{value}")


exp1_averages = []
for datapoint in exp1_out:
    num_proc = len(datapoint[2])
    cpu_time = turnaround_time =  blocked_time = waiting_time = 0
    means = {}
    for process in datapoint[2]:
        cpu_time += process.get("cpuTime")
        turnaround_time += process.get("turnaroundTime")
        blocked_time += process.get("blockedTime")
        waiting_time += process.get("waitingTime")
    means['cpuTime'] = cpu_time / num_proc
    means['turnaroundTime'] = turnaround_time / num_proc
    means['blockedTime'] = blocked_time / num_proc
    means['waitingTime'] = waiting_time /num_proc
    exp1_averages.append((datapoint[0],datapoint[1],means))

algorithms = ['Fcfs', 'RR', 'FeedbackRR', 'SJF', 'IdealSJF']
data = {alg: [] for alg in algorithms}  # Dictionary to hold extracted data for each algorithm

# Extracting data for each algorithm
for alg in algorithms:
    data[alg] = [f for f in exp1_averages if f[0] == alg]

# Setting up variables for 3D plotting data
plot_data = {alg: {'_x': [], '_y': [], '_z': []} for alg in algorithms}

# Populate the plotting data
for alg in algorithms:
    for mean in data[alg]:
        if inpt_cpuio_vals.get(mean[1]):
            plot_data[alg]['_x'].append(inpt_cpuio_vals.get(mean[1])[0])
            plot_data[alg]['_y'].append(inpt_cpuio_vals.get(mean[1])[1])
            plot_data[alg]['_z'].append(mean[2].get('turnaroundTime'))

# Optionally, print data for verification
for alg in algorithms:
    print(f"{alg} x-values: {plot_data[alg]['_x']}")
    print(f"{alg} y-values: {plot_data[alg]['_y']}")
    print(f"{alg} z-values: {plot_data[alg]['_z']}")



fig = plt.figure()
ax = fig.add_subplot(projection="3d")
#ax.scatter(plot_data.get("Fcfs").get('_x'),plot_data.get("Fcfs").get('_y'),plot_data.get("Fcfs").get('_z'), marker='x', label = "FCFS")
#ax.scatter(plot_data.get("RR").get('_x'),plot_data.get("RR").get('_y'),plot_data.get("RR").get('_z'), marker=',', label = "Round Robin")
#ax.scatter(plot_data.get("FeedbackRR").get('_x'),plot_data.get("FeedbackRR").get('_y'),plot_data.get("FeedbackRR").get('_z'), marker='s', label="Feedback RR")
#ax.scatter(plot_data.get("IdealSJF").get('_x'),plot_data.get("IdealSJF").get('_y'),plot_data.get("IdealSJF").get('_z'), marker=7, label ="Ideal SFJ")
#ax.scatter(plot_data.get("SJF").get('_x'),plot_data.get("SJF").get('_y'),plot_data.get("SJF").get('_z'), marker='o', label = "SJF")
ax.set_xlabel('Mean CPU time')
ax.set_ylabel('Mean IO Time')
ax.set_zlabel('Turnaround time')
plt.legend()
plt.show()
"""


def aggregate_data2(experiment):
    scheduler_types = ["Fcfs","IdealSJF", "FeedbackRR","RR","SJF"]
    outputs = []
    #Need this to be (Type, Data)
    output_paths = [f"{experiment}/{f}" for f in os.listdir(experiment) if f.endswith(".out")]
    for path in output_paths:
        scheduler = "ERROR"
        input_sched = "ERROR"
        #Want this to be in order so RR check comes after FeedbackRR, as it is its subsrtring
        for type in scheduler_types:
            if type in path:
                scheduler = type
                break
        cleaned_string = re.sub(r'inpt\d+\.out', '', path).replace(f"{experiment}/","")
        match = re.search(r'([A-Za-z]+\d+)', cleaned_string)
        input_sched = match.group(1)

        if (input_sched == "ERROR" or scheduler == "ERROR"):
            raise RuntimeError(f"Error Extracting Data:{path}")
        outputs.append((scheduler,input_sched,convert_doc(path)))
    return outputs

exp2_out = aggregate_data2("experiment2")

scheduler_prps = [f for f in os.listdir("experiment2") if f.endswith("_.prp")]
interrupt_times = {}
for prp in scheduler_prps:
    with open(f"experiment2/{prp}", "r") as file:
        lines = file.readlines()
        interrupt_time = int(lines[3].replace("interruptTime=",""))
    interrupt_times[prp.replace("_.prp","")] = interrupt_time

exp2_averages=[]

for datapoint in exp2_out:
    num_proc = len(datapoint[2])
    response_time = turnaround_time =  blocked_time = waiting_time = 0
    means = {}
    for process in datapoint[2]:
        response_time += process.get("responseTime")
        turnaround_time += process.get("turnaroundTime")
        blocked_time += process.get("blockedTime")
        waiting_time += process.get("waitingTime")
    means['responseTime'] = response_time / num_proc
    means['turnaroundTime'] = turnaround_time / num_proc
    means['blockedTime'] = blocked_time / num_proc
    means['waitingTime'] = waiting_time /num_proc
    exp2_averages.append((datapoint[0],datapoint[1],means))


algorithms = ['Fcfs', 'RR', 'FeedbackRR', 'SJF', 'IdealSJF']
data2 = {alg: [] for alg in algorithms}  # Dictionary to hold extracted data for each algorithm
# Extracting data for each algorithm
for alg in algorithms:
    data2[alg] = [f for f in exp2_averages if f[0] == alg]

# Setting up variables for 3D plotting data
plot_data2 = {alg: {'_x': [], '_y': []} for alg in algorithms}

# Populate the plotting data
for alg in algorithms:
    for mean in data2[alg]:
        if interrupt_times.get(mean[1]):
            plot_data2[alg]['_x'].append(interrupt_times.get(mean[1]))
            plot_data2[alg]['_y'].append(mean[2].get('turnaroundTime'))

for alg in plot_data2:
    combined = sorted(zip(plot_data2[alg]['_x'], plot_data2[alg]['_y']))
    plot_data2[alg]['_x'], plot_data2[alg]['_y'] = zip(*combined) if combined else ([], [])

# Optionally, print data for verification
#for alg in algorithms:
#    print(f"{alg} x-values: {plot_data2[alg]['_x']}")
#    print(f"{alg} y-values: {plot_data2[alg]['_y']}")

mean_plot2 = {alg: {'_x': [], '_y': []} for alg in algorithms}
for alg in algorithms:
    x = plot_data2.get(alg).get("_x")
    y = plot_data2.get(alg).get("_y")
    new_x = []
    new_y = []
    current_x = x[0]
    sum_y = 0
    count = 0

    for x, y in zip(x, y):
        if x == current_x:
            sum_y += y
            count += 1
        else:
            new_x.append(current_x)
            new_y.append(sum_y / count)
            current_x = x
            sum_y = y
            count = 1

    # Add the last set of values
    new_x.append(current_x)
    new_y.append(sum_y / count)
    mean_plot2[alg]["_x"] = new_x
    mean_plot2[alg]["_y"] = new_y
    
print(plot_data2)

plt.figure()
##

plt.plot(plot_data2["Fcfs"]['_x'],plot_data2["Fcfs"]["_y"],label="FCFS")
plt.plot(plot_data2["RR"]['_x'],plot_data2["RR"]["_y"],label="RR")
plt.plot(plot_data2["IdealSJF"]['_x'],plot_data2["IdealSJF"]["_y"],label="IdealSJf")
plt.plot(plot_data2["FeedbackRR"]['_x'],plot_data2["FeedbackRR"]["_y"],label="FeedbackRR")
plt.plot(plot_data2["SJF"]['_x'],plot_data2["SJF"]["_y"],label="SJF")
plt.xlabel("Interrupt Time")
plt.ylabel("Turnaround Time")
plt.title("Scheduler Turnaround Time as Interrupt time increases")
plt.legend()
plt.show()


loads = ["Light","Average","Heavy"]

def aggregate_data3(experiment,class_boundaries):
    scheduler_types = ["Fcfs","IdealSJF", "FeedbackRR","RR","SJF"]

    outputs = []
    #Need this to be (Type, Data)
    output_paths = [f"{experiment}/{f}" for f in os.listdir(experiment) if f.endswith(".out")]
    for path in output_paths:
        scheduler = "ERROR"
        load = "ERROR"
        #Want this to be in order so RR check comes after FeedbackRR, as it is its subsrtring
        for type in scheduler_types:
            if type in path:
                scheduler = type
                break
        match = re.search(r'inpt(\d+)', path)
        input_num = int(match.group(1))
        for num,boundary in enumerate(class_boundaries):
            if (input_num in boundary):
                load = loads[num]
        if (load == "ERROR" or scheduler == "ERROR"):
            raise RuntimeError(f"Error Extracting Data:{path}")
        outputs.append((scheduler,load,convert_doc(path)))
    return outputs


class_boundaries = [range(0,10),range(10,20),range(20,30)]
exp3_out = aggregate_data3("experiment3",class_boundaries)

inpt_files_exp3 = [f for f in os.listdir("experiment3") if re.search(r'\d+\.prp$', f)]
inpt_loads = {}

for prp in inpt_files_exp3:
    match = re.search(r'inpt(\d+)', prp)
    input_num = int(match.group(1))
    for num,boundary in enumerate(class_boundaries):
            if (input_num in boundary):
                load = loads[num]
    inpt_loads[prp.replace(".prp","")] = load


exp3_averages=[]
for datapoint in exp3_out:
    num_proc = len(datapoint[2])
    resp = 0
    means = {}
    for process in datapoint[2]:
        resp += process.get("responseTime")
    resp = resp / num_proc
    exp3_averages.append((datapoint[0],datapoint[1],resp))
exp3_labelled = []
for datapoint in exp3_out:
    responses = []
    for proc in datapoint[2]:
        responses.append(proc.get("responseTime"))
    exp3_labelled.append((datapoint[0],datapoint[1],responses))


algorithms = ['Fcfs', 'RR', 'FeedbackRR', 'SJF', 'IdealSJF']


data3 = {load: {alg: [] for alg in algorithms} for load in loads}

for load in loads:
    for alg in algorithms:
        for sched,_load,responses in exp3_labelled:
            if (sched == alg and load == _load):
                data3[load][alg].extend(responses)

data3_av = {load: {alg: [] for alg in algorithms} for load in loads}  # Dictionary to hold extracted data for each algorithm
# Extracting data for each algorithm

for load in loads:
    for alg in algorithms:
        data3_av[load][alg] = [f[2] for f in exp3_averages if f[1] == load and f[0] == alg]

data3_means = {load: {alg: [] for alg in algorithms} for load in loads}
for load in loads:
    for alg in algorithms:
        data3_means[load][alg] = np.mean([data3.get(load).get(alg)])

import matplotlib.pyplot as plt
import numpy as np


x = np.arange(len(algorithms))  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots(layout='constrained')

for i, load in enumerate(loads):
    # Extract mean response times for each algorithm under the current load
    means = [data3_means[load][alg] for alg in algorithms]
    
    # Calculate the offset for the bars to be placed side by side
    offset = width * i
    
    # Create bars for each load
    rects = ax.bar(x + offset, means, width, label=f'{load}')
    
    # Optionally add labels to each bar
    ax.bar_label(rects, padding=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean Response Time')
ax.set_title('Mean Response Time by Scheduler and Load')
ax.set_xticks(x + width, algorithms)
ax.set_ylim(0, max(max(data3_means[load][alg] for alg in algorithms) for load in loads) + 10)  # adjust ylim based on your data
ax.legend()

plt.show()

"""