# Revolutionizing-Wireless-Modeling-and-Simulation-with-Network-Oriented-LLMs
## Quick Start
### Install required packages
```
pip3 install -r requirements.txt
```
### Create `.env`  
Please create a file named `.env` in the `/RAG` directory and add the following content. Remember to replace `<YOUR_API_KEY>` with your own openai api key and `<YOUR_COHERE_KEY>` with your own cohere api key.  
```
base_url=https://drchat.xyz
api_key=<YOUR_API_KEY>
embedding_model=text-embedding-3-small
llm=gpt4-1106-preview
vectordb=sionna_db
evaluator=gpt4-32k
cohere_key=<YOUR_COHERE_KEY>
reranker=rerank-english-v3.0
```

### Configure the OpenAI API in data generation and fine-tuning process
For data generation, open the `parallel_request.py` file and locate the line:

```python
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
```
Replace `<YOUR_API_KEY>` with your own OpenAI API key, as specified in the `Create .env` section.

For fine-tuning, open the `model_creation.ipynb` and `model_evaluation.ipynb` notebooks. Locate the line:
```
os.environ["AZURE_OPENAI_KEY"] = "<YOUR_AZURE_API_KEY>"
```
Replace <YOUR_AZURE_API_KEY> with your own Azure OpenAI key for fine-tuning.  

## Run the code
The project comprises three main sections: data generation, instruction fine-tuning, and RAG.

1. **Data Generation**: This section involves crawling, processing, and generating instruction-answer pairs for the Sionna dataset.
2. **Instruction Fine-Tuning**: This section focuses on fine-tuning GPT models using the instruction-answer pairs generated in the data generation section.
3. **RAG**: This section enhances the answer generation process using Retrieval-Augmented Generation (RAG).

For detailed experimental procedures and results, please refer to the associated paper.  

To run the data generation section code, please navigate to the `/DataPreparation` directory and execute the `process.sh` script. 

In the `markdown` and `chunk` directories, you will find:
- Markdown files: These files contains the information crawled by crawler.py from the [Sionna official website](https://nvlabs.github.io/sionna/).
- Cleaned and chunked JSONL files: These files are processed using the following files from the RAG directory:
  - `/code/preprocess/clean.py`
  - `/code/preprocess/chunk.py`

To execute the fine-tuning process, please follow the code in `model_creation.ipynb`. The Sionna dataset is partitioned into training, validation, and testing datasets using `split_IA_total_to_datasets.py`.

To execute the RAG process, begin by referring to `run.sh` for data cleaning, chunking, and vector database creation. Subsequently, refer to `run_gpt4-128k.sh` for the retriever and reranker steps.


## Demo

### Here is a demo video demonstrating how to run the network simulation using natural language with our NetLLM.  




https://github.com/user-attachments/assets/76e26b2c-e36f-4437-adc9-644620fc57c0




### Complete screenshots of the demo are provided below the video.  
![image](https://github.com/user-attachments/assets/1d4f4d7b-fb55-4f6b-bc36-75ff491a361c)




### (a) Question from User Case Demo: 
#### How can you configure and compute propagation paths in a Munich scene using Sionna’s compute_paths function? Include details on setting up planar arrays for transmitters and receivers, adding a transmitter and a receiver to the scene, computing the paths, and visualizing the paths in a preview.
#### Answer:
Our LLM provides step-by-step explanations with code snippets as shown in the Figure below:
![img](https://github.com/ak-maker/sionna-LLMs/assets/64005655/7fa00b4e-8ff0-4fee-9576-58124c4d6903)

We combine the code as follows:  
```python
import numpy as np
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# Step 2: Load the Munich scene
scene = load_scene(sionna.rt.scene.munich)

# Step 3: Configure Antenna Arrays for Transmitters and Receivers
# Configure the antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")

# Configure the antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="dipole", polarization="cross")
tx = Transmitter(name="tx1", position=[8.5, 21, 27]) # Position needs to be adjusted based on the Munich scene coordinates
scene.add(tx)

# Create and add a receiver to the scene
rx = Receiver(name="rx1", position=[45, 90, 1.5], orientation=[0, 0, 0]) # Adjust position and orientation as needed
scene.add(rx)

# Make the transmitter point towards the receiver
tx.look_at(rx.position)

# Step 5: Set Up the Carrier Frequency
scene.frequency = 2.14e9 # 2.14 GHz; adjust as necessary
scene.synthetic_array = True # Optimized computations for large arrays

# Step 6: Compute Propagation Paths
paths = scene.compute_paths(max_depth=5, num_samples=1e6, los=True, reflection=True, diffraction=True, scattering=True) # Include all path interactions

# Step 7: Visualization
scene.preview(paths=paths, show_devices=True, show_paths=True)
```

Simulation Result:  

![usercase](https://github.com/ak-maker/sionna-LLMs/assets/64005655/87682b6f-645b-4398-ac3e-6720b917963d)




### (b) Question 1: 
#### Explain how to call the Scene component of the ray tracer using the Sionna Python package. 
#### Answer:
The code is as follows:
```python
# Import the necessary module
import sionna

# Import the function to load scenes
from sionna.rt import load_scene, Scene

# Load an example scene
scene = load_scene(sionna.rt.scene.etoile)

# If the scene name variable is not predefined and directly accessible,
# you could instead navigate through the sionna.rt module based on the
# structure of the Sionna package. This detail is assumed from
# the example usage pattern and might need adjustment based on the
# actual package structure and naming conventions.
scene.preview()
```

Simulation Result:  

![question1](https://github.com/ak-maker/sionna-LLMs/assets/64005655/bc5863ca-1ad8-40d8-8548-4293a3eec080)


### (c) Question 2: 
#### I have installed Sionna. How to perform raytracing?
#### Answer:
The code is as follows:
```python
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
# Load an example scene (e.g., Munich scene)
scene = load_scene(sionna.rt.scene.munich)
scene.tx_array = PlanarArray(num_rows=8, 
                            num_cols=2,
                            vertical_spacing=0.7,
                            horizontal_spacing=0.5,
                            pattern="tr38901",
                            polarization="VH")
# Configure antenna array for all receivers in the scene
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="cross")
tx = Transmitter(name="tx",
                 position=[8.5, 21, 27],
                 orientation=[0, 0, 0])
scene.add(tx)
# Create a receiver object and add it to the scene
rx = Receiver(name="rx",
              position=[45, 90, 1.5],
              orientation=[0, 0, 0])
scene.add(rx)
# Compute paths with specific parameters
paths = scene.compute_paths(max_depth=3,
                            method="fibonacci",
                            num_samples=int(1e6),
                            los=True,
                            reflection=True,
                            diffraction=False,
                            scattering=False,
                            check_scene=True)
scene.preview(paths=paths, resolution=[1000, 600])
```

Simulation Result:  

![question2](https://github.com/ak-maker/sionna-LLMs/assets/64005655/29f4eb87-1ffa-4720-97cb-bb5eaf54bc6f)


### (d) Question 2 follow-up: 
#### How to compute the channel impulse response accordingly?
#### Answer:
The code is as follows:
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

a, tau = paths.cir()
# Provided compute_gain function
def compute_gain(a, tau):
    """Compute |H(f)|^2 at f = 0 where H(f) is the baseband channel frequency response"""
    a = tf.squeeze(a, axis=-1)
    h_f_2 = tf.math.abs(tf.reduce_sum(a, axis=-1))**2
    h_f_2 = tf.where(h_f_2==0, 1e-24, h_f_2)
    g_db = 10*np.log10(h_f_2)
    return tf.squeeze(g_db)

print(a.shape)

n = 400
plt.figure()
plt.stem(tau[0,0,0,:]/1e-9, 10*np.log10(np.abs(a[0,0,0,0,0,:,0])**2), basefmt=" ")
plt.title(f"Angle of receiver")
plt.xlabel("Delay (ns)")
plt.ylabel("|a|^2 (dB)")
plt.show()

# # Example usage of compute_gain (optional here) for receiver at index n
# g_db_n = compute_gain(a[n:n+1], tau[n:n+1])
# print(f"Computed gain for the receiver {n}: {g_db_n.numpy()} dB")
```

Simulation Result:  

![img_1](https://github.com/ak-maker/sionna-LLMs/assets/64005655/39c5812a-84af-455d-8128-451e55cadafc)


### (e) Question 3: 
####  Create a 2D visualization of the coverage map of muniche scene.
#### Answer:
The code is as follows:
```python
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera
# Assuming you have the Sionna package installed and the necessary dependencies,
# load the Munich scene from the Sionna Ray Tracing module
scene = load_scene(sionna.rt.scene.munich)
# Configure the antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=8,
                             num_cols=2,
                             vertical_spacing=0.7,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="VH")
# Configure the antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="cross")
# Add a transmitter
tx = Transmitter(name="tx",
                 position=[8.5, 21, 30],  # Position adjusted for the example
                 orientation=[0, 0, 0])
scene.add(tx)
tx.look_at([40, 80, 1.5])
# Compute the coverage map with specified parameters
cm = scene.coverage_map(cm_cell_size=[1.,1.],  # Size of each cell in meters
                        num_samples=int(10e6))  # Number of rays to trace
# Fit the camera to view the whole scene. This step configures the camera
# for visualization and is adapted from the context to fit within this example
tx_pos = scene.transmitters["tx"].position.numpy()
bird_pos = tx_pos.copy()
bird_pos[-1] += 100  # Height above the transmitter
scene.add(Camera("birds_view", position=bird_pos, look_at=tx_pos))
# Visualize the coverage map in 2D for the first transmitter
cm.show(tx=0)
```

Simulation Result:  

![img_2](https://github.com/ak-maker/sionna-LLMs/assets/64005655/ec9d9d4c-1f67-49fa-99a0-1bd8bd44d358)

### (f) 3D View of the Coverage Map:
#### Answer:
The code is as follows:
```python
# Open 3D preview (only works in Jupyter notebook)
scene.preview(coverage_map=cm)
```

Simulation Result:  

![last_question7](https://github.com/ak-maker/sionna-LLMs/assets/64005655/be2597cb-62f4-4399-844d-130ea170572c)


### Authors
The NetLLM module and simulator is developed and maintained by:

 1. Jiewen Liu (NCSU)

 2. Zhiyuan Peng (NCSU)

 3. Yuchen Liu (NCSU)

### Citation
@inproceedings{Liu2024revolu,

  title={Revolutionizing Wireless Modeling and Simulation with Network Oriented LLMs},
  
  author={Liu, J. and Peng, Z. and Xu, D. and Liu, Y.},
  
  booktitle={IEEE/ACM xxx},
  
  year={2024}
  
}

