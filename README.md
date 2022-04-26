# Federated Learning from noise

This code is an implementation to do Federated Learing (FL) from noise using Knowledge Distillation (KD)

**Motivation**: FL benefits the privacy concern when training a machine learning models by sharing client weights instead of client datasets. However, recently, some research works claim that sharing weights may not be ideal due to inverse attacks. To this ends, we attempt to do FL by utilizing KD and noise image.

**Overview of proposed method**


## Install Requirements:
```pip3 install -r requirements.txt```

  
## Prepare Dataset: 
* To generate *non-iid* **Mnist** Dataset following the Dirichlet distribution D(&alpha;=0.1) for 20 clients, using 50% of the total available training samples:
<pre><code>cd FedGen/data/Mnist
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 0.5 --alpha 0.1 --n_user 20
### This will generate a dataset located at FedGen/data/Mnist/u20c10-alpha0.1-ratio0.5/
</code></pre>
    

- Similarly, to generate *non-iid* **EMnist** Dataset, using 10% of the total available training samples:
<pre><code>cd FedGen/data/EMnist
python generate_niid_dirichlet.py --sampling_ratio 0.1 --alpha 0.1 --n_user 20 
### This will generate a dataset located at FedGen/data/EMnist/u20-letters-alpha0.1-ratio0.1/
</code></pre> 

## Run Experiments: 

There is a main file "main.py" which allows running all experiments.
There are some done experiments for this project, including:

##### DO FL on MNIST using EMNIST (out-of-domain dataset)
```bash scripts/run_G_EMNIST_MNIST0.1.sh```

##### DO FL on MNIST using client dataset (in-of-domain dataset)
```bash scripts/run_Indomain_MNIST0.1.sh```

##### DO FL on MNIST using noise
```bash scripts/run_G_noise_MNIST0.1.sh```

## Highlight result


