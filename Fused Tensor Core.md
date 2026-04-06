# Fused Tensor Core: A Hardware–Software  Co-Design for Efficient Execution of  Attentions on GPUs

## Problem
Implementing attention layers on GPUs with tensor cores (TCs) using matrixmultiply and accumulate (MMA) operations is suboptimal as the attention layer incurs **an excessively large-memory footprint** and **significant computational complexity**, especially with a higher number of input elements.

TC architecture is designed based on the characteristics of **convolutional and fully connected layers**. However, **the core operators within attention layers exhibit different bottlenecks on the memory hierarchy** compared to convolutional and fully connected layers.

Furthermore, certain operations within attention layers **run on standard CUDA cores which prolong the execution of Transformers**.

## Related Work
1. Buffering intermediate tensors[1]
- Advantages: fast
- Limitations: Limitations on the sequence length because of the constrained capacity of on-chip memory.

2. Utilizing off-chip memory to store intermediate tensors[2]
- Advantages: larger capacity
- Limitations: substantial latency

3. Fewer memory access
- Advantages: no limitations from on-chip memory with low latency


## Methodology
### fusion mechanism
1. Efficient Algorithm for Attention Layer
- conventional approach

- S: $S = Q \times K^T$ 1 CUDA Kernel
- P: $P=softmax(S)$ 1 CUDA Kernel
- O: $O=P \times V$ 1 CUDA Kernel

- novel approach
![alt text](/asset/image.png)
Assuming that V is decomposed into two vectors V1 and V2 such that V = [V1 V2], we can break down the computation of softmax(V) as follows:
![alt text](/asset/image-1.png)

This formulation **allows softmax to be computed in a blockwise manner**, enabling efficient memory usage and reducing numerical instability by normalizing in smaller chunks before computing the final result.

2. Fused TC
- Algorithm
![alt text](/asset/image2.png)
![alt text](/asset/image3.png)
![alt text](/asset/image4.png)
  
### offloading certain non-MMA operations within the attention layer from standard CUDA cores to TCs.

![alt text](/asset/image5.png)


What can I use?
fused kernels