# auto-layout-system-design

## 问题1：layout如何定义？现在由于layout导致的瓶颈问题主要存在哪些类型？

### The definition of layouts
#### Distributed layouts 
Distributed layouts is the definition of the way that tensor elements are distributed across different execution units. This includes **Blocked, Sliced, MMA, and MMA Input layouts**.[2]

#### Memory layouts
Memory layouts indicates tensor elements are stored in certain special memory.
They can be classified into Unswizzled and Swizzled layouts.[2]

- Blocked layouts are often used for contiguous memory access.
- MMA and MMA input layouts are used for the output and inputs of matrix multiplication operations. MMA layouts can be further classified according to hardware instructions they map to, such as mma and wgmma on NVIDIA GPUs, or mfma on AMD GPUs.
- Sliced layouts extract a dimension from their parent layout used as the input to a broadcast or the output of a reduction.
![alt text](/asset/image6.png)

---

### 问题
1. 量化过程中由于pack操作与线程读取解量化后的数据排布不一致，导致的读取延迟、重排开销以及可能引发的错误。

## 问题2：导致layout problem的操作类型
> Tensor Core mma指令与pack操作的联合效果

## 问题3：相关研究


## References
[1] D. Du, S. Cao, J. Cheng, L. Mai, T. Cao, and M. Yang, 
“BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache,” 
arXiv preprint arXiv:2503.18773, 2025.
[2] Keren Zhou, Mario Lezcano-Casado, Adam P. Goucher, Akhmed Rakhmati, Jeff Niu, Justin Lebar, Pawel Szczerbuk, Peter Bell, Phil Tillet, Thomas Raoux, and Zahi Moudallal. 2026. Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2. In Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS ’26), March 22–26, 2026, Pittsburgh, PA, USA. ACM, New York, NY, USA, 18 pages. https://doi.org/10.1145/3760250.3762221
