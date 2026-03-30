# auto-layout-system-design

## 问题1：layout如何定义？现在由于layout导致的瓶颈问题主要存在哪些类型？

### layout的定义
数据在GPU内存中的排放次序与线程存取顺序的映射关系

### 问题
1. 量化过程中由于pack操作与线程读取解量化后的数据排布不一致，导致的读取延迟、重排开销以及可能引发的错误。

## 问题2：导致layout problem的操作类型
> Tensor Core mma指令与pack操作的联合效果

## 问题3：相关研究


## References
[1] D. Du, S. Cao, J. Cheng, L. Mai, T. Cao, and M. Yang, 
“BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache,” 
arXiv preprint arXiv:2503.18773, 2025.
