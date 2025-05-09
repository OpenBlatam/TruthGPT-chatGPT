Algorithm 1: Synthesize table-tasks for table-tuning
input :A corpus of diverse real tables C, a set of table-task types S
output :Diverse synthesized table-tasks 𝐴 = {(𝐼𝑛𝑠,𝑇 ,𝐶)}
1 𝐷 ← {}, 𝐴 ← {}
2 foreach 𝑇 ∈ C, 𝑆 ∈ S do
3 (𝐼𝑛𝑠,𝑇 ,𝐶) ← Synthesize-Table-Task(𝑆,𝑇 ) // (Section 4.2)
4 𝐷 ← 𝐷 ∪ (𝐼𝑛𝑠,𝑇 ,𝐶)
5 foreach (𝐼𝑛𝑠,𝑇 ,𝐶) ∈ 𝐷 do
6 𝐼𝑛𝑠′ ← Augment-Instruction(𝐼𝑛𝑠) // (Section 4.3)
7 𝑇
′ ← Augment-Table(𝑇 ) // (Section 4.3)
8 𝐶
′ ← Augment-Completion(𝐶) // (Section 4.3)
9 𝐴 ← 𝐴 ∪ (𝐼𝑛𝑠′
,𝑇 ′
,𝐶′
)
10 return �


Foundations

Using the synthesis-then-augment approach in Algorithm 1 as described in previous sections,
we produce diverse table-tasks 𝐴 = {(𝐼𝑛𝑠,𝑇 ,𝐶)}. We can now continue to train language models
such as GPT, using serialized (𝐼𝑛𝑠,𝑇 ) as the “prompt” (we will explore different ways to serialize
𝑇 in our experiments), and 𝐶 as the “target completion” that we want language models to learn
from (by minimizing language-modeling loss subject to regularization). This continues to change a language-model weights until it “fits” the given table-tasks in our training data. We refer to this process as table-tuning (analogous to instruction-tuning in NLP).


# Papers

https://dl.acm.org/doi/pdf/10.1145/3654979

https://github.com/microsoft/Table-GPT

https://arxiv.org/pdf/2401.02384