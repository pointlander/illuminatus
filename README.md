# Running
Clone the repo and then inside the repo:
```sh
go build
./illuminatus
```

# Abstract
 When a puzzle is looked at under different lights what doesn't change is the solution. The illuminatus algorithm uses page rank and variance calculations to determine what is invariant in the puzzle definition under multiple random projections.

# Experiment
The algorithm is tested with several puzzles:
```go
var Puzzles = []Puzzle{
	"^abcdabcdabcda",
	"^abcdabcdabcdabcdab",
	"^abcdabcdabcdabc",
	"^abcdabcdabcdabcd",
	"^abcddcbaabcddcbaabcddcbaabcd",
	"^aabbccddaabbccddaabbccd",
	"^aabbccddaabbccddaabbccdd",
}
```
The last symbol is the puzzle answer, and the prefix to this symbol is the puzzle query. So, "^abcdabcdabcda" has an answer of "a" and a query of "^abcdabcdabcd". The puzzle is encoded as an input matrix. Each row of the input matrix is composed of a random vector linking the row to the previous row, another random vector linking the row to the next row, and a random vector corresponding to the symbol. The last two rows of the input matrix are a symbol guess row which can be one of the symbols abcd, and finally the end symbol: $. The input matrix, I, is multiplied by two randomly chosen matrices: X = AI, Y = BI. X and Y are then multiplied using a cosine similarity normalized dot product to create an adjacency matrix. The adjacency matrix is then fed into page rank:
```go
// PageRank computes the page rank of Q, K
func PageRank(Q, K Matrix) []float64 {
	graph := pagerank.NewGraph()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		aa := complex(0.0, 0.0)
		for _, v := range K {
			aa += v * v
		}
		aa = cmplx.Sqrt(aa)
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			bb := complex(0.0, 0.0)
			for _, v := range Q {
				bb += v * v
			}
			bb = cmplx.Sqrt(bb)
			d := cmplx.Abs(Dot(K, Q) / (aa * bb))
			graph.Link(uint32(i), uint32(j), d)
		}
	}
	ranks := make([]float64, K.Rows)
	graph.Rank(1.0, 1e-9, func(node uint32, rank float64) {
		ranks[node] = rank
	})
	return ranks
}
```
The above process is computed multiple times for different randomly sampled matrices A and B and a different symbol guess. The variances of the ranks are calculated and the symbol with the lowest variance is assumed to be correct.

# Conclusion
An algorithm to solve simple one dimensional puzzles has been presented. Scaling to the larger two dimensional ARC-AGI puzzles may take a CUDA impelmentation or a machine with a large number of CPU cores. Lazy evaluation should be explored. Encoding two dimensional puzzles in the input matrix is an open problem.
