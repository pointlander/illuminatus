// Copyright 2024 The Illuminatus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arc

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"

	"github.com/pointlander/illuminatus/vector"

	"github.com/alixaxel/pagerank"
)

// Example is a learning example
type Example struct {
	Input  [][]byte `json:"input"`
	Output [][]byte `json:"output"`
}

// Set is a set of examples
type Set struct {
	Test  []Example `json:"test"`
	Train []Example `json:"train"`
}

// Load loads the data
func Load() []Set {
	dirs, err := os.ReadDir("ARC-AGI/data/training/")
	if err != nil {
		panic(err)
	}
	sets := make([]Set, len(dirs))
	for i, dir := range dirs {
		data, err := os.ReadFile("ARC-AGI/data/training/" + dir.Name())
		if err != nil {
			panic(err)
		}
		err = json.Unmarshal(data, &sets[i])
		if err != nil {
			panic(err)
		}
	}
	fmt.Println("loaded", len(sets))
	test, train := 0, 0
	for _, set := range sets {
		test += len(set.Test)
		train += len(set.Train)
	}
	fmt.Println("test", test)
	fmt.Println("train", train)
	return sets
}

const (
	// Symbols
	Symbols = 32
	// Size is the link size
	Size = 16
	// Input is the network input size
	Input = 2 * Size
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// Scale is the scale of the search
	Scale = 33 //48 96
	// SymbolsCount is the number of unique symbols in a puzzle
	SymbolsCount = 32
	// Samples is the number of samples
	Samples = Scale * (Scale - 1) / 2
)

// Puzzle is a puzzle
type Puzzle []byte

// Q is the query portion of a puzzle
func (p Puzzle) Q() []byte {
	return []byte(p)
}

// Matrix is a float32 matrix
type Matrix struct {
	Cols int
	Rows int
	Data []float32
}

// NewMatrix creates a new float64 matrix
func NewMatrix(cols, rows int, data ...float32) Matrix {
	if data == nil {
		data = make([]float32, 0, cols*rows)
	}
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// NewZeroMatrix creates a new float64 matrix of zeros
func NewZeroMatrix(cols, rows int) Matrix {
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float32, cols*rows),
	}
}

// Random is a gaussian distribution
type Random struct {
	Mean   float32
	Stddev float32
}

// RandomMatrix is a random matrix
type RandomMatrix struct {
	Cols int
	Rows int
	Rand []Random
	Seed int64
}

// NewRandomMatrix creates a new gaussian random matrix
func NewRandomMatrix(cols, rows int, seed int64, r ...Random) RandomMatrix {
	if r == nil {
		factor := float32(math.Sqrt(2.0 / float64(cols)))
		r = make([]Random, cols*rows)
		for i := range r {
			r[i].Stddev = factor
		}
	}
	return RandomMatrix{
		Cols: cols,
		Rows: rows,
		Rand: r,
		Seed: seed,
	}
}

// Sample generates a matrix from a gaussian distribution
func (g RandomMatrix) Sample() Matrix {
	rng := rand.New(rand.NewSource(g.Seed))
	sample := NewMatrix(g.Cols, g.Rows)
	for _, v := range g.Rand {
		a := float32(rng.NormFloat64())*v.Stddev + v.Mean
		sample.Data = append(sample.Data, a)
	}
	return sample
}

// Dot computes the dot product
func Dot(x, y []float32) (z float32) {
	for i := range x {
		z += x[i] * y[i]
	}
	return z
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix) MulT(n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float32, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, Dot(mm, nn))
		}
	}
	return o
}

// Sigmoid computes the sigmoid of a matrix
func (m Matrix) Sigmoid() Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, float32(1/(1+math.Exp(-float64(value)))))
	}
	return o
}

// TanH computes the tanh of a matrix
func (m Matrix) TanH() Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		a := math.Exp(float64(value))
		b := math.Exp(-float64(value))
		o.Data = append(o.Data, float32((a-b)/(a+b)))
	}
	return o
}

// PageRank computes the page rank of Q, K
func PageRank(x, y Matrix, graph []float64) {
	offset := 0
	for i := 0; i < y.Rows; i++ {
		yy := y.Data[i*y.Cols : (i+1)*y.Cols]
		aa := float32(0.0)
		for _, v := range yy {
			aa += v * v
		}
		aa = float32(math.Sqrt(float64(aa)))
		for j := 0; j < x.Rows; j++ {
			xx := x.Data[j*x.Cols : (j+1)*x.Cols]
			bb := float32(0.0)
			for _, v := range xx {
				bb += v * v
			}
			bb = float32(math.Sqrt(float64(bb)))
			d := math.Abs(float64(vector.Dot(yy, xx) / (aa * bb)))
			graph[offset] = d
			offset++
		}
	}
}

// Sample is a sample
type Sample struct {
	A        RandomMatrix
	B        RandomMatrix
	Graph    []float64
	Ranks    []float64
	Variance float64
}

// Search searches for a symbol
func (puzzle Puzzle) Search(seed int64) []Sample {
	length := len(puzzle.Q())
	stride := 2 * length
	rng := rand.New(rand.NewSource(seed))
	projections := make([]RandomMatrix, Scale)
	for i := range projections {
		seed := rng.Int63()
		if seed == 0 {
			seed = 1
		}
		projections[i] = NewRandomMatrix(Input, Input, seed)
	}
	symbol := NewRandomMatrix(Size, Symbols, 1)
	symbols := symbol.Sample()
	index := 0
	samples := make([]Sample, Samples)
	for i := 0; i < Scale; i++ {
		for j := i + 1; j < Scale; j++ {
			samples[index].A = projections[i]
			samples[index].B = projections[j]
			index++
		}
	}

	done := make(chan bool, 8)
	process := func(sample *Sample) {
		q := puzzle.Q()
		input := NewZeroMatrix(Input, 2*length)
		for i := 0; i < len(q); i++ {
			index := 2 * i
			symbol := symbols.Data[Size*int(q[i]) : Size*(int(q[i])+1)]
			if index > 0 {
				copy(input.Data[(index-1)*Input:(index-1)*Input+Size], symbol)
			}
			copy(input.Data[index*Input:index*Input+Size], symbol)
			copy(input.Data[index*Input+Size:(index+1)*Input], symbol)
			if index+1 < 2*len(q) {
				copy(input.Data[(index+1)*Input+Size:(index+2)*Input], symbol)
			}
		}
		a := sample.A.Sample()
		b := sample.B.Sample()
		x := a.MulT(input)
		y := b.MulT(input)
		graph := make([]float64, y.Rows*x.Rows)
		PageRank(x, y, graph)
		sample.Graph = graph
		done <- true
	}
	flight, index, cpus := 0, 0, runtime.NumCPU()
	for flight < cpus && index < len(samples) {
		sample := &samples[index]
		go process(sample)
		index++
		flight++
	}
	for index < len(samples) {
		<-done
		flight--

		sample := &samples[index]
		go process(sample)
		index++
		flight++
	}
	for i := 0; i < flight; i++ {
		<-done
	}

	graph := pagerank.NewGraph()
	ranks := make([]float64, stride*Samples)
	offsetA := 0
	for i := 0; i < Scale-1; i++ {
		offsetB := 0
		for j := i + 1; j < Scale; j++ {
			b := samples[j]
			for k := 0; k < stride; k++ {
				for l := 0; l < stride; l++ {
					graph.Link(uint32(offsetA+k), uint32(offsetA+l), b.Graph[k*stride+l])
				}
			}
			for k := 0; k < stride; k++ {
				graph.Link(uint32(offsetA+k), uint32(offsetB+k), Samples)
				graph.Link(uint32(offsetB+k), uint32(offsetA+k), Samples)
			}
			offsetB += stride
		}
		offsetA += stride
	}
	graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
		ranks[node] = rank
	})
	index = 0
	for i := 0; i < Scale; i++ {
		for j := i + 1; j < Scale; j++ {
			begin := index * stride
			end := (index + 1) * stride
			samples[index].Ranks =
				ranks[begin:end]
			index++
		}
	}
	return samples
}

// Illuminatus
func (puzzle Puzzle) Illuminatus(seed int64) int {
	rng := rand.New(rand.NewSource(seed))
	min, result := math.MaxFloat64, 0
	seed = rng.Int63()
	if seed == 0 {
		seed = 1
	}
	samples := puzzle.Search(seed)
	input := puzzle.Q()

	for symbol := 0; symbol < SymbolsCount; symbol++ {
		indexes := make([]int, 0, 8)
		for key, value := range input {
			if value == byte(symbol) {
				indexes = append(indexes, 2*key, 2*key+1)
				if 2*key > 0 {
					indexes = append(indexes, 2*key-1)
				}
			}
		}
		sum, count := 0.0, 0.0
		for sample := range samples {
			ranks := samples[sample].Ranks
			for _, index := range indexes {
				sum += ranks[index]
				count++
			}
		}
		average := sum / count
		variance := 0.0
		for sample := range samples {
			ranks := samples[sample].Ranks
			for _, index := range indexes {
				diff := average - ranks[index]
				variance += diff * diff
			}
		}
		if variance < min {
			min, result = variance, symbol
		}
	}
	fmt.Println(result)
	return result
}

// Arc is the arc model
func Arc() {
	puzzles := Load()
	encoding := make([]Puzzle, 2*len(puzzles[0].Train)+1)
	index := 0
	for _, t := range puzzles[0].Train {
		for y, tt := range t.Input {
			for x, ttt := range tt {
				encoding[index] = append(encoding[index], ttt, byte(x), byte(y))
			}
			encoding[index] = append(encoding[index], 30)
		}
		encoding[index] = append(encoding[index], 31)
		index++
		for y, tt := range t.Output {
			for x, ttt := range tt {
				encoding[index] = append(encoding[index], ttt, byte(x), byte(y))
			}
			encoding[index] = append(encoding[index], 30)
		}
		encoding[index] = append(encoding[index], 31)
		index++
	}
	for _, t := range puzzles[0].Test[:1] {
		for y, tt := range t.Input {
			for x, ttt := range tt {
				encoding[index] = append(encoding[index], ttt, byte(x), byte(y))
			}
			encoding[index] = append(encoding[index], 30)
		}
		encoding[index] = append(encoding[index], 31)
	}
	solution := [Symbols * Symbols][Symbols]int{}
	for i := range solution {
		for j := range solution[i] {
			solution[i][j] = 1
		}
	}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 128; i++ {
		input := make(Puzzle, 0, 1)
		for j := range encoding {
			for k := 0; k < 2; k++ {
				index := rng.Intn(len(encoding[j]))
				input = append(input, encoding[j][index])
			}
		}
		index, sum := rng.Intn(len(solution)-1), 0
		for _, v := range solution[index] {
			sum += v
		}
		sample := rng.Intn(sum)
		sum = 0
		for j, v := range solution[index] {
			sum += v
			if sample < sum {
				input = append(input, byte(j))
			}
		}
		seed := rng.Int63()
		if seed == 0 {
			seed = 1
		}
		result := input.Illuminatus(seed)
		solution[index+1][result]++
	}
}
