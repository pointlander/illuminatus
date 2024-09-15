// Copyright 2024 The Illuminatus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"runtime"

	"github.com/alixaxel/pagerank"
)

const (
	// Symbols
	Symbols = ('z' - 'a' + 1) + ('Z' - 'A' + 1) + 3
	// Size is the link size
	Size = 16
	// Input is the network input size
	Input = Size + 2*Size
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// Scale is the scale of the search
	Scale = 33 //48 96
	// SymbolsCount is the number of unique symbols in a puzzle
	SymbolsCount = 4
	// Samples is the number of samplee
	Samples = Scale * (Scale - 1) / 2
)

var (
	// To converts to a code
	To = make(map[rune]int, Symbols)
	// From converts from a code
	From = make(map[int]rune, Symbols)
)

func init() {
	index := 0
	for i := 'a'; i <= 'z'; i++ {
		To[i] = index
		From[index] = i
		index++
	}
	for i := 'A'; i <= 'Z'; i++ {
		To[i] = index
		From[index] = i
		index++
	}
	i := '^'
	To[i] = index
	From[index] = i
	index++
	i = '$'
	To[i] = index
	From[index] = i
	index++
	i = ' '
	To[i] = index
	From[index] = i
	index++
}

// Puzzle is a puzzle
type Puzzle string

// Q is the query portion of a puzzle
func (p Puzzle) Q() []int {
	q := []rune(p)
	last := len(q) - 1
	query := make([]int, last)
	for key, value := range q[:last] {
		query[key] = To[value]
	}
	return query
}

// A is the answer portion of a puzzle
func (p Puzzle) A() int {
	q := []rune(p)
	return To[q[len(q)-1]]
}

var Puzzles = []Puzzle{
	"^abcdabcdabcdabcda",
	"^abcdabcdabcdabcdab",
	"^abcdabcdabcdabcdabc",
	"^abcdabcdabcdabcdabcd",
	"^abcddcbaabcddcbaabcddcbaabcd",
	"^aabbccddaabbccddaabbccd",
	"^aabbccddaabbccddaabbccdd",
}

// Matrix is a float64 matrix
type Matrix struct {
	Cols int
	Rows int
	Data []complex128
}

// NewMatrix creates a new float64 matrix
func NewMatrix(cols, rows int, data ...complex128) Matrix {
	if data == nil {
		data = make([]complex128, 0, cols*rows)
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
		Data: make([]complex128, cols*rows),
	}
}

// RandomMatrix is a random matrix
type RandomMatrix struct {
	Cols int
	Rows int
	Seed int64
}

// NewRandomMatrix creates a new gaussian random matrix
func NewRandomMatrix(cols, rows int, seed int64) RandomMatrix {
	return RandomMatrix{
		Cols: cols,
		Rows: rows,
		Seed: seed,
	}
}

// Sample generates a matrix from a gaussian distribution
func (g RandomMatrix) Sample() Matrix {
	rng := rand.New(rand.NewSource(g.Seed))
	factor := math.Sqrt(2.0 / float64(g.Cols))
	sample := NewMatrix(g.Cols, g.Rows)
	for i := 0; i < g.Cols*g.Rows; i++ {
		a := rng.NormFloat64() * factor
		//b := rng.NormFloat64() * factor
		sample.Data = append(sample.Data, complex(a, a))
	}
	return sample
}

// Dot computes the dot product
func Dot(x, y []complex128) (z complex128) {
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
		Data: make([]complex128, 0, m.Rows*n.Rows),
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

// Sample is a sample
type Sample struct {
	A      RandomMatrix
	B      RandomMatrix
	Order  [2]RandomMatrix
	Symbol [2]RandomMatrix
	S      int
	Ranks  []float64
}

// Search searches for a symbol
func Search(s int, seed int64) []Sample {
	length := len(Puzzles[s].Q()) + 2
	rng := rand.New(rand.NewSource(seed))
	projections := make([]RandomMatrix, Scale)
	for i := range projections {
		seed := rng.Int63()
		if seed == 0 {
			seed = 1
		}
		projections[i] = NewRandomMatrix(Input, Input, seed)
	}
	index := 0
	samples := make([]Sample, Samples)
	for i := 0; i < Scale; i++ {
		for j := i + 1; j < Scale; j++ {
			samples[index].A = projections[i]
			samples[index].B = projections[j]
			seed := rng.Int63()
			if seed == 0 {
				seed = 1
			}
			order := NewRandomMatrix(Size, length, seed)
			seed = rng.Int63()
			if seed == 0 {
				seed = 1
			}
			symbol := NewRandomMatrix(Size, Symbols, seed)
			samples[index].Order[0] = order
			samples[index].Symbol[0] = symbol
			seed = rng.Int63()
			if seed == 0 {
				seed = 1
			}
			order = NewRandomMatrix(Size, length, seed)
			seed = rng.Int63()
			if seed == 0 {
				seed = 1
			}
			symbol = NewRandomMatrix(Size, Symbols, seed)
			samples[index].Order[1] = order
			samples[index].Symbol[1] = symbol
			samples[index].S = index % SymbolsCount
			index++
		}
	}

	done := make(chan bool, 8)
	process := func(sample *Sample) {
		var phi [2]Matrix
		input := Puzzles[s].Q()
		phi[0] = NewZeroMatrix(Input, length+3)
		phi[1] = NewZeroMatrix(Input, length+3)
		for i := range phi {
			phi := &phi[i]
			order := sample.Order[i].Sample()
			a, b := 0, 1
			jj := phi.Rows - 4
			for j := 0; j < jj; j++ {
				x, y := (j+a)%phi.Rows, (j+b)%phi.Rows
				copy(phi.Data[j*Input+Size:j*Input+Size+Size],
					order.Data[x*Size:(x+1)*Size])
				copy(phi.Data[j*Input+Size+Size:j*Input+Size+2*Size],
					order.Data[(y)*Size:(y+1)*Size])
				a, b = b, a
			}
			for j := jj; j < jj+3; j++ {
				x, y := (jj-1+b)%phi.Rows, (jj-1+a)%phi.Rows
				copy(phi.Data[j*Input+Size:j*Input+Size+Size],
					order.Data[x*Size:(x+1)*Size])
				copy(phi.Data[j*Input+Size+Size:j*Input+Size+2*Size],
					order.Data[(y)*Size:(y+1)*Size])
			}
			if x := jj + a; x < order.Rows {
				jj += 3
				copy(phi.Data[jj*Input+Size:jj*Input+Size+Size],
					order.Data[x*Size:(x+1)*Size])
			} else if y := jj + b; y < order.Rows {
				jj += 3
				copy(phi.Data[jj*Input+Size+Size:jj*Input+Size+2*Size],
					order.Data[(y)*Size:(y+1)*Size])
			} else {
				panic("shouldn't be here")
			}
			syms := sample.Symbol[i].Sample()
			index := 0
			for i := 0; i < len(input); i++ {
				symbol := syms.Data[Size*input[i] : Size*(input[i]+1)]
				copy(phi.Data[index:index+Input], symbol)
				index += Input
			}
			for i := 0; i < 4; i++ {
				symbol := syms.Data[Size*To[rune(i)] : Size*(To[rune(i)]+1)]
				copy(phi.Data[index:index+Input], symbol)
				index += Input
			}
			{
				symbol := syms.Data[Size*To['$'] : Size*(To['$']+1)]
				copy(phi.Data[index:index+Input], symbol)
			}
		}
		/*for j := 0; j < phi.Rows; j++ {
			for i := 0; i < phi.Cols; i += 2 {
				phi.Data[j*phi.Cols+i] += complex(math.Sin(float64(j)/math.Pow(10000, 2*float64(i)/Size)), 0)
				phi.Data[j*phi.Cols+i+1] += complex(math.Cos(float64(j)/math.Pow(10000, 2*float64(i)/Size)), 0)
			}
		}*/
		query := sample.A.Sample()
		key := sample.B.Sample()
		q := query.MulT(phi[0])
		k := key.MulT(phi[1])
		sample.Ranks = PageRank(q, k)
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

	return samples
}

// Illuminatus
func Illuminatus(s int, seed int64) int {
	rng := rand.New(rand.NewSource(seed))
	fmt.Println(string(Puzzles[s]))
	seed = rng.Int63()
	if seed == 0 {
		seed = 1
	}
	samples := Search(s, seed)

	input := Puzzles[s].Q()
	min, result := math.MaxFloat64, 0
	for symbol := 0; symbol < SymbolsCount; symbol++ {
		indexes := make([]int, 0, 8)
		for key, value := range input {
			if value == symbol {
				indexes = append(indexes, key)
			}
		}
		sum, count := 0.0, 0.0
		for sample := range samples {
			/*if samples[sample].S != symbol {
				continue
			}*/
			ranks := samples[sample].Ranks
			for _, index := range indexes {
				sum += ranks[index]
				count++
			}
		}
		average := sum / count
		variance := 0.0
		for sample := range samples {
			/*if samples[sample].S != symbol {
				continue
			}*/
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

func main() {
	seed := int64(2)
	histogram := [7][4]int{}
	for e := 0; e < 32; e++ {
		correct := 0
		for i := range Puzzles {
			result := Illuminatus(i, seed)
			histogram[i][result]++
			if result == Puzzles[i].A() {
				correct++
			}
		}
		fmt.Println("correct", correct)
		seed++
	}
	correct := 0
	for i := range histogram {
		max, index := 0, 0
		for key, value := range histogram[i] {
			if value > max {
				max, index = value, key
			}
		}
		status := "incorrect"
		if index == Puzzles[i].A() {
			status = "correct"
			correct++
		}
		fmt.Println(histogram[i], status, Puzzles[i])
	}
	fmt.Printf("%d/%d correct\n", correct, len(histogram))
}
