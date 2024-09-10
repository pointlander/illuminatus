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
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

const (
	// Symbols
	Symbols = ('z' - 'a' + 1) + ('Z' - 'A' + 1) + 3
	// Size is the link size
	Size = 32
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
	"^abcdabcdabcda",
	"^abcdabcdabcdabcdab",
	"^abcdabcdabcdabc",
	"^abcdabcdabcdabcd",
	"^abcddcbaabcddcbaabcddcbaabcd",
	"^aabbccddaabbccddaabbccd",
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

// GaussianRandomMatrix is a random matrix
type GaussianRandomMatrix struct {
	Cols int
	Rows int
	Seed int64
}

// NewGaussianRandomMatrix creates a new gaussian random matrix
func NewGaussianRandomMatrix(cols, rows int, seed int64) GaussianRandomMatrix {
	return GaussianRandomMatrix{
		Cols: cols,
		Rows: rows,
		Seed: seed,
	}
}

// Sample generates a matrix from a gaussian distribution
func (g GaussianRandomMatrix) Sample() Matrix {
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
	A      GaussianRandomMatrix
	B      GaussianRandomMatrix
	Order  GaussianRandomMatrix
	Symbol GaussianRandomMatrix
	S      int
	Ranks  []float64
}

// Search searches for a symbol
func Search(s int, seed int64) []Sample {
	length := len(Puzzles[s].Q()) + 2
	rng := rand.New(rand.NewSource(seed))
	projections := make([]GaussianRandomMatrix, Scale)
	for i := range projections {
		seed := rng.Int63()
		if seed == 0 {
			seed = 1
		}
		projections[i] = NewGaussianRandomMatrix(Input, Input, seed)
	}
	index := 0
	samples := make([]Sample, Samples)
	for i := 0; i < Scale; i++ {
		for j := i + 1; j < Scale; j++ {
			seed := rng.Int63()
			if seed == 0 {
				seed = 1
			}
			order := NewGaussianRandomMatrix(Size, length, seed)
			seed = rng.Int63()
			if seed == 0 {
				seed = 1
			}
			symbol := NewGaussianRandomMatrix(Size, Symbols, seed)
			samples[index].A = projections[i]
			samples[index].B = projections[j]
			samples[index].Order = order
			samples[index].Symbol = symbol
			samples[index].S = index % SymbolsCount
			index++
		}
	}
	done := make(chan bool, 8)
	process := func(sample *Sample) {
		phi := NewZeroMatrix(Input, length)
		/*order := make([]float32, Size)
		for i := range order {
			order[i] = float32(rng.NormFloat64())
		}
		for i := 0; i < opt.Opt.Rows; i++ {
			copy(phi.Data[i*Input+Size:i*Input+Size+Size], order)
			for j, value := range order {
				order[j] = (value + float32(rng.NormFloat64())) / 2
			}
		}*/
		order := sample.Order.Sample()
		a, b := 0, 1
		jj := phi.Rows - 1
		for j := 0; j < jj; j++ {
			x, y := (j+a)%phi.Rows, (j+b)%phi.Rows
			copy(phi.Data[j*Input+Size:j*Input+Size+Size],
				order.Data[x*Size:(x+1)*Size])
			copy(phi.Data[j*Input+Size+Size:j*Input+Size+2*Size],
				order.Data[(y)*Size:(y+1)*Size])
			a, b = b, a
		}
		if x := jj + a; x < phi.Rows {
			copy(phi.Data[jj*Input+Size:jj*Input+Size+Size],
				order.Data[x*Size:(x+1)*Size])
		}
		if y := jj + b; y < phi.Rows {
			copy(phi.Data[jj*Input+Size+Size:jj*Input+Size+2*Size],
				order.Data[(y)*Size:(y+1)*Size])
		}
		syms := sample.Symbol.Sample()
		index := 0
		input := Puzzles[s].Q()
		for i := 0; i < len(input); i++ {
			symbol := syms.Data[Size*input[i] : Size*(input[i]+1)]
			copy(phi.Data[index:index+Input], symbol)
			index += Input
		}
		params := phi.Data[Input*(length-2) : Input*length-1]
		symbol := syms.Data[Size*To[rune(sample.S)] : Size*(To[rune(sample.S)]+1)]
		copy(params, symbol)
		{
			params := phi.Data[Input*(length-1) : Input*length]
			symbol := syms.Data[Size*To['$'] : Size*(To['$']+1)]
			copy(params, symbol)
		}
		/*for j := 0; j < phi.Rows; j++ {
			for i := 0; i < phi.Cols; i += 2 {
				phi.Data[j*phi.Cols+i] += complex(math.Sin(float64(j)/math.Pow(10000, 2*float64(i)/Size)), 0)
				phi.Data[j*phi.Cols+i+1] += complex(math.Cos(float64(j)/math.Pow(10000, 2*float64(i)/Size)), 0)
			}
		}*/
		/*factor := 1.0 / float64(Input)
		for i := range phi.Data {
			phi.Data[i] += float32(factor * sample.Rng.Float64())
		}*/
		query := sample.A.Sample()
		key := sample.B.Sample()
		q := query.MulT(phi)
		k := key.MulT(phi)
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

// Model
func Model(full bool, s int, seed int64) int {
	input := Puzzles[s].Q()[1:]
	rng := rand.New(rand.NewSource(seed))
	fmt.Println(string(Puzzles[s]))
	var y [SymbolsCount]*mat.Dense
	var avg [SymbolsCount][]float64
	var c [SymbolsCount]float64
	for i := 0; i < 8; i++ {
		seed := rng.Int63()
		if seed == 0 {
			seed = 1
		}
		samples := Search(s, seed)
		input := make([][]float64, SymbolsCount)
		var counts [SymbolsCount]int
		for sample := range samples {
			h := samples[sample].S
			counts[h]++
		}
		for j := range input {
			input[j] = make([]float64, 0, len(samples[0].Ranks)*counts[j])
		}
		for sample := range samples {
			ranks := samples[sample].Ranks
			h := samples[sample].S
			if avg[h] == nil {
				avg[h] = make([]float64, len(ranks))
			}
			c[h]++
			input[h] = append(input[h], ranks[1:len(ranks)-1]...)
			for j, rank := range ranks {
				avg[h][j] += rank
			}
		}
		for j := range counts {
			x := mat.NewDense(counts[j], len(samples[0].Ranks[1:len(samples[0].Ranks)-1]), input[j])
			dst := mat.SymDense{}
			stat.CovarianceMatrix(&dst, x, nil)
			if y[j] == nil {
				rr, cc := dst.Dims()
				y[j] = mat.NewDense(rr, cc, make([]float64, rr*cc))
			}
			y[j].Add(y[j], &dst)
		}
	}
	for h := range avg {
		for j := range avg[h] {
			avg[h][j] /= c[h]
		}
	}
	/*max, symbol := 0.0, 0
	sum := [SymbolsCount]float64{}
	for h := range y {
		fa := mat.Formatted(y[h], mat.Squeeze())
		fmt.Println(fa)
		rr, cc := y[h].Dims()
		ranks := make([]float64, rr)
		for i := 0; i < 32; i++ {
			graph := pagerank.NewGraph()
			for i := 0; i < rr; i++ {
				for j := 0; j < cc; j++ {
					d := (y[h].At(i, j)*rng.NormFloat64() + avg[h][j])
					if d > 0 {
						graph.Link(uint32(i), uint32(j), d)
					} else {
						graph.Link(uint32(j), uint32(i), -d)
					}
				}
			}
			graph.Rank(1, 1e-6, func(node uint32, rank float64) {
				ranks[node] += float64(rank)
			})
		}
		fmt.Println("ranks", ranks)
		results := make([]float64, 4)
		counts := make([]float64, 4)
		for i, v := range input {
			results[v] += ranks[i]
			counts[v]++
		}
		for i := range results {
			results[i] /= counts[i]
		}
		fmt.Println("results", results)
		for i, v := range results {
			sum[i] += v
			if v > max {
				max, symbol = v, i
			}
		}
	}
	fmt.Println(sum)
	{
		max, symbol := 0.0, 0
		for i, v := range sum {
			if v > max {
				max, symbol = v, i
			}
		}
		fmt.Println(max, symbol+1)
	}
	fmt.Println(avg)
	fmt.Println(max, symbol+1)*/

	var account [SymbolsCount][4]float64
	for h := 0; h < 4; h++ {
		for i, v := range input {
			account[h][v] += y[h].At(i, i) /// avg[h][i]
		}
		//account[h][h] += y[h].At(len(input), len(input)) / avg[h][h]
	}
	for i := range account {
		fmt.Println(account[i])
	}
	result := 0
	{
		min, symbol := math.MaxFloat64, 0
		for i := 0; i < SymbolsCount; i++ {
			v := account[i][i]
			if v < min {
				min, symbol = v, i
			}
		}
		fmt.Println(min, symbol+1)
		result = symbol
	}
	return result
}

func main() {
	seed := int64(2)
	histogram := [6][4]int{}
	for e := 0; e < 32; e++ {
		correct := 0
		for i := range Puzzles {
			result := Model(false, i, seed)
			histogram[i][result]++
			if result == Puzzles[i].A() {
				correct++
			}
		}
		fmt.Println("correct", correct)
		seed++
	}
	for i := range histogram {
		fmt.Println(histogram[i])
	}
}
