// Copyright 2024 The Illuminatus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"

	"github.com/pointlander/illuminatus/matrix"
	"github.com/pointlander/illuminatus/swarm"
	"github.com/pointlander/illuminatus/turing"
	"github.com/pointlander/illuminatus/vector"

	"github.com/alixaxel/pagerank"
)

const (
	// Symbols
	Symbols = ('z' - 'a' + 1) + ('Z' - 'A' + 1) + 3
	// Size is the link size
	Size = 16
	// Input is the network input size
	Input = Symbols + 1 //Size + 2*Size
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// Scale is the scale of the search
	Scale = 33 //48 96
	// SymbolsCount is the number of unique symbols in a puzzle
	SymbolsCount = 4
	// Samples is the number of samplee
	Samples = Scale * Scale
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
	//"^a$ ^ab$ ^abc$ ^abcd$ ^abcda$ ^abcdab",
	"^abcdabcdabcdabcdabcda",
	"^abcdabcdabcdabcdabcdab",
	"^abcdabcdabcdabcdabcdabc",
	"^abcdabcdabcdabcdabcdabcd",
	"^abcddcbaabcddcbaabcddcbaabcd",
	"^aabbccddaabbccddaabbccd",
	"^aabbccddaabbccddaabbccdd",
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
func PageRank(x, y Matrix) []float64 {
	graph := pagerank.NewGraph()
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
			graph.Link(uint32(i), uint32(j), d)
		}
	}
	ranks := make([]float64, y.Rows)
	graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
		ranks[node] = rank
	})
	return ranks
}

// Sample is a sample
type Sample struct {
	A        RandomMatrix
	B        RandomMatrix
	Order    [2]RandomMatrix
	Symbol   [2]RandomMatrix
	Ranks    []float64
	Variance float64
}

// Search searches for a symbol
func (puzzle Puzzle) Search(seed int64, r1, r2 []Random) []Sample {
	length := len(puzzle.Q()) + 1
	rng := rand.New(rand.NewSource(seed))
	projections1 := make([]RandomMatrix, Scale)
	for i := range projections1 {
		seed := rng.Int63()
		if seed == 0 {
			seed = 1
		}
		projections1[i] = NewRandomMatrix(Input, Input, seed, r1...)
	}
	projections2 := make([]RandomMatrix, Scale)
	for i := range projections2 {
		seed := rng.Int63()
		if seed == 0 {
			seed = 1
		}
		projections2[i] = NewRandomMatrix(Input, Input, seed, r2...)
	}
	index := 0
	samples := make([]Sample, Samples)
	for i := 0; i < Scale; i++ {
		for j := 0; j < Scale; j++ {
			samples[index].A =
				projections1[i]
			samples[index].B = projections2[j]
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
			/*seed = rng.Int63()
			if seed == 0 {
				seed = 1
			}*/
			order = NewRandomMatrix(Size, length, 1)
			/*seed = rng.Int63()
			if seed == 0 {
				seed = 1
			}*/
			symbol = NewRandomMatrix(Size, Symbols, 2)
			samples[index].Order[1] = order
			samples[index].Symbol[1] = symbol
			index++
		}
	}

	done := make(chan bool, 8)
	process := func(sample *Sample) {
		var inputs [2]Matrix
		q := puzzle.Q()
		inputs[0] = NewZeroMatrix(Input, length)
		inputs[1] = NewZeroMatrix(Input, length)
		for i := range inputs {
			input := &inputs[i]
			/*order := sample.Order[i].Sample()
			a, b := 0, 1
			jj := input.Rows - 1
			for j := 0; j < jj; j++ {
				x, y := (j+a)%input.Rows, (j+b)%input.Rows
				copy(input.Data[j*Input+Size:j*Input+Size+Size],
					order.Data[x*Size:(x+1)*Size])
				copy(input.Data[j*Input+Size+Size:j*Input+Size+2*Size],
					order.Data[(y)*Size:(y+1)*Size])
				a, b = b, a
			}*/
			/*for j := jj; j < jj+3; j++ {
				x, y := (jj-1+b)%phi.Rows, (jj-1+a)%phi.Rows
				copy(phi.Data[j*Input+Size:j*Input+Size+Size],
					order.Data[x*Size:(x+1)*Size])
				copy(phi.Data[j*Input+Size+Size:j*Input+Size+2*Size],
					order.Data[(y)*Size:(y+1)*Size])
			}*/
			/*if x := jj + a; x < order.Rows {
				//jj += 3
				copy(input.Data[jj*Input+Size:jj*Input+Size+Size],
					order.Data[x*Size:(x+1)*Size])
			} else if y := jj + b; y < order.Rows {
				//jj += 3
				copy(input.Data[jj*Input+Size+Size:jj*Input+Size+2*Size],
					order.Data[(y)*Size:(y+1)*Size])
			} else {
				panic("shouldn't be here")
			}*/
			//syms := sample.Symbol[i].Sample()
			index := 0
			for i := 0; i < len(q); i++ {
				input.Data[index+q[i]] = 1
				index += Input
			}
			{
				input.Data[index+To['$']] = 1
			}
			for j := 0; j < input.Rows; j++ {
				for i := 0; i < input.Cols; i += 2 {
					input.Data[j*input.Cols+i] += float32(math.Sin(float64(j) / math.Pow(10000, 2*float64(i)/Size)))
					input.Data[j*input.Cols+i+1] += float32(math.Cos(float64(j) / math.Pow(10000, 2*float64(i)/Size)))
				}
			}
		}
		a := sample.A.Sample()
		b := sample.B.Sample()
		x := a.MulT(inputs[0]).TanH()
		y := b.MulT(inputs[1]).TanH()
		sample.Ranks = PageRank(x, y)
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
func (puzzle Puzzle) Illuminatus(seed int64) int {
	const (
		// Scale is the scale of the search
		MetaScale = 7
		// Samples is the number of samplee
		MetaSamples = MetaScale * (MetaScale - 1) / 2
		// Cutoff is the cutoff
		Cutoff = 13
	)
	rng := rand.New(rand.NewSource(seed))
	fmt.Println(string(puzzle))
	var r1, r2 []Random
	min, result := math.MaxFloat64, 0
	for e := 0; e < 16; e++ {
		seed = rng.Int63()
		if seed == 0 {
			seed = 1
		}
		samples := puzzle.Search(seed, r1, r2)
		input := puzzle.Q()
		/*projections := make([]RandomMatrix, MetaScale)
		for i := range projections {
			seed := rng.Int63()
			if seed == 0 {
				seed = 1
			}
			projections[i] = NewRandomMatrix(len(input)+1, len(input)+1, seed)
		}
		results := make([][]float64, 0, 8)
		for i := 0; i < MetaScale; i++ {
			for j := i + 1; j < MetaScale; j++ {
				ranks := NewMatrix(len(input)+1, len(samples))
				for sample := range samples {
					for _, rank := range samples[sample].Ranks {
						ranks.Data = append(ranks.Data, complex(rank, 0))
					}
				}
				a := projections[i].Sample()
				b := projections[j].Sample()
				x := a.MulT(ranks)
				y := b.MulT(ranks)
				result := PageRank(x, y)
				results = append(results, result)
			}
		}
		averages := make([]float64, Samples)
		for _, result := range results {
			for i := range result {
				averages[i] += result[i]
			}
		}
		for i := range averages {
			averages[i] /= float64(len(results))
		}
		variances := make([]float64, Samples)
		for _, result := range results {
			for i := range result {
				diff := averages[i] - result[i]
				variances[i] += diff * diff
			}
		}
		for i := range variances {
			samples[i].Variance = variances[i]
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Variance < samples[j].Variance
		})*/

		/*aa := [4][]float64{}
		sums, count := make([]float64, len(input)), 0.0
		sum := 0.0
		for sample := range samples {
			ranks := samples[sample].Ranks
			entropy := 0.0
			for _, value := range ranks {
				if value == 0 {
					continue
				}
				entropy += value * math.Log2(value)
			}
			entropy = -entropy
			sum += entropy
			ranks = ranks[:len(input)]
			for key, value := range ranks {
				sums[key] += value
				if k := input[key]; k == 0 || k == 1 || k == 2 || k == 3 {
					aa[k] = append(aa[k], value)
				}
			}
			count++
		}
		for sample := range samples {
			ranks := samples[sample].Ranks
			entropy := 0.0
			for _, value := range ranks {
				if value == 0 {
					continue
				}
				entropy += value * math.Log2(value)
			}
			entropy = -entropy
			fmt.Println(entropy / sum)
		}
		for i := range sums {
			sums[i] /= count
		}
		type Variance struct {
			Symbol   int
			Variance float64
		}
		variances := make([]Variance, len(input))
		for i := range variances {
			variances[i].Symbol = input[i]
		}
		for sample := range samples {
			ranks := samples[sample].Ranks[:len(input)]
			for key, value := range ranks {
				diff := sums[key] - value
				variances[key].Variance += diff * diff
			}
		}
		sort.Slice(variances, func(i, j int) bool {
			return variances[i].Variance < variances[j].Variance
		})
		for _, variance := range variances {
			fmt.Println(variance.Symbol, variance.Variance)
		}

		for a, aa := range aa {
			sort.Slice(aa, func(i, j int) bool {
				return aa[i] < aa[j]
			})
			sum := 0.0
			for _, value := range aa {
				sum += value
			}
			sum /= float64(len(aa))
			v := 0.0
			for _, value := range aa {
				diff := value - sum
				v += diff * diff
			}
			v /= float64(len(aa))
			max, index := 0.0, 0
			maxA, maxB := 0.0, 0.0
			for i := 1; i < len(aa)-1; i++ {
				sumA, sumB := 0.0, 0.0
				countA, countB := 0.0, 0.0
				for _, value := range aa[:i] {
					sumA += value
					countA++
				}
				for _, value := range aa[i:] {
					sumB += value
					countB++
				}
				sumA /= countA
				sumB /= countB
				varA, varB := 0.0, 0.0
				for _, value := range aa[:i] {
					diff := value - sumA
					varA += diff * diff
				}
				for _, value := range aa[i:] {
					diff := value - sumB
					varB += diff * diff
				}
				varA /= countA
				varB /= countB
				vv := v - (varA + varB)
				if vv > max {
					max, index = vv, i
					maxA, maxB = varA, varB
				}
			}
			fmt.Println(a, max, maxA, maxB, index, len(aa), float64(index)/float64(len(aa)), aa[0], aa[index], aa[len(aa)-1])
		}*/

		for symbol := 0; symbol < SymbolsCount; symbol++ {
			indexes := make([]int, 0, 8)
			for key, value := range input {
				if value == symbol {
					indexes = append(indexes, key)
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

		{
			sum, count := make([]float64, len(samples[0].Ranks)), 0.0
			for sample := range samples {
				ranks := samples[sample].Ranks
				for i, r := range ranks {
					sum[i] += r
					count++
				}
			}
			average := make([]float64, len(sum))
			for i, r := range sum {
				average[i] = r / count
			}
			for sample := range samples {
				variance := 0.0
				ranks := samples[sample].Ranks
				for i, r := range ranks {
					diff := average[i] - r
					variance += diff * diff
				}
				samples[sample].Variance = variance
			}
			sort.Slice(samples, func(i, j int) bool {
				return samples[i].Variance < samples[j].Variance
			})
			fmt.Println(samples[0].Variance)
			d := NewRandomMatrix(Input, Input, 1)
			for i := range d.Rand {
				d.Rand[i].Stddev = 0
			}
			for sample := range samples[:Cutoff] {
				a := samples[sample].A.Sample()
				for i, v := range a.Data {
					d.Rand[i].Mean += v
				}
			}
			for i := range d.Rand {
				d.Rand[i].Mean /= Cutoff
			}
			for sample := range samples[:Cutoff] {
				a := samples[sample].A.Sample()
				for i, v := range a.Data {
					diff := d.Rand[i].Mean - v
					d.Rand[i].Stddev += diff * diff
				}
			}
			for i := range d.Rand {
				d.Rand[i].Stddev /= Cutoff
				d.Rand[i].Stddev = float32(math.Sqrt(float64(d.Rand[i].Stddev)))
			}
			r1 = d.Rand

			d = NewRandomMatrix(Input, Input, 1)
			for i := range d.Rand {
				d.Rand[i].Stddev = 0
			}
			for sample := range samples[:Cutoff] {
				b := samples[sample].B.Sample()
				for i, v := range b.Data {
					d.Rand[i].Mean += v
				}
			}
			for i := range d.Rand {
				d.Rand[i].Mean /= Cutoff
			}
			for sample := range samples[:Cutoff] {
				b := samples[sample].B.Sample()
				for i, v := range b.Data {
					diff := d.Rand[i].Mean - v
					d.Rand[i].Stddev += diff * diff
				}
			}
			for i := range d.Rand {
				d.Rand[i].Stddev /= Cutoff
				d.Rand[i].Stddev = float32(math.Sqrt(float64(d.Rand[i].Stddev)))
			}
			r2 = d.Rand
		}
	}
	fmt.Println(result)

	return result
}

var (
	// FlagTuring in turing mode
	FlagTuring = flag.Bool("turing", false, "turing mode")
	// FlagTarget is the factoring target
	FlagTarget = flag.Int("t", 49, "factoring target")
	// FlagSwarm is the swarm mode
	FlagSwarm = flag.Bool("swarm", false, "swarm mode")
	// FlagMatrix is the matrix mode
	FlagMatrix = flag.Bool("matrix", false, "matrix mode")
)

func main() {
	flag.Parse()

	if *FlagTuring {
		turing.Turing(*FlagTarget)
		return
	} else if *FlagSwarm {
		swarm.Swarm(*FlagTarget)
		return
	} else if *FlagMatrix {
		matrix.Matr1x()
		return
	}

	seed := int64(2)
	histogram := [7][4]int{}
	for e := 0; e < 32; e++ {
		correct := 0
		for i := range Puzzles {
			result := Puzzles[i].Illuminatus(seed)
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
