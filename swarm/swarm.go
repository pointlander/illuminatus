// Copyright 2024 The Illuminatus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package swarm

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"runtime"
	"sort"

	"github.com/alixaxel/pagerank"
)

const (
	// Size is the link size
	Size = 16
	// Input is the network input size
	Input = Size + 2*Size
	// Scale is the scale of the search
	Scale = 33 //48 96
	// Samples is the number of samplee
	Samples = Scale * (Scale - 1) / 2
	// TapeSize is the size of the tape
	TapeSize = 8
	// TapeMask is the mask for the MSB of the tape
	TapeMask = 1 << (TapeSize - 1)
)

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
		sample.Data = append(sample.Data, complex(a, 0))
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
func PageRank(x, y Matrix) []float64 {
	graph := pagerank.NewGraph()
	for i := 0; i < y.Rows; i++ {
		yy := y.Data[i*y.Cols : (i+1)*y.Cols]
		aa := complex(0.0, 0.0)
		for _, v := range yy {
			aa += v * v
		}
		aa = cmplx.Sqrt(aa)
		for j := 0; j < x.Rows; j++ {
			xx := x.Data[j*x.Cols : (j+1)*x.Cols]
			bb := complex(0.0, 0.0)
			for _, v := range xx {
				bb += v * v
			}
			bb = cmplx.Sqrt(bb)
			d := cmplx.Abs(Dot(yy, xx) / (aa * bb))
			graph.Link(uint32(i), uint32(j), d)
		}
	}
	ranks := make([]float64, y.Rows)
	graph.Rank(1.0, 1e-9, func(node uint32, rank float64) {
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

// Cell is a cell
type Cell struct {
	Head int
	Tape []byte
	Rank float64
}

// NewCell creates a new cell
func NewCell(rng *rand.Rand, size int) Cell {
	tape := make([]byte, size)
	for i := range tape {
		tape[i] = byte(rng.Intn(2))
	}
	head := rng.Intn(size)
	return Cell{
		Head: head,
		Tape: tape,
	}
}

// Bits returns the bits
func (c Cell) Bits() uint64 {
	bits := uint64(0)
	for _, bit := range c.Tape {
		bits <<= 1
		if bit == 1 {
			bits |= 1
		}
	}
	return bits
}

// Step steps the cell
func (c *Cell) Step(rng *rand.Rand) {
	state := Step(rng, c.Tape)
	current := c.Tape[c.Head]
	c.Tape[c.Head] = byte(state)
	if (current^byte(state))&1 == 0 {
		c.Head = (c.Head + TapeSize - 1) % TapeSize
	} else {
		c.Head = (c.Head + 1) % TapeSize
	}
}

func (c Cell) Copy() Cell {
	tape := make([]byte, len(c.Tape))
	copy(tape, c.Tape)
	return Cell{
		Head: c.Head,
		Tape: tape,
	}
}

// Swarm is swarm mode
func Swarm(target int) {
	rng := rand.New(rand.NewSource(33))
	cells := make([]Cell, 64)
	for i := range cells {
		cells[i] = NewCell(rng, TapeSize)
	}
	for {
		graph := pagerank.NewGraph()
		for i := 0; i < 256; i++ {
			a := make([]int, 8)
			for j := range a {
				a[j] = rng.Intn(len(cells))
			}
			aa := uint64(0)
			for _, v := range a {
				aa <<= 8
				aa |= cells[v].Bits()
			}
			if aa != 0 {
				aa = aa % uint64(target)
			}
			weight := 0.0
			if aa != 0 {
				weight = float64((uint64(target) % aa))
				if weight == 0 {
					fmt.Println("found", aa)
					return
				}
				weight = float64(uint64(target)) - weight
			}
			for _, v := range a {
				for _, vv := range a {
					graph.Link(uint32(v), uint32(vv), weight)
				}
			}
		}
		graph.Rank(1.0, 1e-9, func(node uint32, rank float64) {
			cells[node].Rank = rank
		})
		sort.Slice(cells, func(i, j int) bool {
			return cells[i].Rank > cells[j].Rank
		})
		min := uint64(math.MaxUint64)
		for i := 0; i < 256; i++ {
			a := make([]int, 8)
			for j := range a {
				sum, r := 0.0, rng.Float64()
				for k := range cells {
					sum += cells[k].Rank
					if sum > r {
						a[j] = k
						break
					}
				}
			}
			aa := uint64(0)
			for _, v := range a {
				aa <<= 8
				aa |= cells[v].Bits()
			}
			if aa != 0 {
				aa = aa % uint64(target)
			}
			if aa < min {
				min = aa
			}
			weight := 0.0
			if aa != 0 {
				weight = float64((uint64(target) % aa))
				if weight == 0 {
					fmt.Println("found", aa)
					return
				}
			}
		}
		fmt.Println(min)
		cells = cells[:32]
		for i := 0; i < 32; i++ {
			cp := cells[i].Copy()
			cp.Step(rng)
			cells = append(cells, cp)
		}
	}
}

// Step steps the turing machine
func Step(rng *rand.Rand, tape []byte) int {
	const Symbols = 2
	length := len(tape)
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
			index++
		}
	}

	done := make(chan bool, 8)
	process := func(sample *Sample) {
		var inputs [2]Matrix
		inputs[0] = NewZeroMatrix(Input, length)
		inputs[1] = NewZeroMatrix(Input, length)
		for i := range inputs {
			input := &inputs[i]
			order := sample.Order[i].Sample()
			a, b := 0, 1
			jj := input.Rows
			for j := 0; j < jj; j++ {
				x, y := (j+a)%input.Rows, (j+b)%input.Rows
				copy(input.Data[j*Input+Size:j*Input+Size+Size],
					order.Data[x*Size:(x+1)*Size])
				copy(input.Data[j*Input+Size+Size:j*Input+Size+2*Size],
					order.Data[(y)*Size:(y+1)*Size])
				a, b = b, a
			}
			syms := sample.Symbol[i].Sample()
			index := 0
			for i := 0; i < len(tape); i++ {
				symbol := syms.Data[Size*tape[i] : Size*(tape[i]+1)]
				copy(input.Data[index:index+Input], symbol)
				index += Input
			}
		}
		a := sample.A.Sample()
		b := sample.B.Sample()
		x := a.MulT(inputs[0])
		y := b.MulT(inputs[1])
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

	min, result := math.MaxFloat64, 0
	for symbol := 0; symbol < Symbols; symbol++ {
		indexes := make([]int, 0, 8)
		for key, value := range tape {
			if int(value) == symbol {
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
	return result
}
