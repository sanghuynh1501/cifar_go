package main

import (
	"flag"
	"fmt"
	"image/jpeg"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"

	_ "net/http/pprof"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	// "cifar_cnn/cifar"
)

var (
	epochs     = flag.Int("epochs", 10, "Number of epochs to train for")
	dataset    = flag.String("dataset", "train", "Which dataset to train on? Valid options are \"train\" or \"test\"")
	dtype      = flag.String("dtype", "float64", "Which dtype to use")
	batchsize  = flag.Int("batchsize", 100, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const loc = "./cifar-10/"

var dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

type weights struct {
	weight []float64
	bias   []float64
}

type convnet struct {
	g                                              *gorgonia.ExprGraph
	w0, b0, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 *gorgonia.Node // weights. the number at the back indicates which layer it's used for
	d0, d1, d2, d3, d4                             float64        // dropout probabilities

	out     *gorgonia.Node
	predVal gorgonia.Value
}

func array_inintial(shape tensor.ConsOpt, input []float64) gorgonia.NodeConsOpt {
	image := tensor.New(shape, tensor.WithBacking(input))
	return gorgonia.WithValue(image)
}

func newConvNet(g *gorgonia.ExprGraph, w0_object weights, w1_object weights, w2_object weights, w3_object weights, w4_object weights, w5_object weights) *convnet {
	w0 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(32, 3, 3, 3), gorgonia.WithName("w0"), array_inintial(tensor.WithShape(32, 3, 3, 3), w0_object.weight))
	b0 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(1, 32, 1, 1), gorgonia.WithName("b0"), array_inintial(tensor.WithShape(1, 32, 1, 1), w0_object.bias))

	w1 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(32, 32, 3, 3), gorgonia.WithName("w1"), array_inintial(tensor.WithShape(32, 32, 3, 3), w1_object.weight))
	b1 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(1, 32, 1, 1), gorgonia.WithName("b1"), array_inintial(tensor.WithShape(1, 32, 1, 1), w1_object.bias))

	w2 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(64, 32, 3, 3), gorgonia.WithName("w2"), array_inintial(tensor.WithShape(64, 32, 3, 3), w2_object.weight))
	b2 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(1, 64, 1, 1), gorgonia.WithName("b2"), array_inintial(tensor.WithShape(1, 64, 1, 1), w2_object.bias))

	w3 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(64, 64, 3, 3), gorgonia.WithName("w3"), array_inintial(tensor.WithShape(64, 64, 3, 3), w3_object.weight))
	b3 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(1, 64, 1, 1), gorgonia.WithName("b3"), array_inintial(tensor.WithShape(1, 64, 1, 1), w3_object.bias))

	w4 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(1600, 512), gorgonia.WithName("w4"), array_inintial(tensor.WithShape(1600, 512), w4_object.weight))
	b4 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(1, 512), gorgonia.WithName("b4"), array_inintial(tensor.WithShape(1, 512), w4_object.bias))

	w5 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(512, 10), gorgonia.WithName("w5"), array_inintial(tensor.WithShape(512, 10), w5_object.weight))
	b5 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(1, 10), gorgonia.WithName("b5"), array_inintial(tensor.WithShape(1, 10), w5_object.bias))

	return &convnet{
		g:  g,
		w0: w0,
		b0: b0,
		w1: w1,
		b1: b1,
		w2: w2,
		b2: b2,
		w3: w3,
		b3: b3,
		w4: w4,
		b4: b4,
		w5: w5,
		b5: b5,

		d0: 0.25,
		d1: 0.25,
		d2: 0.25,
		d3: 0.25,
		d4: 0.5,
	}
}

func (m *convnet) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.b0, m.w1, m.b1, m.w2, m.b2, m.w3, m.b3, m.w4, m.b4, m.w5, m.b5}
}

// This function is particularly verbose for educational reasons. In reality, you'd wrap up the layers within a layer struct type and perform per-layer activations
func (m *convnet) fwd(x *gorgonia.Node) (err error) {
	var c0, c1, c2, c3, r3, r4 *gorgonia.Node
	var b0, b1, b2, b3, b4, b5 *gorgonia.Node
	var a0, a1, a2, a3, a4 *gorgonia.Node
	var p1, p3 *gorgonia.Node
	var l1, l3, l4 *gorgonia.Node

	// LAYER 0
	// here we convolve with stride = (1, 1) and padding = (0, 0),
	// which is your bog standard convolution for convnet
	if c0, err = gorgonia.Conv2d(x, m.w0, tensor.Shape{3, 3}, []int{0, 0}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 0 Convolution failed")
	}
	if b0, err = gorgonia.Add(c0, m.b0); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if a0, err = gorgonia.Rectify(b0); err != nil {
		return errors.Wrap(err, "Layer 0 activation failed")
	}
	log.Printf("layer 0 shape %v", a0.Shape())

	// LAYER 1
	// here we convolve with stride = (1, 1) and padding = (0, 0),
	// which is your bog standard convolution for convnet
	if c1, err = gorgonia.Conv2d(a0, m.w1, tensor.Shape{3, 3}, []int{0, 0}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 1 Convolution failed")
	}
	if b1, err = gorgonia.Add(c1, m.b1); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if a1, err = gorgonia.Rectify(b1); err != nil {
		return errors.Wrap(err, "Layer 1 activation failed")
	}
	if p1, err = gorgonia.MaxPool2D(a1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 1 Maxpooling failed")
	}
	if l1, err = gorgonia.Dropout(p1, m.d1); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout")
	}
	log.Printf("layer 1 shape %v", p1.Shape())

	// LAYER 2
	// here we convolve with stride = (1, 1) and padding = (0, 0),
	// which is your bog standard convolution for convnet
	if c2, err = gorgonia.Conv2d(l1, m.w2, tensor.Shape{3, 3}, []int{0, 0}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 2 Convolution failed")
	}
	if b2, err = gorgonia.Add(c2, m.b2); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if a2, err = gorgonia.Rectify(b2); err != nil {
		return errors.Wrap(err, "Layer 2 activation failed")
	}
	log.Printf("layer 2 shape %v", a2.Shape())

	// LAYER 3
	// here we convolve with stride = (1, 1) and padding = (0, 0),
	// which is your bog standard convolution for convnet
	if c3, err = gorgonia.Conv2d(a2, m.w3, tensor.Shape{3, 3}, []int{0, 0}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 3 Convolution failed")
	}
	if b3, err = gorgonia.Add(c3, m.b3); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if a3, err = gorgonia.Rectify(b3); err != nil {
		return errors.Wrap(err, "Layer 3 activation failed")
	}
	if p3, err = gorgonia.MaxPool2D(a3, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 3 Maxpooling failed")
	}
	if l3, err = gorgonia.Dropout(p3, m.d3); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout")
	}
	log.Printf("layer 3 shape %v", l3.Shape())

	// Flattern
	b, c, h, w := l3.Shape()[0], l3.Shape()[1], l3.Shape()[2], l3.Shape()[3]
	if r3, err = gorgonia.Reshape(l3, tensor.Shape{b, c * h * w}); err != nil {
		return errors.Wrap(err, "Unable to reshape layer 3")
	}
	log.Printf("flattern shape %v", r3.Shape())

	ioutil.WriteFile("tmp.dot", []byte(m.g.ToDot()), 0644)

	// Layer 4
	if r4, err = gorgonia.Mul(r3, m.w4); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if b4, err = gorgonia.Add(r4, m.b4); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if a4, err = gorgonia.Rectify(b4); err != nil {
		return errors.Wrapf(err, "Unable to activate fc")
	}
	if l4, err = gorgonia.Dropout(a4, m.d4); err != nil {
		return errors.Wrapf(err, "Unable to apply a dropout on layer 3")
	}
	log.Printf("layer 4 shape %v", l4.Shape())

	// Layer 5
	var out *gorgonia.Node
	if out, err = gorgonia.Mul(l4, m.w5); err != nil {
		return errors.Wrapf(err, "Unable to multiply l3 and w4")
	}
	if b5, err = gorgonia.Add(out, m.b5); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	m.out, err = gorgonia.SoftMax(b5)
	gorgonia.Read(m.out, &m.predVal)
	log.Printf("layer 5 shape %v", out.Shape())

	return
}

// Get the bi-dimensional pixel array
func getPixels(file io.Reader) ([][]Pixel, error) {
	img, err := jpeg.Decode(file)

	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	var pixels [][]Pixel
	for y := 0; y < height; y++ {
		var row []Pixel
		for x := 0; x < width; x++ {
			row = append(row, rgbaToPixel(img.At(x, y).RGBA()))
		}
		pixels = append(pixels, row)
	}

	return pixels, nil
}

// img.At(x, y).RGBA() returns four uint32 values; we want a Pixel
func rgbaToPixel(r uint32, g uint32, b uint32, a uint32) Pixel {
	return Pixel{int(r / 257), int(g / 257), int(b / 257), int(a / 257)}
}

// Pixel struct example
type Pixel struct {
	R int
	G int
	B int
	A int
}

func main() {
	flag.Parse()
	parseDtype()
	rand.Seed(1337)

	g := gorgonia.NewGraph()
	xT := tensor.New(tensor.WithShape(10, 32, 30, 30), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 288000)))
	yT := tensor.New(tensor.WithShape(1, 32, 1, 1), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 32)))

	x := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(10, 32, 30, 30), gorgonia.WithValue(xT), gorgonia.WithName("x"))
	y := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(1, 32, 1, 1), gorgonia.WithValue(yT), gorgonia.WithName("y"))
	a, b, _ := gorgonia.Broadcast(x, y, gorgonia.NewBroadcastPattern(nil, []byte{9}))
	log.Println(a.Shape())
	log.Println(b.Shape())
	// z, _ := gorgonia.Add(a, b)
	// log.Println(z)

	// // intercept Ctrl+C
	// sigChan := make(chan os.Signal, 1)
	// signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	// doneChan := make(chan bool, 1)

	// var inputs, targets tensor.Tensor
	// var err error

	// // go func() {
	// // 	log.Println(http.ListenAndServe("localhost:6060", nil))
	// // }()

	// trainOn := *dataset
	// if inputs, targets, err = cifar.Load(trainOn, loc); err != nil {
	// 	log.Fatal(err)
	// }

	// numExamples := inputs.Shape()[0]
	// bs := *batchsize

	// if err := inputs.Reshape(numExamples, 3, 32, 32); err != nil {
	// 	log.Fatal(err)
	// }
	// g := gorgonia.NewGraph()
	// x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(bs, 3, 32, 32), gorgonia.WithName("x"))
	// y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 10), gorgonia.WithName("y"))
	// w0, b0, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 := weight.Load()

	// w0_object := weights{w0, b0}
	// w1_object := weights{w1, b1}
	// w2_object := weights{w2, b2}
	// w3_object := weights{w3, b3}
	// w4_object := weights{w4, b4}
	// w5_object := weights{w5, b5}

	// m := newConvNet(g, w0_object, w1_object, w2_object, w3_object, w4_object, w5_object)
	// if err = m.fwd(x); err != nil {
	// 	log.Fatalf("%+v", err)
	// }
	// if err = m.fwd(x); err != nil {
	// 	log.Fatalf("%+v", err)
	// }
	// losses := gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Log(m.out)), y))
	// cost := gorgonia.Must(gorgonia.Sum(losses))
	// cost = gorgonia.Must(gorgonia.Neg(cost))

	// // we wanna track costs
	// var costVal gorgonia.Value
	// gorgonia.Read(cost, &costVal)

	// if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
	// 	log.Fatal(err)
	// }

	// // debug
	// ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)

	// prog, locMap, _ := gorgonia.Compile(g)
	// log.Printf("%v", prog)

	// vm := gorgonia.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(m.learnables()...))
	// defer vm.Close()

	// var profiling bool
	// if *cpuprofile != "" {
	// 	f, err := os.Create(*cpuprofile)
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}
	// 	profiling = true
	// 	pprof.StartCPUProfile(f)
	// 	defer pprof.StopCPUProfile()
	// }
	// go cleanup(sigChan, doneChan, profiling)

	// batches := numExamples / bs
	// log.Printf("Batches %d", batches)
	// bar := pb.New(batches)
	// bar.SetRefreshRate(time.Second)
	// bar.SetMaxWidth(80)

	// // import test data and run more loops
	// if inputs, targets, err = cifar.Load("test", loc); err != nil {
	// 	log.Fatal(err)
	// }

	// batches = inputs.Shape()[0] / bs
	// bar = pb.New(batches)
	// bar.SetRefreshRate(time.Second)
	// bar.SetMaxWidth(80)

	// var testActual, testPred []int

	// total := 0.0
	// total_true := 0.0

	// for i := 0; i < 1; i++ {
	// 	bar.Prefix(fmt.Sprintf("Epoch Test"))
	// 	bar.Set(0)
	// 	bar.Start()
	// 	for b := 0; b < batches; b++ {
	// 		start := b * bs
	// 		end := start + bs
	// 		if start >= numExamples {
	// 			break
	// 		}
	// 		if end > numExamples {
	// 			end = numExamples
	// 		}

	// 		var xVal, yVal tensor.Tensor
	// 		if xVal, err = inputs.Slice(sli{start, end}); err != nil {
	// 			log.Fatal("Unable to slice x")
	// 		}

	// 		if yVal, err = targets.Slice(sli{start, end}); err != nil {
	// 			log.Fatal("Unable to slice y")
	// 		}
	// 		if err = xVal.(*tensor.Dense).Reshape(bs, 3, 32, 32); err != nil {
	// 			log.Fatalf("Unable to reshape %v", err)
	// 		}

	// 		gorgonia.Let(x, xVal)
	// 		gorgonia.Let(y, yVal)
	// 		if err = vm.RunAll(); err != nil {
	// 			log.Fatalf("Failed at epoch test: %v", err)
	// 		}

	// 		arrayOutput := m.predVal.Data().([]float64)
	// 		yOutput := tensor.New(tensor.WithShape(bs, 10), tensor.WithBacking(arrayOutput))

	// 		for j := 0; j < yVal.Shape()[0]; j++ {

	// 			// get label
	// 			yRowT, _ := yVal.Slice(sli{j, j + 1})
	// 			yRow := yRowT.Data().([]float64)
	// 			var rowLabel int
	// 			var yRowHigh float64

	// 			for k := 0; k < 10; k++ {
	// 				if k == 0 {
	// 					rowLabel = 0
	// 					yRowHigh = yRow[k]
	// 				} else if yRow[k] > yRowHigh {
	// 					rowLabel = k
	// 					yRowHigh = yRow[k]
	// 				}
	// 			}

	// 			// get prediction
	// 			predRowT, _ := yOutput.Slice(sli{j, j + 1})
	// 			predRow := predRowT.Data().([]float64)
	// 			var rowGuess int
	// 			var predRowHigh float64

	// 			// guess result
	// 			for k := 0; k < 10; k++ {
	// 				if k == 0 {
	// 					rowGuess = 0
	// 					predRowHigh = predRow[k]
	// 				} else if predRow[k] > predRowHigh {
	// 					rowGuess = k
	// 					predRowHigh = predRow[k]
	// 				}
	// 			}

	// 			if rowGuess == rowLabel {
	// 				total_true += 1
	// 			}
	// 			total += 1

	// 			testActual = append(testActual, rowLabel)
	// 			testPred = append(testPred, rowGuess)
	// 		}

	// 		vm.Reset()
	// 		bar.Increment()
	// 	}
	// 	log.Printf("Epoch Test | cost %v", costVal)
	// 	log.Printf("Epoch Test | accurancy %v", (total_true/total)*100)

	// 	printIntSlice("testActual.txt", testActual)
	// 	printIntSlice("testPred.txt", testPred)
	// }
}

func cleanup(sigChan chan os.Signal, doneChan chan bool, profiling bool) {
	select {
	case <-sigChan:
		log.Println("EMERGENCY EXIT!")
		if profiling {
			log.Println("Stop profiling")
			pprof.StopCPUProfile()
		}
		os.Exit(1)

	case <-doneChan:
		return
	}
}

func handlePprof(sigChan chan os.Signal, doneChan chan bool) {
	var profiling bool
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		profiling = true
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	go cleanup(sigChan, doneChan, profiling)
}

func printIntSlice(filePath string, values []int) error {
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, value := range values {
		fmt.Fprintln(f, value)
	}
	return nil
}
