package main

import (
	"cifar_go/jsbiding"
	"cifar_go/weight"
	"flag"
	"log"
	"math/rand"
	"strconv"
	"syscall/js"
	"time"

	_ "net/http/pprof"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	dtype     = flag.String("dtype", "float64", "Which dtype to use")
	batchsize = flag.Int("batchsize", 100, "Batch size")
)

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

type convnet struct {
	g       *gorgonia.ExprGraph
	weights []*gorgonia.Node // weights. the number at the back indicates which layer it's used for
	out     *gorgonia.Node
	predVal gorgonia.Value
}

func array_inintial(shape tensor.ConsOpt, input []float64) gorgonia.NodeConsOpt {
	image := tensor.New(shape, tensor.WithBacking(input))
	return gorgonia.WithValue(image)
}

func newConvNet(g *gorgonia.ExprGraph, bs int, config []int, weights_array []weight.Weight) *convnet {
	var weights []*gorgonia.Node

	in_channels := 3
	index := 0
	for i, x := range config {
		if x != -1 {
			w := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(x, in_channels, 3, 3), gorgonia.WithName("w"+strconv.FormatInt(int64(i), 10)), array_inintial(tensor.WithShape(x, in_channels, 3, 3), weights_array[index].Value))
			weights = append(weights, w)
			in_channels = x
			index += 1
		}
	}

	linear := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(512*2*2, 512), gorgonia.WithName("w"+string(len(weights))), array_inintial(tensor.WithShape(512*2*2, 512), weights_array[index].Value))
	weights = append(weights, linear)
	index += 1

	classification := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(512, 10), gorgonia.WithName("w"+string(len(weights))), array_inintial(tensor.WithShape(512, 10), weights_array[index].Value))
	weights = append(weights, classification)

	return &convnet{
		g:       g,
		weights: weights,
	}
}

// This function is particularly verbose for educational reasons. In reality, you'd wrap up the layers within a layer struct type and perform per-layer activations
func (m *convnet) fwd(x *gorgonia.Node, config []int) (err error) {
	var out *gorgonia.Node

	index := 0
	for _, layer := range config {
		if layer != -1 {
			if x, err = gorgonia.Conv2d(x, m.weights[index], tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
				return errors.Wrap(err, "Layer 0 Convolution failed")
			}
			// if x, _, _, _, err = gorgonia.BatchNorm(x, nil, nil, 0.1, 0.0001); err != nil {
			// 	return errors.Wrap(err, "Layer 0 Convolution failed")
			// }
			if x, err = gorgonia.Rectify(x); err != nil {
				return errors.Wrap(err, "Layer 0 activation failed")
			}
			index += 1
		} else {
			if x, err = gorgonia.MaxPool2D(x, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
				return errors.Wrap(err, "Layer 1 Maxpooling failed")
			}
		}
	}
	b, c, h, w := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]

	if out, err = gorgonia.Reshape(x, tensor.Shape{b, c * h * w}); err != nil {
		return errors.Wrap(err, "Unable to reshape layer 3")
	}

	if out, err = gorgonia.Mul(out, m.weights[len(m.weights)-2]); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}

	if out, err = gorgonia.Mul(out, m.weights[len(m.weights)-1]); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}

	m.out = out
	gorgonia.Read(m.out, &m.predVal)

	return
}

func main() {
	start := time.Now()
	flag.Parse()
	parseDtype()
	rand.Seed(1337)
	g := gorgonia.NewGraph()
	labels := []string{"airplane", "automobile", "bird", "cat",
		"deer", "dog", "frog", "horse", "ship", "truck"}

	bs := 2
	var err error
	weights_array := weight.Load()
	image1, image2 := jsbiding.GetImage()
	xValue1 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(1, 3*32*32), gorgonia.WithName("xValue1"), array_inintial(tensor.WithShape(1, 3*32*32), image1))
	xValue2 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(1, 3*32*32), gorgonia.WithName("xValue2"), array_inintial(tensor.WithShape(1, 3*32*32), image2))
	xValue1, err = gorgonia.Reshape(xValue1, tensor.Shape{1, 3, 32, 32})
	xValue2, err = gorgonia.Reshape(xValue2, tensor.Shape{1, 3, 32, 32})
	xValue, err := gorgonia.Concat(0, xValue1, xValue2)

	VGG19 := []int{64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512}
	m := newConvNet(g, bs, VGG19, weights_array)

	if err = m.fwd(xValue, VGG19); err != nil {
		log.Fatalf("%+v", err)
	}
	mv := gorgonia.NewTapeMachine(g)
	defer mv.Close()
	if err = mv.RunAll(); err != nil {
		log.Fatal(err)
	}
	predict_array := m.out.Value().Data().([]float64)[:10]
	max := predict_array[0]
	label := 0
	for i := 0; i < 10; i++ {
		if predict_array[i] > max {
			max = predict_array[i]
			label = i
		}
	}
	log.Println("label ", labels[label])
	doc := js.Global().Get("document")
	label_dom := doc.Call("getElementById", "predict_label")
	label_dom.Set("innerHTML", labels[label])
	elapsed := time.Since(start)
	log.Printf("Time took %s", elapsed)
}
