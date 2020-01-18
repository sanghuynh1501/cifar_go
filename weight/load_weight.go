package weight

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"

	"github.com/utahta/go-openuri"
)

type Layer struct {
	Shape []int32
	Value []float64
}

type Weight struct {
	Weight Layer
	Bias   Layer
}

type Weights struct {
	Layer_0 Weight
	Layer_1 Weight
	Layer_2 Weight
	Layer_3 Weight
	Layer_4 Weight
	Layer_5 Weight
}

type Data struct {
	Weights Weights
}

func Load() ([]float64, []float64, []float64, []float64, []float64, []float64, []float64, []float64, []float64, []float64, []float64, []float64) {
	// Open our jsonFile
	// jsonFile, err := os.Open(filepath.Join("weight", "weights.json"))
	// if we os.Open returns an error then handle it
	jsonFile, err := openuri.Open("http://localhost:3000/weights.json")

	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully Opened users.json")
	// defer the closing of our jsonFile so that we can parse it later on
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var data Data
	json.Unmarshal([]byte(byteValue), &data)
	log.Print(data.Weights.Layer_0.Bias.Shape)

	return data.Weights.Layer_0.Weight.Value, data.Weights.Layer_0.Bias.Value,
		data.Weights.Layer_1.Weight.Value, data.Weights.Layer_1.Bias.Value,
		data.Weights.Layer_2.Weight.Value, data.Weights.Layer_2.Bias.Value,
		data.Weights.Layer_3.Weight.Value, data.Weights.Layer_3.Bias.Value,
		data.Weights.Layer_4.Weight.Value, data.Weights.Layer_4.Bias.Value,
		data.Weights.Layer_5.Weight.Value, data.Weights.Layer_5.Bias.Value
}
