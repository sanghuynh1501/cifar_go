package weight

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

type Layer struct {
	Shape []int32
	Value []float64
}

type Weight struct {
	Layer_0 Layer
	Layer_1 Layer
	Layer_2 Layer
	Layer_3 Layer
	Layer_4 Layer
	Layer_5 Layer
}

type Data struct {
	Weights Weight
}

func Load() ([]float64, []float64, []float64, []float64, []float64, []float64) {
	// Open our jsonFile
	jsonFile, err := os.Open(filepath.Join("weight", "weights.json"))
	// if we os.Open returns an error then handle it
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully Opened users.json")
	// defer the closing of our jsonFile so that we can parse it later on
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var data Data
	json.Unmarshal([]byte(byteValue), &data)

	return data.Weights.Layer_0.Value, data.Weights.Layer_1.Value, data.Weights.Layer_2.Value, data.Weights.Layer_3.Value, data.Weights.Layer_4.Value, data.Weights.Layer_5.Value
}
