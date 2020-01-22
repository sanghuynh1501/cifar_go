package weight

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/utahta/go-openuri"
)

type Weight struct {
	Shape []int32
	Value []float64
}

type Data struct {
	Weights []Weight
}

func Load() []Weight {
	jsonFile, err := openuri.Open("http://localhost:3000/weights.json")

	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully Opened users.json")
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var data Data
	json.Unmarshal([]byte(byteValue), &data)

	return data.Weights
}
