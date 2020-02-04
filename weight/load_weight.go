package weight

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	"github.com/utahta/go-openuri"
)

type Weight struct {
	Shape []uint32
	Value []float64
}

type Data struct {
	Weights []Weight
}

func Load() []Weight {
	dataFile, err_file_out := openuri.Open("http://localhost:3000/integerdata.gob")
	if err_file_out != nil {
		fmt.Println(err_file_out)
		os.Exit(1)
	}
	dec := gob.NewDecoder(dataFile)
	var m Data
	err := dec.Decode(&m)
	if err != nil {
		log.Fatal("decode error:", err)
	}
	dataFile.Close()

	return m.Weights
}
