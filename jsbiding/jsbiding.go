package jsbiding

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/jpeg"
	"image/png"
	"log"
	"strings"
	"syscall/js"
	// "syscall/js"
)

func GetImage() ([]float64, []float64) {
	doc := js.Global().Get("document")
	image_element := doc.Call("getElementById", "predict_image")
	image_base64 := image_element.Get("src")
	coI := strings.Index(image_base64.String(), ",")
	raw_image := image_base64.String()[coI+1:]

	unbased, _ := base64.StdEncoding.DecodeString(string(raw_image))
	res := bytes.NewReader(unbased)

	var rs_image image.Image
	switch strings.TrimSuffix(image_base64.String()[5:coI], ";base64") {
	case "image/png":
		pngI, err := png.Decode(res)
		if err != nil {
			log.Printf("err ", err)
		}
		rs_image = pngI
		// rs_image = resize.Resize(32, 32, pngI, resize.Bilinear)
	case "image/jpeg":
		jpgI, err := jpeg.Decode(res)
		if err != nil {
			log.Printf("err ", err)
		}
		rs_image = jpgI
		// rs_image = resize.Resize(32, 32, jpgI, resize.Bilinear)
	}
	image_array, _ := getPixels(rs_image)
	var image_1d []float64
	for i := 0; i < 32; i++ {
		for j := 0; j < 32; j++ {
			image_1d = append(image_1d, float64(image_array[i][j].R)/255)
			image_1d = append(image_1d, float64(image_array[i][j].G)/255)
			image_1d = append(image_1d, float64(image_array[i][j].B)/255)
		}
	}
	return image_1d, image_1d
}

// Get the bi-dimensional pixel array
func getPixels(img image.Image) ([][]Pixel, error) {

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	var pixels [][]Pixel
	for y := 0; y < height; y++ {
		var row []Pixel
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			row = append(row, rgbaToPixel(r, g, b))
		}
		pixels = append(pixels, row)
	}

	return pixels, nil
}

// img.At(x, y).RGBA() returns four uint32 values; we want a Pixel
func rgbaToPixel(r uint32, g uint32, b uint32) Pixel {
	return Pixel{int(r / 257), int(g / 257), int(b / 257)}
}

// Pixel struct example
type Pixel struct {
	R int
	G int
	B int
}
