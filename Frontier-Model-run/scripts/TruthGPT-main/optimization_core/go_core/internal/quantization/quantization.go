// Package quantization provides high-performance tensor quantization utilities.
package quantization

import (
	"encoding/binary"
	"errors"
	"math"
	"sync"

	"go.uber.org/zap"
)

type QuantizationType int

const (
	QuantFP32 QuantizationType = iota
	QuantFP16
	QuantBF16
	QuantINT8
	QuantINT4
	QuantNF4
)

func (q QuantizationType) String() string {
	switch q {
	case QuantFP32:
		return "fp32"
	case QuantFP16:
		return "fp16"
	case QuantBF16:
		return "bf16"
	case QuantINT8:
		return "int8"
	case QuantINT4:
		return "int4"
	case QuantNF4:
		return "nf4"
	default:
		return "unknown"
	}
}

func (q QuantizationType) BytesPerElement() float32 {
	switch q {
	case QuantFP32:
		return 4.0
	case QuantFP16, QuantBF16:
		return 2.0
	case QuantINT8:
		return 1.0
	case QuantINT4, QuantNF4:
		return 0.5
	default:
		return 4.0
	}
}

type QuantizationConfig struct {
	Type           QuantizationType `yaml:"type"`
	PerChannel     bool             `yaml:"per_channel"`
	Symmetric      bool             `yaml:"symmetric"`
	GroupSize      int              `yaml:"group_size"`
	ClipRange      float32          `yaml:"clip_range"`
	CalibrationSamples int          `yaml:"calibration_samples"`
}

func DefaultQuantizationConfig() QuantizationConfig {
	return QuantizationConfig{
		Type:       QuantINT8,
		PerChannel: true,
		Symmetric:  true,
		GroupSize:  128,
		ClipRange:  3.0,
		CalibrationSamples: 512,
	}
}

type QuantizedTensor struct {
	Data       []byte
	Scale      []float32
	ZeroPoint  []int32
	Shape      []int64
	OrigType   QuantizationType
	QuantType  QuantizationType
	GroupSize  int
}

func (t *QuantizedTensor) SizeBytes() int {
	return len(t.Data) + len(t.Scale)*4 + len(t.ZeroPoint)*4
}

func (t *QuantizedTensor) CompressionRatio() float32 {
	origSize := float32(1)
	for _, dim := range t.Shape {
		origSize *= float32(dim)
	}
	origSize *= t.OrigType.BytesPerElement()
	
	return origSize / float32(t.SizeBytes())
}

type Quantizer struct {
	config QuantizationConfig
	logger *zap.Logger
	
	scaleCache sync.Map
}

func NewQuantizer(config QuantizationConfig, logger *zap.Logger) *Quantizer {
	if logger == nil {
		logger, _ = zap.NewDevelopment()
	}
	return &Quantizer{
		config: config,
		logger: logger,
	}
}

func (q *Quantizer) Quantize(data []float32, shape []int64) (*QuantizedTensor, error) {
	if len(data) == 0 {
		return nil, errors.New("empty input data")
	}
	
	switch q.config.Type {
	case QuantINT8:
		return q.quantizeInt8(data, shape)
	case QuantINT4:
		return q.quantizeInt4(data, shape)
	case QuantFP16:
		return q.quantizeFP16(data, shape)
	default:
		return nil, errors.New("unsupported quantization type")
	}
}

func (q *Quantizer) Dequantize(tensor *QuantizedTensor) ([]float32, error) {
	if tensor == nil {
		return nil, errors.New("nil tensor")
	}
	
	switch tensor.QuantType {
	case QuantINT8:
		return q.dequantizeInt8(tensor)
	case QuantINT4:
		return q.dequantizeInt4(tensor)
	case QuantFP16:
		return q.dequantizeFP16(tensor)
	default:
		return nil, errors.New("unsupported quantization type")
	}
}

func (q *Quantizer) quantizeInt8(data []float32, shape []int64) (*QuantizedTensor, error) {
	groupSize := q.config.GroupSize
	if groupSize <= 0 {
		groupSize = len(data)
	}
	
	numGroups := (len(data) + groupSize - 1) / groupSize
	
	quantized := make([]byte, len(data))
	scales := make([]float32, numGroups)
	zeroPoints := make([]int32, numGroups)
	
	for g := 0; g < numGroups; g++ {
		start := g * groupSize
		end := start + groupSize
		if end > len(data) {
			end = len(data)
		}
		
		group := data[start:end]
		
		minVal, maxVal := findMinMax(group)
		
		if q.config.Symmetric {
			absMax := max32(abs32(minVal), abs32(maxVal))
			scale := absMax / 127.0
			if scale == 0 {
				scale = 1.0
			}
			scales[g] = scale
			zeroPoints[g] = 0
			
			for i, v := range group {
				qv := clampInt(int(math.Round(float64(v / scale))), -128, 127)
				quantized[start+i] = byte(qv)
			}
		} else {
			scale := (maxVal - minVal) / 255.0
			if scale == 0 {
				scale = 1.0
			}
			zp := int32(math.Round(float64(-minVal / scale)))
			scales[g] = scale
			zeroPoints[g] = zp
			
			for i, v := range group {
				qv := clampInt(int(math.Round(float64(v/scale)))+int(zp), 0, 255)
				quantized[start+i] = byte(qv)
			}
		}
	}
	
	return &QuantizedTensor{
		Data:      quantized,
		Scale:     scales,
		ZeroPoint: zeroPoints,
		Shape:     shape,
		OrigType:  QuantFP32,
		QuantType: QuantINT8,
		GroupSize: groupSize,
	}, nil
}

func (q *Quantizer) dequantizeInt8(tensor *QuantizedTensor) ([]float32, error) {
	result := make([]float32, len(tensor.Data))
	groupSize := tensor.GroupSize
	if groupSize <= 0 {
		groupSize = len(tensor.Data)
	}
	
	numGroups := len(tensor.Scale)
	
	for g := 0; g < numGroups; g++ {
		start := g * groupSize
		end := start + groupSize
		if end > len(tensor.Data) {
			end = len(tensor.Data)
		}
		
		scale := tensor.Scale[g]
		zp := tensor.ZeroPoint[g]
		
		for i := start; i < end; i++ {
			if q.config.Symmetric {
				result[i] = float32(int8(tensor.Data[i])) * scale
			} else {
				result[i] = (float32(tensor.Data[i]) - float32(zp)) * scale
			}
		}
	}
	
	return result, nil
}

func (q *Quantizer) quantizeInt4(data []float32, shape []int64) (*QuantizedTensor, error) {
	groupSize := q.config.GroupSize
	if groupSize <= 0 {
		groupSize = len(data)
	}
	
	numGroups := (len(data) + groupSize - 1) / groupSize
	
	quantized := make([]byte, (len(data)+1)/2)
	scales := make([]float32, numGroups)
	
	for g := 0; g < numGroups; g++ {
		start := g * groupSize
		end := start + groupSize
		if end > len(data) {
			end = len(data)
		}
		
		group := data[start:end]
		_, maxVal := findMinMax(group)
		absMax := abs32(maxVal)
		
		scale := absMax / 7.0
		if scale == 0 {
			scale = 1.0
		}
		scales[g] = scale
		
		for i := 0; i < len(group); i += 2 {
			v1 := clampInt(int(math.Round(float64(group[i]/scale))), -8, 7)
			v2 := 0
			if i+1 < len(group) {
				v2 = clampInt(int(math.Round(float64(group[i+1]/scale))), -8, 7)
			}
			
			b1 := byte((v1 + 8) & 0x0F)
			b2 := byte((v2 + 8) & 0x0F)
			quantized[(start+i)/2] = (b2 << 4) | b1
		}
	}
	
	return &QuantizedTensor{
		Data:      quantized,
		Scale:     scales,
		ZeroPoint: nil,
		Shape:     shape,
		OrigType:  QuantFP32,
		QuantType: QuantINT4,
		GroupSize: groupSize,
	}, nil
}

func (q *Quantizer) dequantizeInt4(tensor *QuantizedTensor) ([]float32, error) {
	totalElements := 1
	for _, dim := range tensor.Shape {
		totalElements *= int(dim)
	}
	
	result := make([]float32, totalElements)
	groupSize := tensor.GroupSize
	if groupSize <= 0 {
		groupSize = totalElements
	}
	
	idx := 0
	for g := 0; g < len(tensor.Scale); g++ {
		scale := tensor.Scale[g]
		start := g * groupSize
		end := start + groupSize
		if end > totalElements {
			end = totalElements
		}
		
		for i := start; i < end && idx < len(result); i += 2 {
			byteIdx := i / 2
			if byteIdx >= len(tensor.Data) {
				break
			}
			
			b := tensor.Data[byteIdx]
			v1 := int(b&0x0F) - 8
			v2 := int((b>>4)&0x0F) - 8
			
			result[idx] = float32(v1) * scale
			idx++
			if idx < len(result) {
				result[idx] = float32(v2) * scale
				idx++
			}
		}
	}
	
	return result, nil
}

func (q *Quantizer) quantizeFP16(data []float32, shape []int64) (*QuantizedTensor, error) {
	quantized := make([]byte, len(data)*2)
	
	for i, v := range data {
		fp16 := float32ToFloat16(v)
		binary.LittleEndian.PutUint16(quantized[i*2:], fp16)
	}
	
	return &QuantizedTensor{
		Data:      quantized,
		Scale:     []float32{1.0},
		ZeroPoint: nil,
		Shape:     shape,
		OrigType:  QuantFP32,
		QuantType: QuantFP16,
		GroupSize: len(data),
	}, nil
}

func (q *Quantizer) dequantizeFP16(tensor *QuantizedTensor) ([]float32, error) {
	numElements := len(tensor.Data) / 2
	result := make([]float32, numElements)
	
	for i := 0; i < numElements; i++ {
		fp16 := binary.LittleEndian.Uint16(tensor.Data[i*2:])
		result[i] = float16ToFloat32(fp16)
	}
	
	return result, nil
}

func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 16) & 0x8000
	exp := (bits >> 23) & 0xFF
	mant := bits & 0x7FFFFF
	
	if exp == 0xFF {
		if mant != 0 {
			return uint16(sign | 0x7E00)
		}
		return uint16(sign | 0x7C00)
	}
	
	if exp == 0 {
		return uint16(sign)
	}
	
	newExp := int(exp) - 127 + 15
	
	if newExp >= 31 {
		return uint16(sign | 0x7C00)
	}
	
	if newExp <= 0 {
		if newExp < -10 {
			return uint16(sign)
		}
		mant |= 0x800000
		shift := uint(14 - newExp)
		return uint16(sign | uint32(mant>>shift))
	}
	
	return uint16(sign | uint32(newExp<<10) | (mant >> 13))
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h & 0x03FF)
	
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for mant&0x0400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= ^uint32(0x0400)
	} else if exp == 31 {
		if mant == 0 {
			return math.Float32frombits(sign | 0x7F800000)
		}
		return math.Float32frombits(sign | 0x7FC00000)
	}
	
	exp = exp + 127 - 15
	mant = mant << 13
	
	return math.Float32frombits(sign | (exp << 23) | mant)
}

func findMinMax(data []float32) (min, max float32) {
	if len(data) == 0 {
		return 0, 0
	}
	min, max = data[0], data[0]
	for _, v := range data[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func clampInt(v, minV, maxV int) int {
	if v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
}

type QuantizationStats struct {
	OriginalSize   int64
	QuantizedSize  int64
	Ratio          float32
	RMSE           float32
	MaxError       float32
	QuantType      QuantizationType
}

func ComputeStats(original []float32, dequantized []float32, quantType QuantizationType) QuantizationStats {
	if len(original) != len(dequantized) {
		return QuantizationStats{}
	}
	
	var sumSquaredError float64
	var maxError float32
	
	for i := range original {
		diff := original[i] - dequantized[i]
		absDiff := abs32(diff)
		if absDiff > maxError {
			maxError = absDiff
		}
		sumSquaredError += float64(diff * diff)
	}
	
	rmse := float32(math.Sqrt(sumSquaredError / float64(len(original))))
	
	origSize := int64(len(original)) * 4
	quantSize := int64(float32(len(original)) * quantType.BytesPerElement())
	
	return QuantizationStats{
		OriginalSize:  origSize,
		QuantizedSize: quantSize,
		Ratio:         float32(origSize) / float32(quantSize),
		RMSE:          rmse,
		MaxError:      maxError,
		QuantType:     quantType,
	}
}
