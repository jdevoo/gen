package sparql

import (
	"fmt"
	"strings"
	"time"
)

const dateTimeFormat = `"2006-01-02T15:04:05Z07:00"^^xsd:dateTime`

// Param is a parameter to fill placeholders.
type Param struct {
	// If the Name is not empty it should be used for the parameter identifier and
	// not the ordinal position.
	// Name will not have a symbol prefix.
	Name string
	// Ordinal position of the parameter starting from one and is always set.
	Ordinal int
	// Value is the parameter value.
	Value interface{}
	// LanguageTag for monolingual text values.
	LanguageTag string
	// DataTypes for values e.g. item, boolean, string, monolingual text, dates.
	DataType *DataType
}

type DataType struct {
	Prefix string
	Name   string
	Value  interface{}
}

func (dt DataType) Ref() string {
	return fmt.Sprintf("%s:%s", dt.Prefix, dt.Name)
}

// Placeholders returns matching placeholder strings.
func (p Param) Placeholders() []string {
	if p.Name != "" {
		return []string{
			"@" + p.Name,
			fmt.Sprintf("$%d", p.Ordinal),
		}
	}
	return []string{
		fmt.Sprintf("$%d", p.Ordinal),
	}
}

// Serializable serialize data to embed to queries.
type Serializable interface {
	Serialize() string
}

// Serialize returns the serialized literal string.
func (l Literal) Serialize() string {
	s := fmt.Sprint(l.Value)
	s = strings.Replace(s, `"""`, `\"\"\"`, -1)
	if l.LanguageTag != "" {
		return strings.Join([]string{`"""`, s, `"""@`, l.LanguageTag}, "")
	}
	if l.DataType != nil {
		return strings.Join([]string{`"""`, s, `"""^^`, l.DataType.Ref()}, "")
	}
	return strings.Join([]string{`"""`, s, `"""`}, "")
}

// Serialize returns the serialized as query parameter.
// nolint: gocyclo
func (p Param) Serialize() string {
	switch v := p.Value.(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64, bool:
		return fmt.Sprint(v)
	case []byte:
		s := strings.Replace(string(v), `"""`, `\"\"\"`, -1)
		return strings.Join([]string{`"""`, s, `"""`}, "")
	case string:
		s := strings.Replace(v, `"""`, `\"\"\"`, -1)
		if p.LanguageTag != "" {
			return strings.Join([]string{`"""`, s, `"""@`, p.LanguageTag}, "")
		}
		if p.DataType != nil {
			return strings.Join([]string{`"""`, s, `"""^^`, p.DataType.Ref()}, "")
		}
		return strings.Join([]string{`"""`, s, `"""`}, "") // default string literal
	case time.Time:
		return v.Format(dateTimeFormat)
	case DataType:
		s := strings.Replace(fmt.Sprint(v.Value), `"""`, `\"\"\"`, -1)
		return strings.Join([]string{`"""`, s, `"""^^`, v.Ref()}, "")
	case IRIRef:
		return v.Ref()
	case Serializable:
		return v.Serialize()
	default:
		s := strings.Replace(fmt.Sprint(v), `"""`, `\"\"\"`, -1)
		if p.LanguageTag != "" {
			return strings.Join([]string{`"""`, s, `"""@`, p.LanguageTag}, "")
		}
		if p.DataType != nil {
			return strings.Join([]string{`"""`, s, `"""^^`, p.DataType.Ref()}, "")
		}
		return strings.Join([]string{`"""`, s, `"""`}, "")
	}
}
