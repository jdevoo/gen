To use `gen`, you will need a valid Gemini API key set in the `GEMINI_API_KEY` environment variable.
> [!NOTE]
If you don't already have one, go to [Google AI Studio](https://ai.google.dev/tutorials/setup) to create a key.

## Single-turn Examples
Plain text generation  
`$ gen dix noms de fleurs`

At the prompt, best to quote the prompt with characters mean something to the shell  
`$ gen "a scheme function to compute the Levenshtein distance between two strings; only the code"`

Pipe content into gen  
`$ cat cmd.go | gen what does this code do? | tee report.txt`

Obtain the token count  
`$ gen -t how many tokens would this prompt require?`

Parameterize prompts  
`gen -p a=1 -p b=2 "replace {a} apple with {a} banana and {b} oranges for a good ..."`

Set a system instruction and submit prompt as argument  
`echo "you understand english but always reply in french" | gen -s ten names for flowers`

Attach a file to the prompt and return total token count  
`gen -t -f ./photo.png what is this picture about?`

Enter chat mode to generate various SQL statements  
`cat classicmodels.sql | gen -c`

Generate an architecture decision record using a parameterized template  
`cat adr.prompt | gen -p certified="certified AWS Solution Architect Professional" -s an architecture decision record to help my organization decide between storage technologies for storing and accessing 10TB worth of time series data - please provide concrete examples and figures where possible`

List known Gemini models by invoking tool  
`gen -tool known models`

> [!NOTE]
The -tool flag relies on Gemini API's Function Calling feature which is in Beta release.

## Usage
```
Usage: gen [options] <prompt>

Command-line interface to Google Gemini large language models
  Requires a valid GEMINI_API_KEY environment variable set
  The prompt is set from stdin and/or arguments.

Options:
  -V	output model | maxInputTokens | maxOutputTokens | temp | top_p | top_k
  -c	enter chat mode using prompt
    	enter 2 consecutive blank lines to exit
  -f string
    	attach file to prompt
  -h	show this help message and exit
  -json
    	response uses the application/json MIME type
  -m string
    	generative model name (default "gemini-1.5-flash")
  -p value
    	prompt parameter in format key=val
    	replace all occurrences of {key} in prompt with val
  -s	treat prompt as system instruction
    	stdin used if found
  -t	output number of tokens for prompt
  -temp float
    	changes sampling during response generation [0.0,2.0] (default 1)
  -tool
    	invoke one of the tools {KnownModels,QueryPostgres}
  -top_p float
    	change how the model selects tokens for generation [0.0,1.0] (default 0.95)
  -unsafe
    	force generation when gen aborts with FinishReasonSafety
  -v	show version and exit
```

## License
This project is licensed under the MIT License.