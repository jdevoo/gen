To use `gen`, you will need a valid Gemini API key set in the `GEMINI_API_KEY` environment variable.
> [!NOTE]
If you don't already have one, go to [Google AI Studio](https://ai.google.dev/tutorials/setup) to create a key.

## Single-turn Examples
Plain text generation  
`$ gen dix noms de fleurs`

At the command line, you may have to put the prompt inside double quotes to avoid confusing the shell  
`$ gen "a scheme function to compute the Levenshtein distance between two strings; only the code"`

Pipe content into gen and save its output in a text file  
`$ cat cmd.go | gen -f - what does this code do? | tee report.txt`

Obtain the token count  
`$ gen -t how many tokens would this prompt consume?`

Parameterize prompts  
`gen -p a=1 -p b=2 "complete this sentence: replace {a} apple with {a} banana and {b} oranges for a good ..."`

Set a system instruction from stdin and prompt from argument  
`echo "you understand english but always reply in french" | gen -s -f - ten names for flowers`

Set a system instruction from file option and prompt from argument  
`gen -f french.sprompt ten names for flowers"`

Set a system instruction from argument and enter chat  
`gen -c -s you understand english but always answer in German`

Attach two files to the prompt and return total token count  
`gen -t -f ../twitter/img/123497680.jpg -f ../twitter/img/123406895.jpg what are the differences between these photos?`

Enter chat mode to generate various SQL statements  
`cat classicmodels.sql | gen -f - -c`

Generate an [architecture decision record](https://adr.github.io/madr/) using a parameterized template  
`gen -p role="an experienced MLOps architect" -f prompts/adr.sprompt "an architecture decision record for selecting between option 1 MLFlow and option 2 Weights & Biases for tracking data science experiments. The team includes 5 data scientists and maintains already 50 models in production. The team produces 5 new models per quarter. Models include regression models, classification models, computer vision models as well as time series models."`

List known Gemini models by invoking tool  
`gen list known generative models`

Extract entities from text  
`w3m -dump https://lite.cnn.com/2024/07/27/asia/us-austin-trilateral-japan-south-korea-intl-hnk/index.html | gen -json -f - extract entities`

System instruction and prompts as files from iterative Prisonner's Dilemma [paper](https://arxiv.org/html/2406.13605v1)  
`gen -json -f pd.sprompt -f pd.prompt`

Chain of thought  
`gen Please answer this question starting in two ways. First start with yes, then start with no and show your work. Afterwards determine which is correct. Is 3307 a prime number?`

Generate sequence diagram from code using PlantUML system instruction  
`gen -f puml.sprompt -f ~/lib/Duke/duke-core/src/main/java/no/priv/garshol/duke/Duke.java sequence diagram of the main_ method in this file`

Basic information extraction (Google Gemini [cookbook example](https://github.com/google-gemini/cookbook/tree/main))  
`gen -f extract.sprompt -f extract.prompt | gen -f format.sprompt -`

## Multi-turn Example
Generate a brief using an adapted version of Ali Abassi's prompt  
`gen -c -f brief.sprompt -f brief.prompt -p role="Sr. Business Analyst" -p department="ACME Technology Solutions" -p task="create a project brief" -p deliverable="project brief"`

Tree of thought  
`gen -c -f tot.sprompt -f tot.prompt`

## Retrieval Augmented Generation
Use Gemini embedding models to encode text chunks for retrieval augmented generation. [Maximal marginal relevance](mmr.pdf) is used to rank chunks up to a default limit. The text retrieved is prepended to prompts. Altnernatively, use the digest key inside the prompt to position retrieved chunks. Digest files are append-only named `00000000000000000001` and incremented as soon as the limit of 20MB is reached. Persistence logic is adapted from Farhan's [aol](https://github.com/arriqaaq/aol).

Save text chunk to digest  
`gen -e -d /tmp "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the \"Navigation\" icon to get directions to your destination or touch the \"Music\" icon to play your favorite songs."`

Query from digest  
`gen -d /tmp How do I play music in the Google car?`

Add document to digest  
`pdftotext Attali.pdf - | awk 'BEGIN{RS='\f'} {cmd="gen -V -e -f - -d digest"; print | cmd; close(cmd)}'`

Query digest and read out loud using TTS system  
`echo you understand french but always reply in english | gen -s -f - -d digest liste les 30 principales propositions de Jacques Attali | ../Downloads/piper/piper --model ../Downloads/voices/en_US-hfc_female-medium.onnx --output-raw | aplay -r 22050 -f S16_LE -t raw -`

## Usage
```
Usage: gen [options] <prompt>

Command-line interface to Google Gemini large language models
  Requires a valid GEMINI_API_KEY environment variable set.
  Content is generated by a prompt and optional system instructions.
  Use - to assign stdin as prompt argument or as attached file.

Tools:
  * ListAWSServices
  * ListKnownGeminiModels

Options:

  -V	output model details, system instructions and chat history
    	details include model name | maxInputTokens | maxOutputTokens | temp | top_p | top_k
  -c	enter chat mode after content generation
    	type two consecutive blank lines to exit
    	not supported on windows when stdin used
  -d value
    	path to a digest folder
    	use {digest} to inject retrieved chunks into prompt
    	repeat for every digest to query
  -e	write embeddings to digest (default model "text-embedding-004")
    	use with a single digest path
  -f value
    	file to attach where value is the path to the file
    	use extensions .prompt and .sprompt for user and system instructions respectively
    	repeat for every file
  -h	show this help message and exit
  -json
    	response in JavaScript Object Notation
  -k int
    	maximum number of entries from digest to retrieve (default 3)
  -l float
    	trade off accuracy for diversity when querying digests [0.0,1.0] (default 0.5)
  -m string
    	embedding or generative model name (default "gemini-1.5-flash")
  -o	only store metadata with embeddings and ignore the content
  -p value
    	prompt parameter value in format key=val
    	replaces all occurrences of {key} in prompt with val
    	repeat for every parameter
    	used as metadata when computing embeddings
  -s	treat argument as system instruction
    	unless stdin is set as file
  -t	output total number of tokens
  -temp float
    	changes sampling during response generation [0.0,2.0] (default 1)
  -top_p float
    	changes how the model selects tokens for generation [0.0,1.0] (default 0.95)
  -unsafe
    	force generation when gen aborts with FinishReasonSafety
  -v	show version and exit
```

## License
This project is licensed under the MIT License.
