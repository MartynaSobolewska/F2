# F2 - grammar based fuzzer

## Intro

`F2` is a grammar based fuzzzer that builds on the advances made by `F1` and `fzero` fuzzers. 

- F1 is an academic fuzzer built to demonstrate how to improve efficiency of grammar fuzzers. Its paper explains concept of compiled grammar using C and Assembly to build the fuzzer. At the time of writing, it was by a large margin the fastest publically available grammar fuzzer. You can find more information about it here:
  - the paper - "Building Fast Fuzzers" by Rahul Gopinath and Andreas Zeller. https://arxiv.org/pdf/1911.07707.pdf
  - the code - https://github.com/vrthra/F1
  
- Fzero is a fuzzer inspired by F1, which F2 directly improves upon. It managed to beat F1's efficiency in most cases, using Rust's flexibility to compile the given grammar easily independent of a platform. It managed to improve on F1's performance achieving up to 5x speedup with a much more maintainable, short code. In the process of improving efficiency, F1 lost the full correctness of produced inputs. You can find more information about it here:
  - two videos documenting the entire development process (total approx. 14h)
    - part 1 - https://youtu.be/ZfuRDwEUg_Q
    - part 2 - https://youtu.be/Rusl0oI36eo
  - the code - https://github.com/gamozolabs/fzero_fuzzer
  
 F2 improves on Fzero introducing stepcount method allowing to find the shortest path to correct completion once max_depth is exceeded. It introduces new optimisation techniques allowing to boost performance once max_depth is exceeded, while always assuring full correctness unless specified otherwise.

## Literature Review, Final Report and Readme
All three of the documents listed below are a part of my dissertation and provide different information depending on what you're interested in.

- `Literature Review` - useful for: general fuzzer research. A summary of background research and project outline. It compiles information about fuzzer types, history, usefulness and brief outline of F2's deliverables.
- `Final Report` - useful for: understanding optimisation techniques for grammar fuzzers. A paper documenting the development process and results achieved by F2. It goes into detail of how does F2 work, what are the possible improvements in the future.
- `Readme` - useful for: learning how to run F2, understand core purpose of the project.

## Running instructions and parameters

usage: f2 <grammar json> <output Rust file> <output binary name> <max depth>

example usage:
```
> cargo run --release html.json test.rs test.exe 128
    Finished release [optimized] target(s) in 3.43s
     Running `target/release/F2 html.json test.rs test.exe 128`
  
Loaded grammar json

Converted grammar to a Rust structure

Number of single-term expressions simplified: 533
Number of single-term non-terminals simplified: 106
Number of NOPs found: 33

Optimised grammar

Couldn't find shortest path for 44 out of 1434 fragments. If the number is too large, consider rewriting a less recursive grammar.

Found shortest completion paths in grammar.
Skipped a total of 12 fragments in 10 shortest paths

Generated Rust source file
Created Rust binary!

> ./test.exe

MiB/sec:     277.9420
MiB/sec:     290.9204
MiB/sec:     295.4913
MiB/sec:     297.8168
MiB/sec:     299.0855
```

the repository contains four example grammars: html.json, css.json, expressions.json and json.json, which can all be used to see the demo of F2's execution.
  
## Probability
  
To add a probability to a fragment in a grammar, append "p=x" to the end of an expression in a non-terminal for example:
  
```
  "<digit>": [
    ["0"], ["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9", "p=100"]
  ]
```
digit 9 will now be a 100 times morer likely to get chosen before max_depth is exceeded.
  
For more information on grammar formatting check out `Literature Review`
  
## SAFE_ONLY and FULL_CORRECTNESS
  
Those two parameters can be set to true or false depending on your needs. You can find them at the top of main.rs file.
  
- SAFE_ONLY - if set to false makes F2 produce unsafe Rust to add strings to the buffer. First introduced by Gamaozolabs in Fzero fuzzer, it is not known to produce any unexpected behaviour, producing performance boost in some cases.
- FULL_CORRECTNESS - when set to true will always finish unwrapping all elements left in the stack using the path of shortest completion. Setting it to false can improve pefromance in some cases but comes at a price of full input correctness.
  
I would advise playing around with the two to achieve the optimal performance for your usecase. For more detail on their impact on input production, see `Literature Review`
  
## Throughput measurements and comparison to F1 and Fzero
  
### HTML throughput measurements for Fzero, F1 and F1. where C stands for FULL_CORRECTNESS, S for SAFE_ONLY, t for true, f for false. Measured on AMD Ryzen 9 5900X (x86)
  
| depth |   F0 S=f   |   F0 S=t   |     F1    | F2 S=f, C=f | F2 S=f, C=t | F2 S=t, C=f | F2 S=t, C=t |
|:-----:|:----------:|:----------:|:---------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 4     | 3425 MiB/s | 3425 MiB/s | 832 MiB/s | 3356 MiB/s  | 2652 MiB/s  | 3435 MiB/s  | 2619 MiB/s  |
| 8     | 2155 MiB/s | 2165 MiB/s | 425 MiB/s | 2144 MiB/s  | 907 MiB/s   | 2032 MiB/s  | 1167 MiB/s  |
| 16    | 419 MiB/s  | 448 MiB/s  | 442 MiB/s | 452 MiB/s   | 371 MiB/s   | 547 MiB/s   | 531 MiB/s   |
| 32    | 276 MiB/s  | 276 MiB/s  | 221 MiB/s | 279 MiB/s   | 260 MiB/s   | 388 MiB/s   | 402 MiB/s   |
| 64    | 254 MiB/s  | 261 MiB/s  | 220 MiB/s | 256 MiB/s   | 248 MiB/s   | 368 MiB/s   | 393 MiB/s   |
  
### json throughput measurements for Fzero, F1 and F1. where C stands for FULL_CORRECTNESS, S for SAFE_ONLY, t for true, f for false. Measured on AMD Ryzen 9 5900X (x86)
  
| depth |   F0 S=f  |   F0 S=t  |     F1    | F2 S=f, C=f | F2 S=f, C=t | F2 S=t, C=f | F2 S=t, C=t |
|:-----:|:---------:|:---------:|:---------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 4     | 118 MiB/s | 111 MiB/s | 117 MiB/s | 121 MiB/s   | 123 MiB/s   | 182 MiB/s   | 144 MiB/s   |
| 8     | 94 MiB/s  | 98 MiB/s  | 90 MiB/s  | 106 MiB/s   | 114 MiB/s   | 137 MiB/s   | 136 MiB/s   |
| 16    | 100 MiB/s | 101 MiB/s | 107 MiB/s | 107 MiB/s   | 115 MiB/s   | 133 MiB/s   | 136 MiB/s   |
| 32    | 100 MiB/s | 103 MiB/s | 103 MiB/s | 108 MiB/s   | 115 MiB/s   | 131 MiB/s   | 142 MiB/s   |
| 64    | 104 MiB/s | 105 MiB/s | 110 MiB/s | 113 MiB/s   | 115 MiB/s   | 126 MiB/s   | 139 MiB/s   |
