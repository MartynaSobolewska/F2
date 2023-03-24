# F2 - grammar based fuzzer

## Intro

`F2` is a grammar based fuzzzer that building on the advances made by `F1` and `fzero` fuzzers. 

- F1 is an academic fuzzer built to demonstrate how to improve efficiency of grammar fuzzers. Its paper explains concept of compiled grammar using C and Assembly to build the fuzzer. At the time of writing, it was by a large margin the fastest publically available grammar fuzzer. You can find more information about it here:
  - the paper - "Building Fast Fuzzers" by Rahul Gopinath and Andreas Zeller. https://arxiv.org/pdf/1911.07707.pdf
  - the code - https://github.com/vrthra/F1
  
- Fzero is a fuzzer inspired by F1, which F2 directly improves upon. It managed to beat F1's efficiency in most cases, using Rust's flexibility to compile the given grammar easily independent of a platform. It managed to improve on F1's performance achieving up to 5x speedup with a much more maintainable, short code. You can find more information about it here:
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

