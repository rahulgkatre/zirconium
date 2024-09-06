# Zirconium

Zirconium is an embedded domain specific language (eDSL) for defining nested loops and executing them. Its API is similar to other libraries and DSLs like Halide, TVM, and Accera, but its differentiating feature is that the loops can be executed without an explicit code generation step.

This is done through the power of `comptime` in Zig, where loop parameters and transforms are defined in comptime and stored in datastructures. Evaluation / execution resolves comptime values as constants, allowing for zero overhead execution. To validate this, try building with `-OReleaseSmall` on loop code and export the assembly.

This library is to be interfaced with [Tesseract](https://github.com/rahulgkatre/tesseract) and be used as an intermediate representation for compiling tensor graphs. 

## Core Principles

### Verification During Compilation

- Use type system to keep track of shapes and function types
- Verify that shapes and functions are valid
- Invalid operations will fail to compile and provide helpful error messages

### Optimized Code Generation

- Execute loops with zero overhead (even from multi dimensional indexing)
- Support various hardware such as vector registers, matrix multiplication hardware, GPU

### Minimal Dependencies

- Minimize compile dependencies by only using the Zig compiler (as the language develops, use it more)
- Minimize executable dependencies (including standard library)

## Roadmap

### Minimum Feature Checklist

- Basic loop transforms
    - [x] Split
    - [x] Vectorize
    - [x] Unroll
    - [x] Reorder
    - [ ] Skew
- Advanced transforms
    - [ ] Loop fusion
    - [ ] Tensorization
    - [ ] GPU kernelization
- Other
    - [ ] Runtime-known loop sizes

### Future goals

- Autotuning
    - Find different options for optimization that result in the highest performance based on some input types
- Polyhedral compilation
    - Automatically calculate transforms using polyhedral compilation techniques
