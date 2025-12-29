# Ferrum

**Ferrum** is a high-performance, UCI-compatible chess engine written in Rust. It combines modern search techniques with the state-of-the-art NNUE (Efficiently Updatable Neural Network) evaluation function to play high-quality chess.

## Key Features

### 🧠 Evaluation
- **NNUE Technology**: Uses a neural network for static position evaluation, providing deep understanding of positional play while maintaining high speed.
- **Embedded Network**: The engine comes pre-packaged with a default network file (`network.nnue`) in the root directory, so it's ready to use out of the box.
- **Module**: Powered by `ferrum-nnue`, a custom optimized implementation of the NNUE inference code.

### 🔍 Search Algorithms
- **Principal Variation Search (PVS)**: Efficient alpha-beta search variant.
- **Iterative Deepening**: Progressive search depth for better time management.
- **Lazy SMP**: Multithreaded search scaling to utilize modern multi-core processors.
- **Transposition Table**: Caches positions to avoid re-searching identical subtrees.
- **Quiescence Search**: Resolves tactical sequences to avoid the horizon effect.

### ✂️ Pruning & Optimizations
Ferrum employs aggressive pruning techniques to reduce the search space without sacrificing strength:
- **Reverse Futility Pruning (RFP)**: Early cutoffs for positions with large static advantages.
- **Null Move Pruning**: Dynamic reduction based on depth.
- **Late Move Reductions (LMR)**: Variable reductions for quiet moves.
- **History Pruning**: Prunes quiet moves that historically perform poorly.
- **SEE Pruning**: Uses Static Exchange Evaluation to prune losing captures and quiet moves.
- **Futility Pruning & Razoring**: Methods to prune branches at low depths.
- **ProbCut**: Probabilistic cutoffs for likely winning lines.
- **Move Ordering**: Optimizes search order using MVP-LVA, Killers, Countermoves, and History Heuristics.

### ⚡ Move Generation
- Built on **ferrum-movegen**, a fast localized move generation library designed specifically for this engine.

## Getting Started

### Prerequisites
- [Rust Toolchain](https://www.rust-lang.org/tools/install) (stable)

### Installation
1. Clone the repository:
   ```bash
   git clone --recursive https://github.com/TheChii/Ferrum.git
   cd Ferrum
   ```
2. Build in release mode:
   ```bash
   cargo build --release
   ```
   The executable will be located in `target/release/chessinrust.exe` (or `chessinrust` on Linux/Mac).

### Usage
Ferrum is a command-line engine that speaks the UCI (Universal Chess Interface) protocol. It is not designed to be played against directly in a terminal. Instead, install a chess GUI such as:
- **Arena**
- **CuteChess**
- **BanksiaGUI**
- **En Croissant**

Point your GUI to the compiled executable.

#### Common UCI Commands
- `uci`: Initialize communication.
- `isready`: Check readiness.
- `ucinewgame`: Reset for a new game.
- `position startpos moves ...`: Set board state.
- `go wtime <ms> btime <ms> ...`: Start searching.

## NNUE File
The engine requires `network.nnue` to run. This file is **already included** in the main directory. 

**IMPORTANT**: When running the engine (e.g., in a GUI), ensuring `network.nnue` is in the **same directory as the executable** or in the root directory if running via `cargo`. If you build the release version, copy `network.nnue` into `target/release/` alongside `chessinrust.exe`.

## License
MIT
