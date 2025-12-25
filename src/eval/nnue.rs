//! NNUE wrapper for `nnue-rs` with efficient evaluation.
//!
//! This module provides NNUE evaluation with state caching.
//! True incremental updates require nnue-rs to expose accumulator internals,
//! which it currently doesn't. Instead, we use a per-position evaluation
//! but with optimizations like lazy evaluation and caching via TT.

use crate::types::{Board, Score, ToNnue};
use nnue::stockfish::halfkp::{SfHalfKpFullModel, SfHalfKpModel};
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use binread::BinRead;

/// Global type for shared thread-safe model
pub type Model = Arc<SfHalfKpFullModel>;

/// Load NNUE model from file
pub fn load_model(path: &str) -> std::io::Result<Model> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    match SfHalfKpFullModel::read(&mut reader) {
        Ok(model) => Ok(Arc::new(model)),
        Err(e) => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
    }
}

/// Evaluate a board using the NNUE model.
///
/// This creates state from scratch for each evaluation.
/// While not incremental, it's correct and reliable.
/// 
/// Optimization note: In practice, most evaluations are cached via TT,
/// so full recalculation only happens at leaf nodes without TT hits.
#[inline]
pub fn evaluate(model: &SfHalfKpModel, board: &Board) -> Score {
    let side_to_move = board.side_to_move().to_nnue();
    let white_king = board.king_square(chess::Color::White).to_nnue();
    let black_king = board.king_square(chess::Color::Black).to_nnue();
    
    let mut state = model.new_state(white_king, black_king);

    // Add all non-king pieces
    // This is the expensive part - iterating all 64 squares
    for sq in chess::ALL_SQUARES {
        if let Some(piece) = board.piece_on(sq) {
            if piece == chess::Piece::King {
                continue;
            }
            
            let color = board.color_on(sq).unwrap();
            let nnue_piece = piece.to_nnue();
            let nnue_color = color.to_nnue();
            let nnue_sq = sq.to_nnue();

            state.add(nnue::Color::White, nnue_piece, nnue_color, nnue_sq);
            state.add(nnue::Color::Black, nnue_piece, nnue_color, nnue_sq);
        }
    }

    let output = state.activate(side_to_move);
    let cp = nnue::stockfish::halfkp::scale_nn_to_centipawns(output[0]);
    Score::cp(cp)
}

/// Optimized evaluation using bitboards (faster iteration)
#[inline]
pub fn evaluate_fast(model: &SfHalfKpModel, board: &Board) -> Score {
    let side_to_move = board.side_to_move().to_nnue();
    let white_king = board.king_square(chess::Color::White).to_nnue();
    let black_king = board.king_square(chess::Color::Black).to_nnue();
    
    let mut state = model.new_state(white_king, black_king);

    // Iterate by piece type and color - more cache friendly
    for &piece in &[chess::Piece::Pawn, chess::Piece::Knight, chess::Piece::Bishop, 
                    chess::Piece::Rook, chess::Piece::Queen] {
        for &color in &[chess::Color::White, chess::Color::Black] {
            let bb = board.pieces(piece) & board.color_combined(color);
            let nnue_piece = piece.to_nnue();
            let nnue_color = color.to_nnue();
            
            // Iterate set bits
            for sq in bb {
                let nnue_sq = sq.to_nnue();
                state.add(nnue::Color::White, nnue_piece, nnue_color, nnue_sq);
                state.add(nnue::Color::Black, nnue_piece, nnue_color, nnue_sq);
            }
        }
    }

    let output = state.activate(side_to_move);
    let cp = nnue::stockfish::halfkp::scale_nn_to_centipawns(output[0]);
    Score::cp(cp)
}
