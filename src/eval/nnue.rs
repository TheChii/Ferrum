//! NNUE wrapper for `nnue-rs` with incremental update support.
//!
//! Uses forked nnue-rs with exposed state for efficient incremental updates.

use crate::types::{Board, Score, ToNnue, Move};
use nnue::stockfish::halfkp::{SfHalfKpFullModel, SfHalfKpModel, SfHalfKpState};
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

/// Create a fresh NNUE state from a board position
pub fn create_state<'m>(model: &'m SfHalfKpModel, board: &Board) -> SfHalfKpState<'m> {
    let white_king = board.king_square(chess::Color::White).to_nnue();
    let black_king = board.king_square(chess::Color::Black).to_nnue();
    
    let mut state = model.new_state(white_king, black_king);

    // Add all non-king pieces using bitboard iteration
    for &piece in &[chess::Piece::Pawn, chess::Piece::Knight, chess::Piece::Bishop, 
                    chess::Piece::Rook, chess::Piece::Queen] {
        for &color in &[chess::Color::White, chess::Color::Black] {
            let bb = board.pieces(piece) & board.color_combined(color);
            let nnue_piece = piece.to_nnue();
            let nnue_color = color.to_nnue();
            
            for sq in bb {
                let nnue_sq = sq.to_nnue();
                state.add(nnue::Color::White, nnue_piece, nnue_color, nnue_sq);
                state.add(nnue::Color::Black, nnue_piece, nnue_color, nnue_sq);
            }
        }
    }
    
    state
}

/// Evaluate using a pre-built state (fast - just runs network)
#[inline]
pub fn evaluate_state(state: &mut SfHalfKpState<'_>, side_to_move: chess::Color) -> Score {
    let output = state.activate(side_to_move.to_nnue());
    let cp = nnue::stockfish::halfkp::scale_nn_to_centipawns(output[0]);
    Score::cp(cp)
}

/// Evaluate from scratch (creates new state)
#[inline]
pub fn evaluate_scratch(model: &SfHalfKpModel, board: &Board) -> Score {
    let mut state = create_state(model, board);
    evaluate_state(&mut state, board.side_to_move())
}

/// Update state for a move (incremental)
/// Returns true if update succeeded, false if full refresh needed
pub fn update_state_for_move(
    state: &mut SfHalfKpState<'_>,
    board: &Board,  // Position BEFORE the move
    mv: Move,
) -> bool {
    let from = mv.get_source();
    let to = mv.get_dest();
    let moving_piece = match board.piece_on(from) {
        Some(p) => p,
        None => return false,
    };
    let moving_color = board.side_to_move();
    let captured = board.piece_on(to);

    // King moves require full refresh (king position changes feature indexing)
    if moving_piece == chess::Piece::King {
        return false;
    }

    let nnue_piece = moving_piece.to_nnue();
    let nnue_color = moving_color.to_nnue();
    let from_sq = from.to_nnue();
    let to_sq = to.to_nnue();

    // Remove piece from old square (both perspectives)
    state.sub(nnue::Color::White, nnue_piece, nnue_color, from_sq);
    state.sub(nnue::Color::Black, nnue_piece, nnue_color, from_sq);

    // Handle capture
    if let Some(captured_piece) = captured {
        if captured_piece != chess::Piece::King {
            let cap_nnue = captured_piece.to_nnue();
            let cap_color = (!moving_color).to_nnue();
            state.sub(nnue::Color::White, cap_nnue, cap_color, to_sq);
            state.sub(nnue::Color::Black, cap_nnue, cap_color, to_sq);
        }
    }

    // Handle promotion
    let final_piece = if let Some(promo) = mv.get_promotion() {
        promo.to_nnue()
    } else {
        nnue_piece
    };

    // Add piece to new square (both perspectives)
    state.add(nnue::Color::White, final_piece, nnue_color, to_sq);
    state.add(nnue::Color::Black, final_piece, nnue_color, to_sq);

    true
}

/// Refresh state from board (when incremental update not possible)
/// Note: Due to lifetime constraints, caller should use create_state instead
#[allow(dead_code)]
pub fn refresh_state_inplace<'m>(
    state: &mut SfHalfKpState<'m>,
    board: &Board,
) {
    // Re-initialize using the existing model reference
    let white_king = board.king_square(chess::Color::White).to_nnue();
    let black_king = board.king_square(chess::Color::Black).to_nnue();
    
    // Reset kings
    state.kings = [white_king, black_king.rotate()];
    
    // Reset accumulators to biases
    state.model.transformer.input_layer.empty(&mut state.accumulator[0]);
    state.model.transformer.input_layer.empty(&mut state.accumulator[1]);
    
    // Re-add all pieces
    for &piece in &[chess::Piece::Pawn, chess::Piece::Knight, chess::Piece::Bishop, 
                    chess::Piece::Rook, chess::Piece::Queen] {
        for &color in &[chess::Color::White, chess::Color::Black] {
            let bb = board.pieces(piece) & board.color_combined(color);
            let nnue_piece = piece.to_nnue();
            let nnue_color = color.to_nnue();
            
            for sq in bb {
                let nnue_sq = sq.to_nnue();
                state.add(nnue::Color::White, nnue_piece, nnue_color, nnue_sq);
                state.add(nnue::Color::Black, nnue_piece, nnue_color, nnue_sq);
            }
        }
    }
}
