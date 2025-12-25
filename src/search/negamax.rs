//! Negamax alpha-beta search implementation.
//!
//! This is the core search algorithm with:
//! - Transposition table probing and storing
//! - Alpha-beta pruning
//! - Quiescence search for captures
//!
//! Future extensions: null move pruning, LMR, futility pruning

use super::{Searcher, SearchStats, ordering};
use super::tt::BoundType;
use crate::types::{Board, Move, Score, Depth, Ply, MoveGen};
use crate::eval;

/// Result from a search
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: Score,
    pub pv: Vec<Move>,
    pub stats: SearchStats,
}

/// Main negamax search function with TT integration and null move pruning
pub fn search(
    searcher: &mut Searcher,
    board: &Board,
    depth: Depth,
    ply: Ply,
    mut alpha: Score,
    beta: Score,
    allow_null: bool,
) -> SearchResult {
    searcher.inc_nodes();
    searcher.update_seldepth(ply);

    let orig_alpha = alpha;
    let hash = board.get_hash();
    let mut tt_move: Option<Move> = None;

    // === TT Probe ===
    if let Some(entry) = searcher.tt.probe(hash) {
        tt_move = entry.best_move();
        
        // Only use TT score if depth is sufficient
        if entry.depth() >= depth {
            let tt_score = entry.score().from_tt(ply.raw());
            
            match entry.bound() {
                BoundType::Exact => {
                    return SearchResult {
                        best_move: tt_move,
                        score: tt_score,
                        pv: tt_move.map(|m| vec![m]).unwrap_or_default(),
                        stats: searcher.stats().clone(),
                    };
                }
                BoundType::LowerBound => {
                    if tt_score >= beta {
                        return SearchResult {
                            best_move: tt_move,
                            score: tt_score,
                            pv: tt_move.map(|m| vec![m]).unwrap_or_default(),
                            stats: searcher.stats().clone(),
                        };
                    }
                    if tt_score > alpha {
                        alpha = tt_score;
                    }
                }
                BoundType::UpperBound => {
                    if tt_score <= alpha {
                        return SearchResult {
                            best_move: tt_move,
                            score: tt_score,
                            pv: tt_move.map(|m| vec![m]).unwrap_or_default(),
                            stats: searcher.stats().clone(),
                        };
                    }
                }
                BoundType::None => {}
            }
        }
    }

    // Check for stop condition
    if searcher.should_stop() {
        return SearchResult {
            best_move: None,
            score: Score::draw(),
            pv: Vec::new(),
            stats: searcher.stats().clone(),
        };
    }

    let in_check = *board.checkers() != chess::EMPTY;

    // === Null Move Pruning ===
    // Skip if: in check, depth too low, null move disabled, or only king+pawns
    if allow_null && !in_check && depth.raw() >= 3 {
        // Don't do null move in pure pawn endgames (zugzwang risk)
        let dominated_by_pawns = (board.pieces(chess::Piece::Knight)
            | board.pieces(chess::Piece::Bishop)
            | board.pieces(chess::Piece::Rook)
            | board.pieces(chess::Piece::Queen)).popcnt() == 0;
        
        if !dominated_by_pawns {
            // Reduction: R=3 if depth > 6, else R=2
            let r = if depth.raw() > 6 { 3 } else { 2 };
            
            if let Some(null_board) = board.null_move() {
                let null_result = search(
                    searcher,
                    &null_board,
                    Depth::new((depth.raw() - 1 - r).max(0)),
                    ply.next(),
                    -beta,
                    -beta + Score::cp(1),
                    false,  // Don't allow consecutive null moves
                );
                
                let null_score = -null_result.score;
                
                if null_score >= beta {
                    // Null move cutoff
                    return SearchResult {
                        best_move: None,
                        score: beta,
                        pv: Vec::new(),
                        stats: searcher.stats().clone(),
                    };
                }
            }
        }
    }

    // Generate legal moves
    let mut moves: Vec<Move> = MoveGen::new_legal(board).collect();

    // Check for checkmate or stalemate
    if moves.is_empty() {
        let score = if *board.checkers() != chess::EMPTY {
            Score::mated_in(ply.raw())
        } else {
            Score::draw()
        };
        return SearchResult {
            best_move: None,
            score,
            pv: Vec::new(),
            stats: searcher.stats().clone(),
        };
    }

    // Quiescence search at depth 0
    if depth.is_qs() {
        return quiescence(searcher, board, ply, alpha, beta);
    }

    // Get killers for this ply
    let killers = searcher.killers.get(ply);
    let color = board.side_to_move();

    // Order moves (TT, killers, and history)
    ordering::order_moves_full(board, &mut moves, tt_move, killers, &searcher.history, color);

    let mut best_move = None;
    let mut best_score = Score::neg_infinity();
    let mut pv = Vec::new();
    let mut searched_quiets: Vec<Move> = Vec::new();

    for (move_idx, &m) in moves.iter().enumerate() {
        let new_board = board.make_move_new(m);

        // Prefetch TT entry for next position
        searcher.tt.prefetch(new_board.get_hash());

        // Determine if this is a quiet move (for LMR)
        let is_capture = board.piece_on(m.get_dest()).is_some();
        let is_promotion = m.get_promotion().is_some();
        let is_killer = killers[0] == Some(m) || killers[1] == Some(m);
        let is_quiet = !is_capture && !is_promotion;
        let gives_check = new_board.checkers().popcnt() > 0;

        // LMR: Late Move Reductions
        // Reduce depth for late quiet moves that aren't special
        let mut reduced = false;
        
        // Check extension: extend +1 when in check to avoid horizon effect
        let extension = if in_check { 1 } else { 0 };
        
        let search_depth = if move_idx >= 4 
            && depth.raw() >= 3 
            && is_quiet 
            && !in_check 
            && !gives_check
            && !is_killer
        {
            // Logarithmic reduction formula
            let d = (depth.raw() as f32).ln();
            let m_idx = ((move_idx + 1) as f32).ln();
            let reduction = ((d * m_idx) / 2.5) as i32;
            let reduction = reduction.min(depth.raw() - 2).max(1);
            reduced = true;
            Depth::new((depth.raw() - 1 - reduction + extension).max(1))
        } else {
            Depth::new((depth.raw() - 1 + extension).max(0))
        };

        // Search with potentially reduced depth
        let mut result = search(
            searcher,
            &new_board,
            search_depth,
            ply.next(),
            -beta,
            -alpha,
            true,
        );

        let mut score = -result.score;

        // Re-search at full depth if reduced search beats alpha
        if reduced && score > alpha && !searcher.should_stop() {
            result = search(
                searcher,
                &new_board,
                depth - 1,
                ply.next(),
                -beta,
                -alpha,
                true,
            );
            score = -result.score;
        }

        if searcher.should_stop() {
            break;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(m);

            pv.clear();
            pv.push(m);
            pv.extend(result.pv);

            if score > alpha {
                alpha = score;

                if score >= beta {
                    // Beta cutoff - update killer and history for quiet moves
                    if is_quiet {
                        searcher.killers.store(ply, m);
                        // Update history: bonus for cutoff move, penalty for searched quiets
                        searcher.history.update_on_cutoff(color, m, depth.raw(), &searched_quiets);
                    }
                    break;
                }
            }
        }
        
        // Track searched quiet moves for history penalty
        if is_quiet {
            searched_quiets.push(m);
        }
    }

    // === TT Store ===
    if !searcher.should_stop() {
        let bound = if best_score >= beta {
            BoundType::LowerBound
        } else if best_score > orig_alpha {
            BoundType::Exact
        } else {
            BoundType::UpperBound
        };

        searcher.tt.store(
            hash,
            best_move,
            best_score.to_tt(ply.raw()),
            depth,
            bound,
        );
    }

    SearchResult {
        best_move,
        score: best_score,
        pv,
        stats: searcher.stats().clone(),
    }
}

/// Quiescence search - search captures only to avoid horizon effect
fn quiescence(
    searcher: &mut Searcher,
    board: &Board,
    ply: Ply,
    mut alpha: Score,
    beta: Score,
) -> SearchResult {
    searcher.inc_nodes();
    searcher.update_seldepth(ply);

    // Stand-pat evaluation
    let stand_pat = eval::evaluate(board, searcher.nnue.as_ref());

    if stand_pat >= beta {
        return SearchResult {
            best_move: None,
            score: beta,
            pv: Vec::new(),
            stats: searcher.stats().clone(),
        };
    }

    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Generate capture moves only
    let mut moves: Vec<Move> = MoveGen::new_legal(board)
        .filter(|m| board.piece_on(m.get_dest()).is_some())
        .collect();

    if moves.is_empty() {
        return SearchResult {
            best_move: None,
            score: alpha,
            pv: Vec::new(),
            stats: searcher.stats().clone(),
        };
    }

    ordering::order_captures(board, &mut moves);

    let mut best_score = stand_pat;
    let mut pv = Vec::new();

    for &m in &moves {
        if searcher.should_stop() {
            break;
        }

        let new_board = board.make_move_new(m);

        let result = quiescence(searcher, &new_board, ply.next(), -beta, -alpha);
        let score = -result.score;

        if score > best_score {
            best_score = score;

            pv.clear();
            pv.push(m);
            pv.extend(result.pv);

            if score > alpha {
                alpha = score;
                if score >= beta {
                    break;
                }
            }
        }
    }

    SearchResult {
        best_move: None,
        score: best_score,
        pv,
        stats: searcher.stats().clone(),
    }
}
